use anyhow::Result;
use clap::Parser;
use either::Either;
use itransformer::ITransformer;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use tch::{nn, nn::VarStore, Device, Tensor, Kind, IndexOp};

/// Inference binary for an iTransformer model using `itransformer` + `tch`.
///
/// Notes:
/// - The program expects a `VarStore` weights file (saved via `VarStore::save`) and the
///   constructor hyperparameters for recreating the `ITransformer` structure. Those
///   hyperparameters can be supplied via CLI flags. This keeps the code generic and
///   avoids assuming a particular training metadata format.
#[derive(Parser, Debug)]
struct Args {
    /// Path to the VarStore weights file (tch saved weights)
    #[clap(short = 'w', long = "weights", parse(from_os_str))]
    weights_file: PathBuf,

    /// Optional labels sidecar JSON file mapping numeric ids -> s_mac strings
    #[clap(short = 'l', long = "labels-file", parse(from_os_str))]
    labels_file: Option<PathBuf>,

    /// Input JSONL test file
    #[clap(short, long, parse(from_os_str))]
    test_file: PathBuf,

    /// Number of variates (features). Example: 17
    #[clap(long)]
    num_variates: i64,

    /// Lookback length (time dimension). For static feature vectors use `1`.
    #[clap(long, default_value = "1")]
    lookback_len: i64,

    /// Transformer depth (number of layers)
    #[clap(long, default_value = "6")]
    depth: i64,

    /// Hidden dimension
    #[clap(long, default_value = "256")]
    dim: i64,

    /// Optional number tokens per variate (pass empty for None)
    #[clap(long)]
    num_tokens_per_variate: Option<i64>,

    /// Prediction lengths as comma-separated integers (e.g. "1" or "1,2")
    #[clap(long, default_value = "1")]
    pred_length: String,

    /// Optional dimensionality per head
    #[clap(long)]
    dim_head: Option<i64>,

    /// Optional number of heads
    #[clap(long)]
    heads: Option<i64>,

    /// Use reversible instance norm
    #[clap(long, takes_value = false)]
    use_reversible_instance_norm: bool,

    /// Whether to enable flash attention (where supported)
    #[clap(long, takes_value = false)]
    flash_attn: bool,

    /// Print classifier weight norm after loading weights (debug)
    #[clap(long, takes_value = false)]
    debug_print_weights: bool,
    /// Dump logits and top-k predictions as JSON lines instead of decoded label
    #[clap(long, takes_value = false)]
    dump_logits: bool,
    /// Top-k to include when dumping logits
    #[clap(long, default_value = "5")]
    topk: i64,
}

fn parse_pred_lengths(s: &str) -> Vec<i64> {
    s.split(',')
        .filter_map(|p| p.trim().parse::<i64>().ok())
        .collect()
}

fn tensor_to_f64_vec(t: &Tensor) -> Vec<f64> {
    // Convert to CPU, double and read element-wise (fallback safe method).
    let t = t.to_device(Device::Cpu).to_kind(Kind::Double).flatten(0, -1);
    let len = t.size()[0];
    let mut out = Vec::with_capacity(len as usize);
    for i in 0..len {
        let v = t.double_value(&[i]);
        out.push(v);
    }
    out
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load labels sidecar if present
    let labels_path = args
        .labels_file
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("{}.labels.json", args.weights_file.display())));
    let labels: Option<Vec<String>> = match std::fs::read_to_string(&labels_path) {
        Ok(s) => serde_json::from_str(&s).ok(),
        Err(_) => None,
    };

    // Try to load additive per-class logit offsets from a sidecar next to the
    // weights file. The sidecar may contain either an object mapping label->value
    // or an array under the key `bias` where the array order matches the labels
    // sidecar. If present we map label->bias for quick lookup during inference.
    let bias_path = PathBuf::from(format!("{}.additive_bias.json", args.weights_file.display()));
    let mut additive_bias_map: Option<HashMap<String, f64>> = None;
    if bias_path.exists() {
        if let Ok(s) = std::fs::read_to_string(&bias_path) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                // Case 1: bias is an array -> map by labels order (requires labels)
                if let Some(arr) = v.get("bias").and_then(|b| b.as_array()) {
                    if let Some(ref lbls) = labels {
                        let mut map = HashMap::new();
                        for (i, val) in arr.iter().enumerate() {
                            if let Some(num) = val.as_f64() {
                                let key = lbls.get(i).cloned().unwrap_or_else(|| format!("unknown_{}", i));
                                map.insert(key, num);
                            }
                        }
                        additive_bias_map = Some(map);
                    }
                } else if let Some(obj) = v.get("bias").and_then(|b| b.as_object()) {
                    // Case 2: bias is an object mapping label->value
                    let mut map = HashMap::new();
                    for (k, va) in obj.iter() {
                        if let Some(num) = va.as_f64() {
                            map.insert(k.clone(), num);
                        }
                    }
                    additive_bias_map = Some(map);
                }
            }
        }
    }

    // Build VarStore
    let device = Device::Cpu;
    let mut vs = VarStore::new(device);

    // Parse pred lengths
    let pred_lengths = parse_pred_lengths(&args.pred_length);

    // Construct model (registers variables in VarStore)
    let model = ITransformer::new(
        &(vs.root() / "itransformer"),
        args.num_variates,
        args.lookback_len,
        args.depth,
        args.dim,
        args.num_tokens_per_variate,
        pred_lengths.clone(),
        args.dim_head,
        args.heads,
        None,
        None,
        None,
        None,
        Some(args.use_reversible_instance_norm),
        None,
        args.flash_attn,
        &device,
    )?;

    // Determine if the saved weights have a labels sidecar (may differ from the
    // runtime `labels_file` passed in). If the checkpoint head size differs from
    // the runtime labels, we register a temporary loader classifier with the
    // checkpoint head size (named `classifier`) so `vs.load` succeeds, and also
    // register an actual inference classifier (named `classifier_head`) for the
    // current runtime label set.
    let mut classifier_opt: Option<nn::Linear> = None;
    // try read labels from the weights file sidecar (e.g. weights.ot.labels.json)
    let weights_labels_path = PathBuf::from(format!("{}.labels.json", args.weights_file.display()));
    let weights_labels: Option<Vec<String>> = match std::fs::read_to_string(&weights_labels_path) {
        Ok(s) => serde_json::from_str(&s).ok(),
        Err(_) => None,
    };
    // quick binary scan of the weights file to detect if it contains both
    // `classifier|weight` and `classifier_head|weight` keys; if so prefer to
    // register only `classifier_head` to avoid accidental mismatch copying into
    // a runtime-registered `classifier` variable.
    let weights_blob = std::fs::read(&args.weights_file).ok();
    let file_has_classifier = weights_blob.as_ref().map(|b| b.windows(b"classifier|weight".len()).any(|w| w==b"classifier|weight")).unwrap_or(false);
    let file_has_classifier_head = weights_blob.as_ref().map(|b| b.windows(b"classifier_head|weight".len()).any(|w| w==b"classifier_head|weight")).unwrap_or(false);
    let prefer_classifier_head_only = file_has_classifier && file_has_classifier_head;

    if let Some(ref lbls) = labels {
        // dummy input to get pred feature shape
        let sample_xs = Tensor::zeros(&[1, args.lookback_len, args.num_variates], (Kind::Float, device));
        let sample_preds = model.forward(&sample_xs, None, false)?;
        let sample_preds = match sample_preds {
            Either::Left(v) => v,
            Either::Right(_) => Vec::new(),
        };
        if !sample_preds.is_empty() {
            let (_pl, sample_tensor) = &sample_preds[0];
            let feat_dim = sample_tensor.size().iter().skip(1).product::<i64>();

            let runtime_num_classes = lbls.len() as i64;
            // If the weights sidecar exists and reports a different size, register a
            // loader classifier with the checkpoint size under the name `classifier`.
            // Also register our runtime classifier under `classifier_head` for decoding.
            if prefer_classifier_head_only {
                // register only classifier_head to match checkpoint's finer naming
                let classifier_cfg = nn::LinearConfig { bias: true, ..Default::default() };
                classifier_opt = Some(nn::linear(&(vs.root() / "classifier_head"), feat_dim, runtime_num_classes, classifier_cfg));
            } else if let Some(ref wlbls) = weights_labels {
                let saved_num_classes = wlbls.len() as i64;
                if saved_num_classes != runtime_num_classes {
                    let classifier_cfg = nn::LinearConfig { bias: true, ..Default::default() };
                    // temporary loader with checkpoint size (must match checkpoint keys)
                    let _loader = nn::linear(&(vs.root() / "classifier"), feat_dim, saved_num_classes, classifier_cfg.clone());
                    // real runtime classifier for decoding
                    classifier_opt = Some(nn::linear(&(vs.root() / "classifier_head"), feat_dim, runtime_num_classes, classifier_cfg));
                } else {
                    let classifier_cfg = nn::LinearConfig { bias: true, ..Default::default() };
                    classifier_opt = Some(nn::linear(&(vs.root() / "classifier"), feat_dim, runtime_num_classes, classifier_cfg));
                }
            } else {
                // no weights sidecar — default to registering classifier matching runtime labels
                let classifier_cfg = nn::LinearConfig { bias: true, ..Default::default() };
                classifier_opt = Some(nn::linear(&(vs.root() / "classifier"), feat_dim, runtime_num_classes, classifier_cfg));
            }
        }
    }

    // Debug: list registered variable names/shapes before loading weights
    eprintln!("Registered vars before load:");
    for (name, tensor) in vs.variables() {
        let s: Vec<i64> = tensor.size();
        eprintln!("  {} -> {:?}", name, s);
    }

    // Now that model and classifier shapes are registered (including a temporary
    // loader head if needed), load the saved weights
    match vs.load(args.weights_file.to_str().unwrap()) {
        Ok(_) => {}
        Err(e) => {
            // If load fails, present the error (but continue where possible)
            eprintln!("warning: vs.load failed: {}", e);
            return Err(e.into());
        }
    }

    // Debug: print classifier weight norm if requested
    if args.debug_print_weights {
        if let Some(classifier) = classifier_opt.as_ref() {
            let weight = &classifier.ws;
            let w = weight.to_device(Device::Cpu);
            let sq = &w * &w;
            let sum = sq.sum(Kind::Double);
            let norm = sum.sqrt().double_value(&[]);
            eprintln!("classifier.weight.norm = {}", norm);
        } else {
            eprintln!("no classifier registered (no labels sidecar or unable to infer feature dim)");
        }
    }

    // Fallback keys (same order used elsewhere in the repo)
    let fallback_keys = vec![
        "dur",
        "proto",
        "s_bytes_sum",
        "d_bytes_sum",
        "s_ttl_mean",
        "d_ttl_mean",
        "s_load",
        "d_load",
        "s_pkt_cnt",
        "d_pkt_cnt",
        "s_bytes_mean",
        "d_bytes_mean",
        "s_iat_mean",
        "d_iat_mean",
        "tcp_rtt",
        "syn_ack",
        "ack_dat",
    ];

    let fh = File::open(&args.test_file)?;
    let reader = BufReader::new(fh);

    for (ln, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&line)
            .map_err(|e| anyhow::anyhow!("parsing JSON on line {}: {}", ln + 1, e))?;

        let row: Vec<f64> = if let Some(Value::Array(arr)) = v.get("fts") {
            arr.iter()
                .map(|val| match val {
                    Value::Number(n) => n.as_f64().unwrap_or(0.0),
                    _ => 0.0,
                })
                .collect()
        } else {
            fallback_keys
                .iter()
                .map(|k| v.get(*k).and_then(|val| val.as_f64()).unwrap_or(0.0))
                .collect()
        };

        // Convert row into tensor of shape [batch=1, lookback_len, num_variates]
        // If the row length equals num_variates * lookback_len, reshape accordingly.
        let expected_len = (args.num_variates * args.lookback_len) as usize;
        if row.len() != expected_len {
            // Try to handle single-step features: if row.len() == num_variates and lookback_len==1
            if args.lookback_len == 1 && row.len() == args.num_variates as usize {
                // ok
            } else {
                eprintln!("warning: feature length {} != expected {} for sample {}", row.len(), expected_len, ln+1);
            }
        }

        let mut data_f32: Vec<f32> = row.iter().map(|x| *x as f32).collect();
        // If row length == num_variates (and lookback_len == 1), reshape to [1,1,num_variates]
        let xs = if data_f32.len() == args.num_variates as usize && args.lookback_len == 1 {
            Tensor::f_from_slice(&data_f32)?.reshape(&[1, 1, args.num_variates]).to_device(device)
        } else {
            Tensor::f_from_slice(&data_f32)?.reshape(&[1, args.lookback_len, args.num_variates]).to_device(device)
        };

        // Call forward (inference) -> Either<Vec<(i64, Tensor)>, f64>
        let preds_either = model.forward(&xs, None, false)?;

        match preds_either {
            Either::Right(loss) => {
                // Training loss returned unexpectedly during inference
                println!("{{\"loss\": {} }}", loss);
            }
            Either::Left(vec_preds) => {
                // vec_preds is Vec<(pred_length, Tensor)>. We'll try to interpret the first head.
                if vec_preds.is_empty() {
                    println!("null");
                    continue;
                }
                let (_pl, tensor) = &vec_preds[0];
                // tensor shape for a batch is typically [batch, pred_length, output_dim]
                let shape = tensor.size();
                // For batch==1
                if shape.len() >= 2 && shape[0] == 1 {
                    // If we have a classifier registered (labels present), apply it to the
                    // flattened transformer features and decode the argmax.
                    if let Some(ref lbls) = labels {
                        if let Some(ref classifier) = classifier_opt {
                            let features = tensor.reshape(&[1, -1]);
                            let mut logits = features.apply(classifier);

                            // Apply additive per-class bias if a sidecar was found. We build a
                            // bias vector in the same label order and add it to the logits.
                            if let Some(ref bias_map) = additive_bias_map.as_ref() {
                                let mut bias_vec: Vec<f32> = Vec::with_capacity(lbls.len());
                                for lbl in lbls.iter() {
                                    let v = *bias_map.get(lbl).unwrap_or(&0.0) as f32;
                                    bias_vec.push(v);
                                }
                                let bias_t = Tensor::f_from_slice(&bias_vec)?.to_device(logits.device());
                                logits = &logits + &bias_t;
                            }

                            if args.dump_logits {
                                // compute softmax probs and top-k
                                let probs = logits.softmax(-1, Kind::Float).to_device(Device::Cpu);
                                let (vals, idxs) = probs.topk(args.topk, -1, true, true);
                                let vals_v = tensor_to_f64_vec(&vals);
                                let idxs_v = tensor_to_f64_vec(&idxs.to_kind(Kind::Double));
                                // build topk list of (label,prob)
                                let mut topk_out: Vec<(String, f64)> = Vec::new();
                                for i in 0..idxs_v.len() {
                                    let idx = idxs_v[i] as usize;
                                    let prob = vals_v[i];
                                    let lbl = lbls.get(idx).cloned().unwrap_or_else(|| format!("unknown_{}", idx));
                                    topk_out.push((lbl, prob));
                                }
                                // try to extract true label from parsed JSON (if present)
                                let true_label: Option<String> = v.get("s_mac").and_then(|sv| sv.as_str().map(|s| s.to_string()));
                                let out_obj = serde_json::json!({"true": true_label, "topk": topk_out});
                                println!("{}", serde_json::to_string(&out_obj)?);
                                continue;
                            }
                            let (_vals, idxs) = logits.max_dim(-1, false);
                            let idxs = idxs.to_device(Device::Cpu);
                            let idx = if idxs.numel() == 1 { idxs.int64_value(&[]) } else { idxs.int64_value(&[0]) };
                            let class_idx = idx as usize;
                            let decoded = lbls.get(class_idx).cloned().unwrap_or_else(|| format!("unknown_{}", class_idx));
                            println!("{}", serde_json::to_string(&decoded)?);
                            continue;
                        }
                    }

                    // Otherwise, print raw prediction numbers as JSON array
                    let out_vec = tensor_to_f64_vec(tensor);
                    println!("{}", serde_json::to_string(&out_vec)?);
                } else {
                    // Fallback: print flattened tensor
                    let out_vec = tensor_to_f64_vec(tensor);
                    println!("{}", serde_json::to_string(&out_vec)?);
                }
            }
        }
    }

    Ok(())
}
