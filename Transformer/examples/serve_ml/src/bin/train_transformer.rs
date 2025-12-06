use anyhow::{Context, Result};
use clap::Parser;
use itransformer::ITransformer;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use tch::{nn, nn::VarStore, Device, Kind, Tensor, nn::OptimizerConfig};
use rand::distributions::Distribution;

/// Train an iTransformer + linear classifier for single-step classification.
#[derive(Parser, Debug)]
struct Args {
    /// Input JSONL training file
    #[clap(short, long, parse(from_os_str))]
    train_file: PathBuf,

    /// Output weights file (VarStore save path)
    #[clap(short, long, parse(from_os_str))]
    out_weights: PathBuf,

    /// Output labels sidecar JSON file (e.g. model.weights.ot.labels.json)
    #[clap(short = 'l', long = "labels-out", parse(from_os_str))]
    labels_out: Option<PathBuf>,

    /// Number of epochs
    #[clap(long, default_value = "10")]
    epochs: i64,

    /// Optional initial weights file to load before training (useful for finetuning)
    #[clap(long, parse(from_os_str))]
    init_weights: Option<PathBuf>,

    /// Batch size
    #[clap(long, default_value = "16")]
    batch_size: usize,

    /// Learning rate
    #[clap(long, default_value = "0.001")]
    lr: f64,

    /// num_variates (features). If omitted and lookback_len==1, inferred from first sample.
    #[clap(long)]
    num_variates: Option<i64>,

    /// lookback length (time dimension). Default 1 (single-step features)
    #[clap(long, default_value = "1")]
    lookback_len: i64,

    /// Transformer depth
    #[clap(long, default_value = "6")]
    depth: i64,

    /// Hidden dimension
    #[clap(long, default_value = "256")]
    dim: i64,

    /// Optional num tokens per variate
    #[clap(long)]
    num_tokens_per_variate: Option<i64>,

    /// pred_length as comma-separated list (default "1")
    #[clap(long, default_value = "1")]
    pred_length: String,

    /// Optional dim_head
    #[clap(long)]
    dim_head: Option<i64>,

    /// Optional heads
    #[clap(long)]
    heads: Option<i64>,

    /// Use reversible instance norm
    #[clap(long, takes_value = false)]
    use_reversible_instance_norm: bool,

    /// Use flash attention (where supported)
    #[clap(long, takes_value = false)]
    flash_attn: bool,
    /// Oversample minority classes when forming training batches
    #[clap(long, takes_value = false)]
    oversample: bool,
    /// Use class-weighted cross-entropy (weights = 1 / class_count)
    #[clap(long, takes_value = false)]
    class_weighted: bool,
    /// Use focal loss instead of cross-entropy
    #[clap(long, takes_value = false)]
    focal_loss: bool,
    /// Focal loss gamma parameter
    #[clap(long, default_value = "2.0")]
    focal_gamma: f64,
    /// Apply a power to the inverse-frequency class weight (weight = (total/(ncls*count))^power)
    #[clap(long, default_value = "1.0")]
    class_weight_pow: f64,
    /// Multiply class weights by this scale (for stronger weighting)
    #[clap(long, default_value = "1.0")]
    class_weight_scale: f64,
    /// Use a balanced batch sampler that selects classes uniformly per-sample
    #[clap(long, takes_value = false)]
    balanced_batches: bool,
    /// L2 weight decay applied to classifier parameters (adds wd * sum(params^2) to loss)
    #[clap(long, default_value = "0.0")]
    weight_decay: f64,

    /// Gradient clipping: max norm (0.0 = disabled)
    #[clap(long, default_value = "0.0")]
    clip_grad_norm: f64,

    /// When set, do not update backbone (itransformer) parameters — only update classifier.
    #[clap(long, takes_value = false)]
    freeze_backbone: bool,

    /// Fraction of data to use for validation (0.0 = no validation)
    #[clap(long, default_value = "0.0")]
    val_split: f64,
    /// When set, include validation samples in the training set so each epoch uses
    /// all available samples. Validation metrics are still computed on the holdout
    /// set, but those samples are not excluded from training.
    #[clap(long, takes_value = false)]
    include_val_in_train: bool,
}

fn parse_pred_lengths(s: &str) -> Vec<i64> {
    s.split(',')
        .filter_map(|p| p.trim().parse::<i64>().ok())
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Read training file and collect features + labels
    let fh = File::open(&args.train_file).with_context(|| format!("opening {}", args.train_file.display()))?;
    let reader = BufReader::new(fh);

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

    let mut features_rows: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();
    let mut label_map: HashMap<String, usize> = HashMap::new();

    for (ln, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("reading line {}", ln + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&line).with_context(|| format!("parsing JSON on line {}", ln + 1))?;

        // Extract s_mac label
        let s_mac = match v.get("s_mac") {
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
            None => return Err(anyhow::anyhow!("missing s_mac on line {}", ln + 1)),
        };

        // Map label to id
        let next_id = label_map.len();
        let label_idx = match label_map.entry(s_mac) {
            std::collections::hash_map::Entry::Occupied(o) => *o.get(),
            std::collections::hash_map::Entry::Vacant(v) => { v.insert(next_id); next_id }
        };

        let row: Vec<f64> = if let Some(Value::Array(arr)) = v.get("fts") {
            arr.iter().map(|val| match val { Value::Number(n) => n.as_f64().unwrap_or(0.0), _ => 0.0 }).collect()
        } else {
            fallback_keys.iter().map(|k| v.get(*k).and_then(|val| val.as_f64()).unwrap_or(0.0)).collect()
        };

        features_rows.push(row);
        labels.push(label_idx);
    }

    if features_rows.is_empty() {
        return Err(anyhow::anyhow!("no training samples found"));
    }

    let n_samples = features_rows.len();
    // Determine num_variates: either provided or inferred as row.len() / lookback_len
    let inferred_len = features_rows[0].len();
    let num_variates = args.num_variates.unwrap_or_else(|| {
        if args.lookback_len == 0 { 1 } else { (inferred_len as i64 / args.lookback_len) }
    });

    // Flatten feature data into f32
    let mut flat: Vec<f32> = Vec::with_capacity(n_samples * inferred_len);
    for r in &features_rows {
        if r.len() != inferred_len { return Err(anyhow::anyhow!("inconsistent feature vector lengths")); }
        for v in r { flat.push(*v as f32); }
    }

    // Build tensors
    let xs_all = Tensor::f_from_slice(&flat)?.reshape(&[n_samples as i64, args.lookback_len, num_variates]);
    let ys_vec = labels.iter().map(|&i| i as i64).collect::<Vec<_>>();
    let ys_all = Tensor::f_from_slice(&ys_vec)?.to_kind(Kind::Int64);

    // Create VarStore, model, and classifier
    let device = Device::Cpu;
    let mut vs = VarStore::new(device);

    let pred_lengths = parse_pred_lengths(&args.pred_length);
    let model = ITransformer::new(
        &(vs.root() / "itransformer"),
        num_variates,
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

    // Determine classifier input dim by running a single forward pass to inspect shape
    let sample_xs = xs_all.narrow(0, 0, 1).to_device(device);
    let sample_preds = model.forward(&sample_xs, None, false)?;
    let sample_preds = match sample_preds {
        either::Either::Left(v) => v,
        either::Either::Right(_) => return Err(anyhow::anyhow!("model returned loss when expecting predictions")),
    };
    if sample_preds.is_empty() { return Err(anyhow::anyhow!("model produced no prediction heads")); }
    let (_pl, sample_tensor) = &sample_preds[0];
    let feat_dim = sample_tensor.size().iter().skip(1).product::<i64>();

    let num_classes = label_map.len() as i64;
    let classifier_cfg = nn::LinearConfig { bias: true, ..Default::default() };

    // Handle optional init weights where the saved classifier head may have a different
    // number of classes (e.g., when finetuning on a balanced subset). Strategy:
    // - If `--init-weights` is provided and a labels sidecar exists at
    //   `<init>.labels.json`, read its length (saved_num_classes).
    // - If saved_num_classes != current num_classes, register a temporary loader
    //   classifier matching saved_num_classes at the name "classifier" so the
    //   checkpoint can be loaded without size mismatch. After loading, register
    //   the actual training classifier under `classifier_head` with the current
    //   `num_classes` and use that for training. This leaves the loaded checkpoint
    //   intact for backbone weights while allowing a different-sized head.

    let mut classifier: nn::Linear;
    if let Some(ref init) = args.init_weights {
        // attempt to find labels sidecar next to init weights
        let labels_sidecar = PathBuf::from(format!("{}.labels.json", init.display()));
        let mut saved_num_classes: Option<i64> = None;
        if labels_sidecar.exists() {
            let s = std::fs::read_to_string(&labels_sidecar)
                .with_context(|| format!("reading labels sidecar {}", labels_sidecar.display()))?;
            let v: Vec<String> = serde_json::from_str(&s).with_context(|| format!("parsing labels sidecar {}", labels_sidecar.display()))?;
            saved_num_classes = Some(v.len() as i64);
        }

        if let Some(saved) = saved_num_classes {
            if saved != num_classes {
                // register a temporary loader classifier that matches the saved checkpoint
                let _loader_classifier = nn::linear(&(vs.root() / "classifier"), feat_dim, saved, classifier_cfg);
                // load checkpoint (backbone + loader head will be loaded)
                vs.load(init.to_str().unwrap()).with_context(|| format!("loading init weights {}", init.display()))?;
                println!("Loaded initial weights from {} (head size {}), will create new head with {} classes", init.display(), saved, num_classes);
                // now create the actual training classifier under a different name
                classifier = nn::linear(&(vs.root() / "classifier_head"), feat_dim, num_classes, classifier_cfg);
            } else {
                // same size: create classifier at expected name and load
                classifier = nn::linear(&(vs.root() / "classifier"), feat_dim, num_classes, classifier_cfg);
                vs.load(init.to_str().unwrap()).with_context(|| format!("loading init weights {}", init.display()))?;
                println!("Loaded initial weights from {} (head size {})", init.display(), saved);
            }
        } else {
            // no labels sidecar found; fall back to creating classifier for current classes
            classifier = nn::linear(&(vs.root() / "classifier"), feat_dim, num_classes, classifier_cfg);
            // try loading; if checkpoint mismatches, this may return an error which bubble up
            vs.load(init.to_str().unwrap()).with_context(|| format!("loading init weights {}", init.display()))?;
            println!("Loaded initial weights from {}", init.display());
        }
    } else {
        // no init weights: create classifier normally
        classifier = nn::linear(&(vs.root() / "classifier"), feat_dim, num_classes, classifier_cfg);
    }

    // Optimizer
    let mut opt = nn::Adam::default().build(&vs, args.lr)?;

    // Compute class counts for the whole dataset
    let mut class_counts: Vec<usize> = vec![0; label_map.len()];
    for &lab in &labels { class_counts[lab] += 1; }

    // Optional class-weight vector (for weighted loss)
    let mut class_weight_tensor: Option<Tensor> = None;
    if args.class_weighted {
        let total: f64 = class_counts.iter().sum::<usize>() as f64;
        let ncls = class_counts.len() as f64;
        let mut cw: Vec<f32> = Vec::with_capacity(class_counts.len());
        for &c in &class_counts {
            let base = if c == 0 { 1.0 } else { total / (ncls * (c as f64)) };
            // apply power and scaling to allow stronger or smoother weighting
            let w = base.powf(args.class_weight_pow) * args.class_weight_scale;
            cw.push(w as f32);
        }
        class_weight_tensor = Some(Tensor::f_from_slice(&cw)?.to_device(device).to_kind(Kind::Float));
    }

    // Create train/validation split indices if requested
    let mut all_idxs: Vec<usize> = (0..n_samples).collect();
    let mut train_idxs: Vec<usize> = all_idxs.clone();
    let mut val_idxs: Vec<usize> = Vec::new();
    if args.val_split > 0.0 && args.val_split < 1.0 {
        use rand::seq::SliceRandom;
        use rand::thread_rng;
        all_idxs.shuffle(&mut thread_rng());
        let split_at = ((1.0 - args.val_split) * (n_samples as f64)).round() as usize;
        // Keep a separate validation index set (holdout). If `include_val_in_train`
        // is set, we still compute validation metrics on `val_idxs` but do not
        // remove those samples from the training set (i.e., training will use
        // the full `all_idxs`). This allows using all available samples each
        // epoch while preserving a held-out validation evaluation.
        val_idxs = all_idxs[split_at..].to_vec();
        if args.include_val_in_train {
            train_idxs = all_idxs.clone();
        } else {
            train_idxs = all_idxs[..split_at].to_vec();
        }
    }

    // Prepare oversampling weights if requested (built over the training set)
    let mut weighted_index = None;
    if args.oversample {
        use rand::distributions::WeightedIndex;
        // sample weight per sample is inverse frequency (computed on train set)
        let mut sample_weights: Vec<f64> = Vec::with_capacity(train_idxs.len());
        for &i in &train_idxs {
            let lab = labels[i];
            let cnt = class_counts[lab];
            let w = if cnt == 0 { 1.0 } else { 1.0 / (cnt as f64) };
            sample_weights.push(w);
        }
        // create WeightedIndex (indices will refer to positions inside `train_idxs`)
        let dist = WeightedIndex::new(&sample_weights).map_err(|e| anyhow::anyhow!(format!("creating WeightedIndex: {}", e)))?;
        weighted_index = Some(dist);
    }

    // Build per-class index lists for balanced batch sampling
    let mut per_class_idxs: Vec<Vec<usize>> = vec![Vec::new(); class_counts.len()];
    for (i, &lab) in labels.iter().enumerate() {
        per_class_idxs[lab].push(i);
    }

    // Training loop (simple, no advanced batching features)

    let mut idxs: Vec<usize> = train_idxs.clone();
    for epoch in 1..=args.epochs {
        let mut epoch_loss = 0.0f64;
        let mut seen = 0usize;

        // If not oversampling, shuffle the training indices once per epoch
        if weighted_index.is_none() {
            use rand::seq::SliceRandom;
            use rand::thread_rng;
            idxs.shuffle(&mut thread_rng());
        }

        let train_n = idxs.len();
        for batch_start in (0..train_n).step_by(args.batch_size) {
            // select batch indices according to sampling strategy:
            // - if `balanced_batches` -> sample classes uniformly and pick one sample per class
            // - else if `oversample` -> sample with replacement using weighted_index
            // - else -> take contiguous slice from shuffled idxs
            let mut batch_idxs: Vec<usize> = Vec::with_capacity(args.batch_size);
            if args.balanced_batches {
                use rand::seq::SliceRandom;
                use rand::thread_rng;
                let mut rng = thread_rng();
                // sample classes uniformly for each sample slot in the batch
                for _ in 0..args.batch_size {
                    // choose a random class among those with at least one sample
                    let mut cls = None;
                    // try a few times to pick a non-empty class
                    for _ in 0..5 {
                        let c = (rand::random::<usize>() % per_class_idxs.len()) as usize;
                        if !per_class_idxs[c].is_empty() { cls = Some(c); break; }
                    }
                    let c = match cls {
                        Some(x) => x,
                        None => { // fallback to random from train_idxs
                            let &ix = train_idxs.choose(&mut rng).unwrap();
                            batch_idxs.push(ix);
                            continue;
                        }
                    };
                    // pick a random sample from class c
                    let &chosen = per_class_idxs[c].choose(&mut rng).unwrap();
                    batch_idxs.push(chosen);
                }
            } else if let Some(ref dist) = weighted_index {
                use rand::thread_rng;
                let mut rng = thread_rng();
                let bs_usize = std::cmp::min(args.batch_size, train_n);
                for _ in 0..bs_usize { let pos = dist.sample(&mut rng); batch_idxs.push(train_idxs[pos]); }
            } else {
                let end = std::cmp::min(train_n, batch_start + args.batch_size);
                batch_idxs.extend_from_slice(&idxs[batch_start..end]);
            }
            let bs = batch_idxs.len() as i64;

            // gather batch xs and ys
            let mut batch_x_vec: Vec<f32> = Vec::with_capacity(batch_idxs.len() * inferred_len);
            let mut batch_y_vec: Vec<i64> = Vec::with_capacity(batch_idxs.len());
            for &i in &batch_idxs {
                for v in &features_rows[i] { batch_x_vec.push(*v as f32); }
                batch_y_vec.push(labels[i] as i64);
            }

            let xs = Tensor::f_from_slice(&batch_x_vec)?.reshape(&[bs, args.lookback_len, num_variates]).to_device(device);
            let ys = Tensor::f_from_slice(&batch_y_vec)?.to_kind(Kind::Int64).to_device(device);

            // forward through transformer
            let preds_either = model.forward(&xs, None, true)?;
            let preds_vec = match preds_either {
                either::Either::Left(v) => v,
                either::Either::Right(_) => return Err(anyhow::anyhow!("unexpected loss returned from model forward during training")),
            };
            let (_pl, t) = preds_vec.into_iter().next().expect("no preds");
            let features = t.reshape(&[bs, -1]);

            // classifier forward
            let logits = features.apply(&classifier);

            // loss: either standard cross-entropy, class-weighted, or focal
            let loss = if args.focal_loss {
                // numerically stable focal loss: clamp p_t to [eps, 1-eps]
                // focal loss: -(1 - p_t)^gamma * log(p_t)
                let probs = logits.softmax(-1, Kind::Float);
                let pt = probs.gather(1, &ys.unsqueeze(1), false).view([bs]);
                let eps = 1e-7_f64;
                let pt_clamped = pt.clamp(eps, 1.0 - eps);
                let logp = pt_clamped.log();
                let gamma = args.focal_gamma;
                let focal_factor = (Tensor::ones(&[bs], (Kind::Float, device)) - &pt_clamped).pow_tensor_scalar(gamma);
                let focal = &focal_factor * (-&logp);
                if let Some(ref cw) = class_weight_tensor {
                    let sample_w = cw.index_select(0, &ys);
                    (&sample_w * focal).mean(Kind::Float)
                } else {
                    focal.mean(Kind::Float)
                }
            } else if let Some(ref cw) = class_weight_tensor {
                // compute log-probs and gather per-target log-probabilities
                let logp = logits.log_softmax(-1, Kind::Float).gather(1, &ys.unsqueeze(1), false).view([bs]);
                // per-sample weights from class weight tensor
                let sample_w = cw.index_select(0, &ys);
                let batch_loss = -(sample_w * logp).mean(Kind::Float);
                batch_loss
            } else {
                logits.cross_entropy_for_logits(&ys)
            };

            // Apply L2 weight decay to classifier parameters if requested.
            let loss = if args.weight_decay > 0.0 {
                // classifier.ws always exists; classifier.bs is Option<Tensor>
                let mut reg = classifier.ws.shallow_clone().pow_tensor_scalar(2.0).sum(Kind::Float);
                if let Some(ref b) = classifier.bs.as_ref() {
                    reg = reg + b.shallow_clone().pow_tensor_scalar(2.0).sum(Kind::Float);
                }
                let wd_t = Tensor::from(args.weight_decay).to_device(device).to_kind(Kind::Float);
                loss + wd_t * reg
            } else { loss };

            // Perform backward + optional gradient clipping + optimizer step.
            // Use explicit zero_grad / backward / step so we can clip classifier gradients
            opt.zero_grad();
            loss.backward();
            // If requested, zero-out backbone (itransformer) gradients so they are not updated.
            if args.freeze_backbone {
                // vs.variables() returns a HashMap<String, Tensor> of registered params
                for (name, tensor) in vs.variables() {
                    if name.starts_with("itransformer") {
                        let mut g = tensor.grad();
                        if g.defined() {
                            g.zero_();
                        }
                    }
                }
            }
            if args.clip_grad_norm > 0.0 {
                // Compute L2 norm over classifier gradients (ws and optional bs)
                let mut total_norm = 0.0_f64;
                let g = classifier.ws.grad();
                if g.defined() {
                    let n = g.pow_tensor_scalar(2.0).sum(Kind::Float).double_value(&[]);
                    total_norm += n;
                }
                if let Some(ref b) = classifier.bs.as_ref() {
                    let gb = b.grad();
                    if gb.defined() {
                        let nb = gb.pow_tensor_scalar(2.0).sum(Kind::Float).double_value(&[]);
                        total_norm += nb;
                    }
                }
                total_norm = total_norm.sqrt();
                let max_norm = args.clip_grad_norm;
                if total_norm > max_norm && total_norm > 0.0 {
                    let clip_coef = (max_norm / (total_norm + 1e-6)) as f64;
                    // scale gradients in-place
                    let mut g2 = classifier.ws.grad();
                    if g2.defined() { g2.copy_(&(&g2 * clip_coef)); }
                    if let Some(ref b) = classifier.bs.as_ref() {
                        let mut gb2 = b.grad();
                        if gb2.defined() { gb2.copy_(&(&gb2 * clip_coef)); }
                    }
                }
            }

            opt.step();

            epoch_loss += loss.double_value(&[]);
            seen += bs as usize;
        }

        println!("epoch {}/{} loss={:.6}", epoch, args.epochs, epoch_loss / (seen as f64));
        // Validation pass if requested
        if !val_idxs.is_empty() {
            let mut correct = 0usize;
            let mut total = 0usize;
            let val_n = val_idxs.len();
            for vs_start in (0..val_n).step_by(args.batch_size) {
                let end = std::cmp::min(val_n, vs_start + args.batch_size);
                let mut bx: Vec<f32> = Vec::with_capacity((end - vs_start) * inferred_len);
                let mut by: Vec<i64> = Vec::with_capacity(end - vs_start);
                for &i in &val_idxs[vs_start..end] {
                    for v in &features_rows[i] { bx.push(*v as f32); }
                    by.push(labels[i] as i64);
                }
                let bbs = (end - vs_start) as i64;
                let xs = Tensor::f_from_slice(&bx)?.reshape(&[bbs, args.lookback_len, num_variates]).to_device(device);
                let ys = Tensor::f_from_slice(&by)?.to_kind(Kind::Int64).to_device(device);
                let preds_either = model.forward(&xs, None, false)?;
                let preds_vec = match preds_either { either::Either::Left(v) => v, either::Either::Right(_) => Vec::new(), };
                if preds_vec.is_empty() { continue; }
                let (_pl, t) = &preds_vec[0];
                let features = t.reshape(&[bbs, -1]);
                let logits = features.apply(&classifier);
                let (_vals, idxs_t) = logits.max_dim(-1, false);
                let idxs_cpu = idxs_t.to_device(Device::Cpu);
                for i in 0..bbs as i64 {
                    let pred = idxs_cpu.int64_value(&[i]) as usize;
                    let truev = by[i as usize] as usize;
                    if pred == truev { correct += 1; }
                    total += 1;
                }
            }
            let acc = if total == 0 { 0.0 } else { (correct as f64) / (total as f64) };
            println!("validation acc={:.4} ({} samples)", acc, total);
        }
    }

    // Save weights (VarStore)
    vs.save(args.out_weights.to_str().unwrap()).context("saving weights")?;

    // Write labels sidecar
    let labels_fname = match args.labels_out {
        Some(p) => p,
        None => PathBuf::from(format!("{}.labels.json", args.out_weights.display())),
    };
    let mut labels_vec: Vec<String> = vec![String::new(); label_map.len()];
    for (mac, &idx) in &label_map {
        if idx >= labels_vec.len() { return Err(anyhow::anyhow!("label index out of range when writing mapping")); }
        labels_vec[idx] = mac.clone();
    }
    let lf = File::create(&labels_fname).context("creating labels file")?;
    serde_json::to_writer_pretty(lf, &labels_vec).context("writing labels mapping")?;

    // Write metadata (hyperparameters) so runner can auto-load later
    let meta = serde_json::json!({
        "num_variates": num_variates,
        "lookback_len": args.lookback_len,
        "depth": args.depth,
        "dim": args.dim,
        "num_tokens_per_variate": args.num_tokens_per_variate,
        "pred_length": parse_pred_lengths(&args.pred_length),
        "dim_head": args.dim_head,
        "heads": args.heads,
        "use_reversible_instance_norm": args.use_reversible_instance_norm,
        "flash_attn": args.flash_attn
    });
    let meta_fname = format!("{}.meta.json", args.out_weights.display());
    let mf = File::create(&meta_fname).context("creating metadata file")?;
    serde_json::to_writer_pretty(mf, &meta).context("writing metadata")?;

    println!("Saved weights to {} and labels to {}", args.out_weights.display(), labels_fname.display());

    Ok(())
}
