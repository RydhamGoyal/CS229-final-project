use anyhow::{Context, Result};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use serde_json::Value;

/// Train a RandomForestClassifier from a JSONL file where each line is a JSON object.
/// Each object must contain a `s_mac` field (string) which is used as the label. The
/// feature vector is taken from the `fts` array if present, otherwise it falls back to
/// extracting a known ordered set of numeric fields.
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: train_model <train.jsonl> <out_model.bin>");
        std::process::exit(2);
    }
    let infile = &args[1];
    let outfile = &args[2];

    let fh = File::open(infile).with_context(|| format!("opening input file {}", infile))?;
    let reader = BufReader::new(fh);

    let mut features_rows: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();
    let mut label_map: HashMap<String, usize> = HashMap::new();

    // Known fallback order if `fts` is not present. This mirrors features_all ordering.
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

    for (ln, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("reading line {}", ln + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&line)
            .with_context(|| format!("parsing JSON on line {}", ln + 1))?;

        // Extract label from s_mac
        let s_mac = match v.get("s_mac") {
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
            None => {
                return Err(anyhow::anyhow!("missing s_mac on line {}", ln + 1));
            }
        };

        // To avoid overlapping mutable/immutable borrows on `label_map`, read the current
        // length first and then perform the `entry` operation.
        let next_id = label_map.len();
        let label_idx = match label_map.entry(s_mac) {
            std::collections::hash_map::Entry::Occupied(o) => *o.get(),
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(next_id);
                next_id
            }
        };

        // Extract features
        let row: Vec<f64> = if let Some(Value::Array(arr)) = v.get("fts") {
            // Use fts array if present
            arr.iter()
                .map(|val| match val {
                    Value::Number(n) => n.as_f64().unwrap_or(0.0),
                    _ => 0.0,
                })
                .collect()
        } else {
            // Fallback: collect values for known keys in fallback_keys order. Missing values -> 0.0
            fallback_keys
                .iter()
                .map(|k| v.get(*k).and_then(|val| val.as_f64()).unwrap_or(0.0))
                .collect()
        };

        // Exclude d_mac: if d_mac was encoded into features somehow, it's ignored here because
        // we only use numeric vectors (fts) or the numeric fallback keys which don't include macs.

        features_rows.push(row);
        labels.push(label_idx);
    }

    if features_rows.is_empty() {
        return Err(anyhow::anyhow!("no training samples found in {}", infile));
    }

    let n_samples = features_rows.len();
    let n_features = features_rows[0].len();

    // Flatten row-major
    let mut flat: Vec<f64> = Vec::with_capacity(n_samples * n_features);
    for r in &features_rows {
        if r.len() != n_features {
            return Err(anyhow::anyhow!("inconsistent feature vector lengths"));
        }
        flat.extend_from_slice(&r);
    }

    // Build DenseMatrix
    let x = DenseMatrix::new(n_samples, n_features, flat, false).context("constructing DenseMatrix")?;

    // Train a RandomForest classifier with default params
    let params = RandomForestClassifierParameters::default();
    let clf = RandomForestClassifier::fit(&x, &labels, params).context("training RandomForest")?;

    // Save serialized classifier
    let mut out = File::create(outfile).with_context(|| format!("creating output file {}", outfile))?;
    bincode::serialize_into(&mut out, &clf).context("serializing classifier")?;

    // Also save a sidecar JSON file that maps class indices back to `s_mac` strings. We store an
    // array where the index is the class id and the value is the corresponding MAC string.
    let mut labels_vec: Vec<String> = vec![String::new(); label_map.len()];
    for (mac, &idx) in &label_map {
        if idx >= labels_vec.len() {
            return Err(anyhow::anyhow!("label index out of range when writing mapping"));
        }
        labels_vec[idx] = mac.clone();
    }
    let labels_fname = format!("{}.labels.json", outfile);
    let lf = File::create(&labels_fname).with_context(|| format!("creating labels file {}", labels_fname))?;
    serde_json::to_writer_pretty(lf, &labels_vec).context("writing labels mapping")?;

    println!("Saved model to {} and labels to {} ({} classes)", outfile, labels_fname, label_map.len());
    Ok(())
}
