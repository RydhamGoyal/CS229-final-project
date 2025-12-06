use anyhow::Result;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use clap::Parser;
use serde_json::Value;

/// Small helper to run a bincode-serialized SmartCore RandomForest model against a JSONL
/// file of test samples. Each input line should be a JSON object containing either an `fts`
/// array (numeric features) or the known fallback numeric keys. The program prints one
/// JSON value per line to stdout: the decoded predicted `s_mac` string (or `unknown_<id>`).
#[derive(Parser, Debug)]
struct Args {
    /// Path to the bincode-serialized model file
    #[clap(short, long, parse(from_os_str))]
    model_file: PathBuf,

    /// Optional labels sidecar JSON file mapping numeric ids -> s_mac strings
    #[clap(short = 'l', long = "labels-file", parse(from_os_str))]
    labels_file: Option<PathBuf>,

    /// Input JSONL test file
    #[clap(short, long, parse(from_os_str))]
    test_file: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load classifier
    let mut f = File::open(&args.model_file)?;
    let clf: RandomForestClassifier<f64, usize, DenseMatrix<f64>, Vec<usize>> =
        bincode::deserialize_from(&mut f)?;

    // Determine labels sidecar
    let labels_path = args
        .labels_file
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("{}.labels.json", args.model_file.display())));
    let labels: Vec<String> = serde_json::from_str(&std::fs::read_to_string(&labels_path)?)?;

    // Fallback keys in the same order as the features_all exposer
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

        let instance = DenseMatrix::new(1, row.len(), row.clone(), false).map_err(|e| anyhow::anyhow!("creating DenseMatrix: {:?}", e))?;
        let pred = clf.predict(&instance).map_err(|e| anyhow::anyhow!("prediction error: {:?}", e))?;
        let class_idx = pred[0];
        let decoded = labels
            .get(class_idx)
            .cloned()
            .unwrap_or_else(|| format!("unknown_{}", class_idx));

        // Write JSON string per line
        let out = serde_json::to_string(&decoded)?;
        println!("{}", out);
    }

    Ok(())
}
