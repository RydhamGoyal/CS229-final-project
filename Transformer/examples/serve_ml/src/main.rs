use retina_core::config::load_config;
// Import the Features type that exposes a Vec<f64> of all connection features
use retina_core::subscription::features_all::Features;
use retina_core::Runtime;
use retina_filtergen::filter;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, Arc};

use anyhow::Result;
use clap::Parser;

// use smartcore::dataset::Dataset;
// use smartcore::linalg::basic::arrays::{Array, Array2};
use smartcore::linalg::basic::matrix::DenseMatrix;
// use smartcore::metrics::accuracy;
// use smartcore::model_selection::train_test_split;
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;

// Define command-line arguments.
#[derive(Parser, Debug)]
struct Args {
    #[clap(short, long, parse(from_os_str), value_name = "CONFIG_FILE")]
    config: PathBuf,
    #[clap(short, long, parse(from_os_str), value_name = "MODEL_FILE")]
    model_file: PathBuf,
    /// Optional labels sidecar JSON file mapping class ids to s_mac strings
    #[clap(short = 'l', long = "labels-file", parse(from_os_str), value_name = "LABELS_FILE")]
    labels_file: Option<PathBuf>,
    #[clap(short, long, parse(from_os_str), value_name = "OUT_FILE")]
    outfile: PathBuf,
}

//#[filter("ipv4 and tls and (tls.sni ~ 'nflxvideo\\.net$' or tls.sni ~ 'row\\.aiv-cdn\\.net$' or tls.sni ~ 'media\\.dssott\\.com$' or tls.sni ~ 'vod-adaptive\\.akamaized\\.net$' or tls.sni ~ 'hls\\.ttvnw\\.net$' or tls.sni ~ 'aoc\\.tv\\.apple\\.com$' or tls.sni ~ 'airspace-.*\\.cbsivideo\\.com$' or tls.sni ~ 'prod\\.gccrunchyroll\\.com$' or tls.sni ~ 'vrv\\.akamaized\\.net$')")]
#[filter("ipv4 and tcp and tls")]
fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    let config = load_config(&args.config);

    let file = Mutex::new(BufWriter::new(File::create(&args.outfile)?));
    let cnt = Arc::new(AtomicUsize::new(0));
    let clf = load_clf(&args.model_file)?;

    // Determine labels sidecar path: use provided flag or default to "<model_file>.labels.json"
    let labels_path = args
        .labels_file
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("{}.labels.json", args.model_file.display())));
    let labels: Vec<String> = serde_json::from_str(&std::fs::read_to_string(&labels_path)?)?;

    // move closure so clf and labels are captured by value and live for the runtime
    // Clone the Arc so the original `cnt` remains available after the closure is moved.
    let cnt_cl = Arc::clone(&cnt);
    let callback = move |conn: Features| {
        let features = conn.features;
        // DenseMatrix::new may return a Result in some smartcore versions; unwrap here so we
        // pass the concrete DenseMatrix type to `predict`.
        let instance = DenseMatrix::new(1, features.len(), features, false).unwrap();
        let pred = clf.predict(&instance).unwrap();

    cnt_cl.fetch_add(1, Ordering::Relaxed);

        // decode numeric class back to MAC address if available
        let class_idx = pred[0];
        let decoded = labels
            .get(class_idx)
            .cloned()
            .unwrap_or_else(|| format!("unknown_{}", class_idx));

        let res = serde_json::to_string(&decoded).unwrap();
        let mut wtr = file.lock().unwrap();
        wtr.write_all(res.as_bytes()).unwrap();
        wtr.write_all(b"\n").unwrap();
    };
    let mut runtime = Runtime::new(config, filter, callback)?;
    runtime.run();

    println!("Done. Processed {:?} connections", cnt.load(Ordering::Relaxed));
    Ok(())
}

/// Loads a trained classifier from `file`.
fn load_clf(
    fname: &PathBuf,
) -> Result<DecisionTreeClassifier<f64, usize, DenseMatrix<f64>, Vec<usize>>> {
    let mut file = File::open(fname)?;
    let clf: DecisionTreeClassifier<f64, usize, DenseMatrix<f64>, Vec<usize>> =
        bincode::deserialize_from(&mut file)?;
    Ok(clf)
}
