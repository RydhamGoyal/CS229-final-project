use retina_core::config::load_config;
use retina_core::subscription::features::Features;
use retina_core::Runtime;
use retina_core::config::RuntimeConfig;
use retina_filtergen::filter;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use anyhow::Result;
use clap::Parser;
use serde::Serialize;

// Define command-line arguments.
#[derive(Parser, Debug)]
struct Args {
    #[clap(short, long, parse(from_os_str), value_name = "FILE")]
    config: PathBuf,
    #[clap(short, long, parse(from_os_str), value_name = "FILE")]
    outfile: PathBuf,
}

#[filter("ipv4 and tcp")]
fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    let config = load_config(&args.config);
    let mut file = File::create(args.outfile)?;

    let cnt = AtomicUsize::new(0);

    let callback = |conn: Features| {
        cnt.fetch_add(1, Ordering::Relaxed);
    };
    let mut runtime = Runtime::new(config.clone(), filter, callback)?;
    runtime.run();

    let output = Output {
        config,
        num_conns: cnt.load(Ordering::SeqCst),
    };
    if let Ok(serialized) = serde_json::to_string(&output) {
        file.write_all(serialized.as_bytes())?;
    }
    println!("Done. Extract features from {:?} connections", cnt);
    Ok(())
}
#[derive(Debug, Serialize)]
struct Output {
    config: RuntimeConfig,
    num_conns: usize,
}
