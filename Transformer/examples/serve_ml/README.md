# Serve ML

Serve ML models

# Retina ML Model Serving Example

This example shows how to use a trained SmartCore classifier (RandomForest by default) to make real-time predictions on network connection features extracted by Retina.

## Overview

The `serve_ml` example:
1. Loads a trained classifier from a file
2. Processes network traffic using Retina's packet processing runtime
3. For each connection, extracts features and makes a prediction using the loaded model
4. Writes prediction results to a JSONL file (one JSON value per line)

## Required Files

You need:

1. A config file (e.g., `configs/online.toml`) for Retina runtime configuration
2. A trained model file containing a serialized SmartCore classifier (see below)
3. (Optional) A labels sidecar JSON file mapping numeric class ids to `s_mac` strings
4. An output file path where predictions will be written

## Model Format

The example expects a SmartCore classifier serialized with `bincode`. By default the training example uses `RandomForestClassifier` and the serving code expects models that accept `f64` features and `usize` labels; the matrix type is `DenseMatrix<f64>` and labels are `Vec<usize>`.

Example of saving a RandomForest model:

```rust
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::fs::File;

// Assume you have trained a RandomForest classifier
let clf: RandomForestClassifier<f64, usize> = /* your trained model */;

let mut file = File::create("model.bin")?;
bincode::serialize_into(&mut file, &clf)?;
```

## Labels sidecar

The training example (`train_model`) writes a sidecar JSON file named `<model-file>.labels.json` by default. This file is a JSON array indexed by numeric class id; each element is the corresponding `s_mac` string. For example:

```json
[
  "02:00:00:00:00:01",
  "02:00:00:00:00:02"
]
```

When present, `serve_ml` will load this sidecar and decode numeric predictions back to the original `s_mac` strings before writing outputs. If you want to use a different labels filename, see the `--labels-file` flag below.

## Building and Running

From the repository root:

```bash
# Build the binary
cargo build --release -p serve_ml

# Run (may need sudo for packet capture)
sudo env LD_LIBRARY_PATH=$LD_LIBRARY_PATH RUST_LOG=error \
  ./target/release/serve_ml \
  --config configs/online.toml \
  --model-file your_model.bin \
  --outfile predictions.jsonl
```

You may optionally pass an explicit labels sidecar file:

```bash
# Use explicit labels file
./target/release/serve_ml \
  --config configs/online.toml \
  --model-file your_model.bin \
  --labels-file your_model.labels.json \
  --outfile predictions.jsonl
```

If no `--labels-file` is provided, the server will look for `<model-file>.labels.json` by default.

The program will:
1. Load the model from `your_model.bin`
2. Load the labels sidecar if present
3. Start processing packets according to the config
4. For each TCP connection:
   - Extract the connection features
   - Make a prediction using the model
   - Decode prediction to `s_mac` (if labels sidecar is available) and write the decoded MAC string to `predictions.jsonl`
5. Print the total number of connections processed when done

## Model Input Features

The `serve_ml` example uses the `features_all` subscribable which exposes the feature vector as `conn.features` (a `Vec<f64>`). The exact order of elements in that vector is:

1. dur            — connection duration (seconds)
2. proto          — IP protocol number (as float)
3. s_bytes_sum    — sum of IPv4 lengths from source → dest (bytes)
4. d_bytes_sum    — sum of IPv4 lengths dest → source (bytes)
5. s_ttl_mean     — mean TTL of source → dest packets
6. d_ttl_mean     — mean TTL of dest → source packets
7. s_load         — source bitrate estimate (bits/sec)
8. d_load         — destination bitrate estimate (bits/sec)
9. s_pkt_cnt      — source packet count
10. d_pkt_cnt     — destination packet count
11. s_bytes_mean  — mean bytes per source packet
12. d_bytes_mean  — mean bytes per destination packet
13. s_iat_mean    — mean inter-arrival time for source direction (seconds)
14. d_iat_mean    — mean inter-arrival time for destination direction (seconds)
15. tcp_rtt       — measured TCP RTT (seconds)
16. syn_ack       — SYN → SYN-ACK time (seconds)
17. ack_dat       — SYN-ACK → ACK time (seconds)

Features are converted to a `DenseMatrix` row before prediction. Make sure your trained model expects features in this exact order and uses `f64` for feature values.

## Example Output

The output file (`predictions.jsonl`) will contain one prediction per line. When a labels sidecar is present, each line will be the decoded `s_mac` string (JSON string). Example:

```jsonl
"02:00:00:00:00:01"
"02:00:00:00:00:02"
"02:00:00:00:00:01"
```

If labels sidecar is missing, the example may emit `"unknown_<id>"` placeholders for classes it cannot decode.

## Training with JSONL

The repository includes `examples/serve_ml/src/bin/train_model.rs`, a trainer that:

- Reads a JSONL file where each line is a JSON object representing a sample.
- Uses the `s_mac` field as the sample label (string).
- Uses an `fts` array if present, otherwise falls back to extracting a known ordered set of numeric fields matching the `features_all` ordering.
- Trains a `RandomForestClassifier` and writes two files:
  - `<outfile>`: the bincode-serialized model (e.g., `model.bin`)
  - `<outfile>.labels.json`: a sidecar JSON array mapping label ids -> `s_mac` strings

Usage:

```bash
cargo run -p serve_ml --bin train_model --release -- train.jsonl model.bin
# Produces model.bin and model.bin.labels.json
```

## Notes

- The example is currently filtered to IPv4+TCP+TLS traffic via the `#[filter]` attribute.
- Predictions are made in real-time as connections are processed.
- The example uses a mutex-protected `BufWriter` to safely write predictions from the callback.
- Model loading and prediction are done using the SmartCore ML library.

If you want, I can add a small sample `train.jsonl` fixture and a script that runs train -> serve -> sample predictions end-to-end.

## Transformer (iTransformer) Trainer and Runner

This repository also includes trainer and runner binaries that use the iTransformer model (`itransformer` + `tch`).
They train a transformer-based model (saved using `tch::VarStore`) and a small linear classifier on top to predict `s_mac` classes.

Notes:
- The trainer `train_transformer` writes three artifacts on success:
  - `<out_weights>`: VarStore weights file (e.g. `model.ot`)
  - `<out_weights>.labels.json`: labels sidecar (class id -> `s_mac` string)
  - `<out_weights>.meta.json`: training hyperparameters used (num_variates, lookback_len, etc.)
- The runner `run_transformer` loads the weights file and (optionally) the labels sidecar to decode numeric predictions back to `s_mac`.

Examples (development run via `cargo run`):

Train (example):
```bash
# Run the transformer trainer on `train.jsonl`, produce `model.ot` and `model.ot.labels.json`
cargo run -p serve_ml --bin train_transformer --release -- \
  --train-file train.jsonl \
  --out-weights model.ot \
  --labels-out model.ot.labels.json \
  --lookback_len 1 \
  --num_variates 17 \
  --epochs 5
```

Run / Inference (example):
```bash
# Run the transformer inference binary on a JSONL test file.
cargo run -p serve_ml --bin run_transformer --release -- \
  --weights model.ot \
  --labels-file model.ot.labels.json \
  --test-file test.jsonl \
  --num-variates 17 \
  --lookback_len 1 \
  --pred_length 1
```

The `train_transformer` and `run_transformer` binaries accept several hyperparameter flags (depth, dim, heads, dim_head, num_tokens_per_variate, use_reversible_instance_norm, flash_attn, etc.). When possible the runner can be pointed at the `<out_weights>.meta.json` file produced by the trainer instead of re-specifying all flags.

