# Connection Features Extractor

Extracts connection features and does nothing.

### Build and run
```
cargo build --release --bin extract_features
sudo env LD_LIBRARY_PATH=$LD_LIBRARY_PATH RUST_LOG=error ./target/release/extract_features -c <path/to/config.toml> -o <path/to/output.json>
```
