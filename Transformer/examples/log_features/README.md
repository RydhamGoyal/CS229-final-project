# Connection Features Logger

Demonstrates logging connection features to a file.

### Build and run
```
cargo build --release --bin log_features
sudo env LD_LIBRARY_PATH=$LD_LIBRARY_PATH RUST_LOG=error ./target/release/log_features -c <path/to/config.toml>
```

# Features produced by `log_features`

This document lists which cargo feature flags in `retina-core` enable each field in the `Features` record, and shows how to build/run the `log_features` example to collect a JSONL file containing all fields.

## Fields -> cargo features

The `Features` struct (see `core/src/subscription/features.rs`) contains these fields. The cargo feature you enable in `retina-core` controls whether the corresponding field is present in serialized output.

- `dur` — enabled by feature: `dur` (connection duration in seconds).
- `proto` — enabled by feature: `proto` (IP protocol number).
- `s_bytes_sum` — enabled by feature: `s_bytes_sum` (sum of IPv4 lengths from source→dest).
- `d_bytes_sum` — enabled by feature: `d_bytes_sum` (sum of IPv4 lengths dest→source).
- `s_ttl_mean` — enabled by feature: `s_ttl_mean` (mean TTL of source→dest packets).
- `d_ttl_mean` — enabled by feature: `d_ttl_mean` (mean TTL of dest→source packets).
- `s_load` — enabled by feature: `s_load` (source bit-rate estimate in bits/sec).
- `d_load` — enabled by feature: `d_load` (destination bit-rate estimate in bits/sec).
- `s_pkt_cnt` — enabled by feature: `s_pkt_cnt` (source packet count).
- `d_pkt_cnt` — enabled by feature: `d_pkt_cnt` (destination packet count).
- `s_bytes_mean` — enabled by feature: `s_bytes_mean` (mean bytes per packet source→dest).
- `d_bytes_mean` — enabled by feature: `d_bytes_mean` (mean bytes per packet dest→source).
- `s_iat_mean` — enabled by feature: `s_iat_mean` (mean inter-arrival time source→dest in seconds).
- `d_iat_mean` — enabled by feature: `d_iat_mean` (mean inter-arrival time dest→source in seconds).
- `tcp_rtt` — enabled by feature: `tcp_rtt` (measured TCP RTT in seconds, derived from handshake timings).
- `syn_ack` — enabled by feature: `syn_ack` (SYN → SYN-ACK time in seconds).
- `ack_dat` — enabled by feature: `ack_dat` (SYN-ACK → ACK time in seconds).
- `s_mac` — present when `timing` is NOT enabled (serialized MAC address of source).
- `d_mac` — present when `timing` is NOT enabled (serialized MAC address of destination).

Notes:
- All numeric fields are serialized as floating-point (`f64`).
- Field presence depends on cargo compile-time features declared in `core/Cargo.toml`.
- When the `timing` feature is enabled, the code uses DPDK TSC counters for timestamps; when `timing` is disabled the code uses mbuf packet timestamps and includes `s_mac`/`d_mac` fields.

## How to build the example with all features

`retina-core` defines these features (see `core/Cargo.toml`):

```
timing dur proto s_bytes_sum d_bytes_sum s_ttl_mean d_ttl_mean s_load d_load s_pkt_cnt d_pkt_cnt s_bytes_mean d_bytes_mean s_iat_mean d_iat_mean tcp_rtt syn_ack ack_dat
```

Recommended (explicit, reproducible): enable the desired `retina-core` features in the `log_features` dependency entry inside `examples/log_features/Cargo.toml`.

Edit `examples/log_features/Cargo.toml` and change the `retina-core` dependency to include a `features` array, for example:

```toml
retina-core = { path = "../../core", features = [
  "timing",
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
] }

After that change, build and run the example from the repository root:

```bash
cargo run -p log_features --release -- --config configs/online.toml --outfile features.jsonl
```

Notes:
- The `--` separates cargo arguments from arguments passed to the binary; the `--config` and `--outfile` flags are those supported by the example.
- If you prefer not to edit `Cargo.toml`, you can also temporarily build `retina-core` with features and then run the example, but the safest approach is to declare the dependency features as shown above so Cargo uses the correct feature set for linking.

## Including source/destination MAC addresses

To include `s_mac` and `d_mac` fields in each `Features` JSON record, do NOT enable the `timing` feature for `retina-core` — the code serializes MAC addresses only when the `timing` feature is *not* enabled. In practice that means either:

- Omit `"timing"` from the `features` array in the `retina-core` dependency in `examples/log_features/Cargo.toml`, or
- If you use a single-shot `cargo build` with `--features`, avoid specifying `timing` in that feature list.

Example dependency entry (all features except `timing`, which ensures MACs are emitted):

```toml
retina-core = { path = "../../core", features = [
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
] }
```

Then build/run as before:

```bash
cargo run -p log_features --release -- --config configs/online.toml --outfile features_with_macs.jsonl
```

Implementation detail: `s_mac` and `d_mac` are captured from the Ethernet header of the first seen packet for the connection (the code sets them when the tracked packet count is 1). That means the MACs appear when the program sees at least one packet for the connection and timing is disabled.

If you prefer the TSC-based `timing` mode (higher-resolution timestamps) but still want MACs, that isn't supported by the current code: `s_mac`/`d_mac` are compiled out when `timing` is enabled. You would need a small code change to always capture MAC addresses regardless of the `timing` feature.



## Example JSON line (all fields present)

Below is an example JSON object (one line in a JSONL file) that demonstrates the shape when all features are enabled. Values are illustrative.

```json
{
  "dur": 2.345,
  "proto": 6.0,
  "s_bytes_sum": 14500.0,
  "d_bytes_sum": 9200.0,
  "s_ttl_mean": 64.0,
  "d_ttl_mean": 63.5,
  "s_load": 49491500.0,
  "d_load": 31400000.0,
  "s_pkt_cnt": 120.0,
  "d_pkt_cnt": 85.0,
  "s_bytes_mean": 120.8333333333,
  "d_bytes_mean": 108.2352941176,
  "s_iat_mean": 0.019,
  "d_iat_mean": 0.027,
  "tcp_rtt": 0.120,
  "syn_ack": 0.060,
  "ack_dat": 0.060,
  "s_mac": "02:00:00:00:00:01",
  "d_mac": "02:00:00:00:00:02"
}
```

## Quick checklist

- Add the desired `retina-core` features into `examples/log_features/Cargo.toml` as shown.
- Build/run with `cargo run -p log_features --release -- --config <config> --outfile <path>`.
- The program writes one JSON object per line to the specified `--outfile` (default `features.jsonl`).

## If you want a single-shot CLI invocation

If you don't want to edit the example's Cargo.toml, you can alternatively (less reliably) attempt to build `retina-core` first with features, then run the example. Example:

```bash
# Build retina-core with features
cargo build -p retina-core --release --features "timing dur proto s_bytes_sum d_bytes_sum s_ttl_mean d_ttl_mean s_load d_load s_pkt_cnt d_pkt_cnt s_bytes_mean d_bytes_mean s_iat_mean d_iat_mean tcp_rtt syn_ack ack_dat"

# Then run the log_features binary
cargo run -p log_features --release -- --config configs/online.toml --outfile features.jsonl
```

Note: Cargo may rebuild `retina-core` if `log_features`'s dependency resolution requests different features; declaring features in the `log_features/Cargo.toml` dependency is the recommended approach for reproducible builds.

---

If you'd like, I can prepare a small patch that updates `examples/log_features/Cargo.toml` to include an `all-features` example dependency entry, or add a tiny script that runs the correct cargo commands. Which would you prefer?

