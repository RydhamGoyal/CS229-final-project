//! Features.

use crate::conntrack::conn_id::FiveTuple;
use crate::conntrack::pdu::{L4Context, L4Pdu};
use crate::conntrack::ConnTracker;
use crate::dpdk::{rte_get_tsc_hz, rte_rdtsc};
use crate::filter::FilterResult;
use crate::memory::mbuf::Mbuf;
use crate::protocols::packet::ethernet::Ethernet;
use crate::protocols::packet::ipv4::Ipv4;
use crate::protocols::packet::tcp::Tcp;
use crate::protocols::packet::Packet;
use crate::protocols::stream::{ConnParser, Session};
use crate::subscription::*;

use std::fmt;

use anyhow::Result;
use serde::ser::{SerializeStruct, Serializer};
use serde::Serialize;

use lazy_static::lazy_static;

lazy_static! {
    static ref TSC_GHZ: f64 = unsafe { rte_get_tsc_hz() } as f64 / 1e9;
}

/// A features record.
#[derive(Debug, Serialize)]
pub struct Features {
    dur: f64,
    proto: f64,
    s_bytes_sum: f64,
    d_bytes_sum: f64,
    s_ttl_mean: f64,
    d_ttl_mean: f64,
    s_load: f64,
    d_load: f64,
    s_pkt_cnt: f64,
    d_pkt_cnt: f64,
    s_bytes_mean: f64,
    d_bytes_mean: f64,
    s_iat_mean: f64,
    d_iat_mean: f64,
    tcp_rtt: f64,
    syn_ack: f64,
    ack_dat: f64,
}

impl Features {}


// impl Serialize for Features {
//     fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
//     where
//         S: Serializer,
//     {
//         let mut state = serializer.serialize_struct("Features", 1)?;
//         state.serialize_field("fts", &self)?;
//         state.end()
//     }
// }

// impl fmt::Display for Features {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(f, "{:?}", self.features,)?;
//         Ok(())
//     }
// }

impl Subscribable for Features {
    type Tracked = TrackedFeatures;

    fn level() -> Level {
        Level::Connection
    }

    // TODO: return a vector of all known parsers.
    fn parsers() -> Vec<ConnParser> {
        vec![]
    }

    fn process_packet(
        mbuf: Mbuf,
        subscription: &Subscription<Self>,
        conn_tracker: &mut ConnTracker<Self::Tracked>,
    ) {
        match subscription.filter_packet(&mbuf) {
            FilterResult::MatchTerminal(idx) | FilterResult::MatchNonTerminal(idx) => {
                if let Ok(ctxt) = L4Context::new(&mbuf, idx) {
                    conn_tracker.process(mbuf, ctxt, subscription);
                }
            }
            FilterResult::NoMatch => drop(mbuf),
        }
    }
}

/// Tracks a feature record throughout its lifetime.
///
/// ## Note
/// Internal connection state is an associated type of a `pub` trait, and therefore must also be
/// public. Documentation is hidden by default to avoid confusing users.
#[doc(hidden)]
pub struct TrackedFeatures {
    cnt: u64,
    get_curr_ts: u64,
    get_mbuf: u64,
    parse_eth: u64,
    parse_ipv4: u64,
    update_syn_ts: u64,
    update_s_last_ts: u64,
    update_s_pkt_cnt: u64,
    update_s_bytes_sum: u64,
    update_s_ttl_sum: u64,
    update_ack_ts: u64,
    update_proto: u64,
    update_d_last_ts: u64,
    update_d_pkt_cnt: u64,
    update_d_bytes_sum: u64,
    update_d_ttl_sum: u64,
    update_syn_ack_ts: u64,

    syn_ts: f64,
    syn_ack_ts: f64,
    ack_ts: f64,
    s_last_ts: f64,
    d_last_ts: f64,
    s_pkt_cnt: f64,
    d_pkt_cnt: f64,
    s_bytes_sum: f64,
    d_bytes_sum: f64,
    s_ttl_sum: f64,
    d_ttl_sum: f64,
    proto: f64,
}

impl TrackedFeatures {
    #[inline]
    fn update(&mut self, segment: L4Pdu, sub: &Subscription<Features>) -> Result<()> {
        self.cnt += 1;

        let start_ts = (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64;
        let curr_ts = unsafe { rte_rdtsc() } as f64 / *TSC_GHZ;
        self.get_curr_ts += (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64 - start_ts;

        let start_ts = (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64;
        let mbuf = segment.mbuf_ref();
        self.get_mbuf += (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64 - start_ts;

        let start_ts = (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64;
        let eth = mbuf.parse_to::<Ethernet>()?;
        self.parse_eth += (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64 - start_ts;

        let start_ts = (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64;
        let ipv4 = eth.parse_to::<Ipv4>()?;
        self.parse_ipv4 += (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64 - start_ts;


        if segment.dir {
            let start_ts = (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64;
            if self.syn_ts.is_nan() {
                self.syn_ts = curr_ts;
            }
            self.update_syn_ts += (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64 - start_ts;


            let start_ts = (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64;
            {
                self.s_last_ts = curr_ts;
            }
            self.update_s_last_ts += (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64 - start_ts;


            let start_ts = (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64;
            {
                self.s_pkt_cnt += 1.0;
            }
            self.update_s_pkt_cnt += (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64 - start_ts;

            let start_ts = (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64;
            {
                self.s_bytes_sum += ipv4.total_length() as f64;
            }
            self.update_s_bytes_sum += (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64 - start_ts;


            let start_ts = (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64;
            {
                self.s_ttl_sum += ipv4.time_to_live() as f64;
            }
            self.update_s_ttl_sum += (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64 - start_ts;


            let start_ts = (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64;
            if !self.syn_ack_ts.is_nan() && self.ack_ts.is_nan() {
                let tcp = ipv4.parse_to::<Tcp>()?;
                if tcp.ack() {
                    self.ack_ts = curr_ts;
                }
            }
            self.update_ack_ts += (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64 - start_ts;


            let start_ts = (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64;
            {
                self.proto = ipv4.protocol() as f64;
            }
            self.update_proto += (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64 - start_ts;

        } else {

            let start_ts = (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64;
            {
                self.d_last_ts = curr_ts;
            }
            self.update_d_last_ts += (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64 - start_ts;

            let start_ts = (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64;
            {
                self.d_pkt_cnt += 1.0;
            }
            self.update_d_pkt_cnt += (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64 - start_ts;

            let start_ts = (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64;
            {
                self.d_bytes_sum += ipv4.total_length() as f64;
            }
            self.update_d_bytes_sum += (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64 - start_ts;

            let start_ts = (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64;
            {
                self.d_ttl_sum += ipv4.time_to_live() as f64;
            }
            self.update_d_ttl_sum += (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64 - start_ts;

            let start_ts = (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64;
            if self.syn_ack_ts.is_nan() {
                let tcp = ipv4.parse_to::<Tcp>()?;
                if tcp.synack() {
                    self.syn_ack_ts = curr_ts;
                }
            }
            self.update_syn_ack_ts += (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64 - start_ts;

        }
        Ok(())
    }

    #[inline]
    fn extract_features(&mut self) -> Features {

        let dur = self.s_last_ts.max(self.d_last_ts) - self.syn_ts;
        let s_ttl_mean = self.s_ttl_sum / self.s_pkt_cnt;
        let d_ttl_mean = self.d_ttl_sum - self.d_pkt_cnt;
        let s_load = self.s_bytes_sum * 8e9 / dur;
        let d_load = self.d_bytes_sum * 8e9 / dur;
        let s_bytes_mean = self.s_bytes_sum / self.s_pkt_cnt;
        let d_bytes_mean = self.d_bytes_sum / self.d_pkt_cnt;
        let s_iat_mean = (self.s_last_ts - self.syn_ts) / (self.s_pkt_cnt - 1.0);
        let d_iat_mean = (self.d_last_ts - self.syn_ack_ts) / (self.d_pkt_cnt - 1.0);
        let syn_ack = self.syn_ack_ts - self.syn_ts;
        let ack_dat = self.ack_ts - self.syn_ack_ts;
        let tcp_rtt = syn_ack + ack_dat;
        let features = Features {
            dur,
            proto: self.proto,
            s_bytes_sum: self.s_bytes_sum,
            d_bytes_sum: self.d_bytes_sum,
            s_ttl_mean,
            d_ttl_mean,
            s_load,
            d_load,
            s_pkt_cnt: self.s_pkt_cnt,
            d_pkt_cnt: self.d_pkt_cnt,
            s_bytes_mean,
            d_bytes_mean,
            s_iat_mean,
            d_iat_mean,
            tcp_rtt,
            syn_ack,
            ack_dat,
        };
        features
    }
}

impl Trackable for TrackedFeatures {
    type Subscribed = Features;

    fn new(_five_tuple: FiveTuple) -> Self {
        TrackedFeatures {
            cnt: 0,
            get_curr_ts: 0,
            get_mbuf: 0,
            parse_eth: 0,
            parse_ipv4: 0,
            update_syn_ts: 0,
            update_s_last_ts: 0,
            update_s_pkt_cnt: 0,
            update_s_bytes_sum: 0,
            update_s_ttl_sum: 0,
            update_ack_ts: 0,
            update_proto: 0,
            update_d_last_ts: 0,
            update_d_pkt_cnt: 0,
            update_d_bytes_sum: 0,
            update_d_ttl_sum: 0,
            update_syn_ack_ts: 0,

            syn_ts: f64::NAN,
            syn_ack_ts: f64::NAN,
            ack_ts: f64::NAN,
            s_last_ts: f64::NAN,
            d_last_ts: f64::NAN,
            s_pkt_cnt: 0.0,
            d_pkt_cnt: 0.0,
            s_bytes_sum: 0.0,
            d_bytes_sum: 0.0,
            s_ttl_sum: 0.0,
            d_ttl_sum: 0.0,
            proto: f64::NAN,
        }
    }

    fn pre_match(
        &mut self,
        pdu: L4Pdu,
        _session_id: Option<usize>,
        subscription: &Subscription<Self::Subscribed>,
    ) {
        self.update(pdu, subscription).unwrap_or(());
    }

    fn on_match(&mut self, _session: Session, _subscription: &Subscription<Self::Subscribed>) {}

    fn post_match(&mut self, pdu: L4Pdu, subscription: &Subscription<Self::Subscribed>) {
        self.update(pdu, subscription).unwrap_or(());
    }

    fn on_terminate(&mut self, subscription: &Subscription<Self::Subscribed>) {
        timer_start!(t);
        let features = self.extract_features();
        timer_elapsed_nanos!(subscription.timers, "extract_features", t);
        timer_record!(subscription.timers, "get_curr_ts", self.get_curr_ts);
        timer_record!(subscription.timers, "get_mbuf", self.get_mbuf);
        timer_record!(subscription.timers, "parse_eth", self.parse_eth);
        timer_record!(subscription.timers, "parse_ipv4", self.parse_ipv4);
        timer_record!(subscription.timers, "update_syn_ts", self.update_syn_ts);
        timer_record!(subscription.timers, "update_s_last_ts", self.update_s_last_ts);
        timer_record!(subscription.timers, "update_s_pkt_cnt", self.update_s_pkt_cnt);
        timer_record!(subscription.timers, "update_s_bytes_sum", self.update_s_bytes_sum);
        timer_record!(subscription.timers, "update_s_ttl_sum", self.update_s_ttl_sum);
        timer_record!(subscription.timers, "update_ack_ts", self.update_ack_ts);
        timer_record!(subscription.timers, "update_proto", self.update_proto);
        timer_record!(subscription.timers, "update_d_last_ts", self.update_d_last_ts);
        timer_record!(subscription.timers, "update_d_pkt_cnt", self.update_d_pkt_cnt);
        timer_record!(subscription.timers, "update_d_bytes_sum", self.update_d_bytes_sum);
        timer_record!(subscription.timers, "update_d_ttl_sum", self.update_d_ttl_sum);
        timer_record!(subscription.timers, "update_syn_ack_ts", self.update_syn_ack_ts);

        let conn = features;
        subscription.invoke(conn);
    }

    fn early_terminate(&self) -> bool {
        false
    }
}
