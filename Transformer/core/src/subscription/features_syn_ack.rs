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
#[derive(Debug)]
pub struct Features {
    /// Features,
    pub features: Vec<f64>,
}

impl Features {}

impl Serialize for Features {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Features", 1)?;
        state.serialize_field("fts", &self.features)?;
        state.end()
    }
}

impl fmt::Display for Features {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.features,)?;
        Ok(())
    }
}

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
    #[cfg(feature = "timing")]
    compute_ns: u64,
    cnt: u64,
    syn_ts: f64,
    syn_ack_ts: f64,
    ack_ts: f64,
}

impl TrackedFeatures {
    #[inline]
    fn update(&mut self, segment: L4Pdu) -> Result<()> {
        self.cnt += 1;
        #[cfg(feature = "timing")]
        let start_ts = (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64;

        let curr_ts = unsafe { rte_rdtsc() } as f64 / *TSC_GHZ;
        // let curr_ts = segment.mbuf_ref().timestamp() as f64 * 1e3;

        let mbuf = segment.mbuf_ref();
        let eth = mbuf.parse_to::<Ethernet>()?;
        let ipv4 = eth.parse_to::<Ipv4>()?;

        if segment.dir {
            if self.syn_ts.is_nan() {
                // first packet is SYN
                self.syn_ts = curr_ts;
            }
            if !self.syn_ack_ts.is_nan() && self.ack_ts.is_nan() {
                let tcp = ipv4.parse_to::<Tcp>()?;
                if tcp.ack() {
                    self.ack_ts = curr_ts;
                }
            }
        } else {
            if self.syn_ack_ts.is_nan() && self.ack_ts.is_nan() {
                let tcp = ipv4.parse_to::<Tcp>()?;
                if tcp.synack() {
                    self.syn_ack_ts = curr_ts;
                }
            }
        }
        #[cfg(feature = "timing")]
        {
            let end_ts = (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64;
            self.compute_ns += end_ts - start_ts;
        }
        Ok(())
    }

    #[inline]
    fn extract_features(&mut self) -> Vec<f64> {
        #[cfg(feature = "timing")]
        let start_ts = (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64;
        let syn_ack = self.syn_ack_ts - self.syn_ts;
        let features = vec![syn_ack];
        #[cfg(feature = "timing")]
        {
            let end_ts = (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64;
            self.compute_ns += end_ts - start_ts;
        }
        features
    }
}

impl Trackable for TrackedFeatures {
    type Subscribed = Features;

    fn new(_five_tuple: FiveTuple) -> Self {
        TrackedFeatures {
            #[cfg(feature = "timing")]
            compute_ns: 0,
            cnt: 0,
            syn_ts: f64::NAN,
            syn_ack_ts: f64::NAN,
            ack_ts: f64::NAN,
        }
    }

    fn pre_match(
        &mut self,
        pdu: L4Pdu,
        _session_id: Option<usize>,
        subscription: &Subscription<Self::Subscribed>,
    ) {
        timer_start!(t);
        self.update(pdu).unwrap_or(());
        timer_elapsed_nanos!(subscription.timers, "update", t);
    }

    fn on_match(&mut self, _session: Session, _subscription: &Subscription<Self::Subscribed>) {}

    fn post_match(&mut self, pdu: L4Pdu, subscription: &Subscription<Self::Subscribed>) {
        timer_start!(t);
        self.update(pdu).unwrap_or(());
        timer_elapsed_nanos!(subscription.timers, "update", t);
    }

    fn on_terminate(&mut self, subscription: &Subscription<Self::Subscribed>) {
        timer_start!(t);
        let features = self.extract_features();
        timer_elapsed_nanos!(subscription.timers, "extract_features", t);
        let conn = Features { features };
        timer_record!(subscription.timers, "compute_ns", self.compute_ns);
        subscription.invoke(conn);
    }

    fn early_terminate(&self) -> bool {
        false
    }
}
