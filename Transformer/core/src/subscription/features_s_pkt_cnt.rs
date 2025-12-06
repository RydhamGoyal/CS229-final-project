//! Features.

use crate::conntrack::conn_id::FiveTuple;
use crate::conntrack::pdu::{L4Context, L4Pdu};
use crate::conntrack::ConnTracker;
use crate::dpdk::{rte_get_tsc_hz, rte_rdtsc};
use crate::filter::FilterResult;
use crate::memory::mbuf::Mbuf;
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
    s_pkt_cnt: f64,
}

impl TrackedFeatures {
    #[inline]
    fn update(&mut self, segment: L4Pdu) -> Result<()> {
        self.cnt += 1;
        #[cfg(feature = "timing")]
        let start_ts = (unsafe { rte_rdtsc() } as f64 / *TSC_GHZ) as u64;

        if segment.dir {
            self.s_pkt_cnt += 1.0;
        } else {
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
        let features = vec![self.s_pkt_cnt];
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
            s_pkt_cnt: 0.0,
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
