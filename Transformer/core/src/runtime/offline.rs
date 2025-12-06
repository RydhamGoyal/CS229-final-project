use crate::config::{ConnTrackConfig, OfflineConfig};
use crate::conntrack::{ConnTracker, TrackerConfig};
use crate::dpdk;
use crate::filter::Filter;
use crate::lcore::{CoreId, SocketId};
use crate::memory::mbuf::Mbuf;
use crate::memory::mempool::Mempool;
use crate::protocols::stream::ParserRegistry;
use crate::subscription::*;

use std::collections::BTreeMap;
use std::ffi::CString;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use cpu_time::ProcessTime;
use pcap::Capture;

pub(crate) struct OfflineRuntime<'a, S>
where
    S: Subscribable,
{
    pub(crate) mempool_name: String,
    pub(crate) filter: Filter,
    pub(crate) subscription: Arc<Subscription<'a, S>>,
    pub(crate) options: OfflineOptions,
}

impl<'a, S> OfflineRuntime<'a, S>
where
    S: Subscribable,
{
    pub(crate) fn new(
        options: OfflineOptions,
        mempools: &BTreeMap<SocketId, Mempool>,
        filter: Filter,
        subscription: Arc<Subscription<'a, S>>,
    ) -> Self {
        let core_id = CoreId(unsafe { dpdk::rte_lcore_id() } as u32);
        let mempool_name = mempools
            .get(&core_id.socket_id())
            .expect("Get offline mempool")
            .name()
            .to_string();
        OfflineRuntime {
            mempool_name,
            filter,
            subscription,
            options,
        }
    }

    pub(crate) fn run(&self) {
        log::info!(
            "Launched offline analysis. Processing pcap: {}",
            self.options.offline.pcap,
        );

        let mut nb_pkts = 0;
        let mut nb_bytes = 0;

        let config = TrackerConfig::from(&self.options.conntrack);
        let registry = ParserRegistry::build::<S>(&self.filter).expect("Unable to build registry");
        log::debug!("{:#?}", registry);
        let mut stream_table = ConnTracker::<S::Tracked>::new(config, registry);

        let mempool_raw = self.get_mempool_raw();
        let pcap_path = Path::new(self.options.offline.pcap.as_str());
        let pcaps = if pcap_path.is_dir() {
            get_pcaps(pcap_path).expect("Error reading pcap directory.")
        } else {
            vec![pcap_path.to_path_buf()]
        };

        let start = ProcessTime::try_now().expect("Getting process time failed");

        for pcap in pcaps.iter() {
            log::info!("Processing {}", pcap.display());
            let mut cap = Capture::from_file(pcap).expect("Error opening pcap. Aborting.");
            while let Ok(frame) = cap.next() {
                if frame.header.len as usize > self.options.offline.mtu {
                    continue;
                }
                let timeval = frame.header.ts;
                let unix_ts_us = timeval.tv_sec * 1_000_000 + timeval.tv_usec;
                let mbuf = Mbuf::from_bytes(frame.data, unix_ts_us, mempool_raw)
                    .expect("Unable to allocate mbuf. Try increasing mempool size.");
                nb_pkts += 1;
                nb_bytes += mbuf.data_len() as u64;

                S::process_packet(mbuf, &self.subscription, &mut stream_table);
            }

            // // Deliver remaining data in table
            stream_table.drain(&self.subscription);
        }
        let cpu_time = start.elapsed();
        println!("Processed: {} pkts, {} bytes", nb_pkts, nb_bytes);
        println!("CPU time: {:?}ms", cpu_time.as_millis());
    }

    fn get_mempool_raw(&self) -> *mut dpdk::rte_mempool {
        let cname = CString::new(self.mempool_name.clone()).expect("Invalid CString conversion");
        unsafe { dpdk::rte_mempool_lookup(cname.as_ptr()) }
    }
}

/// Read-only runtime options for the offline core
#[derive(Debug)]
pub(crate) struct OfflineOptions {
    pub(crate) offline: OfflineConfig,
    pub(crate) conntrack: ConnTrackConfig,
}

/// Returns a list of pcaps in `dir`.
fn get_pcaps(dir_path: &Path) -> Result<Vec<PathBuf>> {
    let entries = fs::read_dir(dir_path)?;
    let mut pcaps = vec![];
    for entry in entries {
        let path = entry?.path();
        if path.is_file() {
            if let Some(extension) = path.extension() {
                if extension == "pcap" {
                    pcaps.push(path);
                }
            }
        }
    }
    pcaps.sort();
    log::info!("Found {} pcaps", pcaps.len());
    Ok(pcaps)
}
