/// Records number of CPU cycles since processor reset.
#[allow(unused_macros)]
macro_rules! timer_start {
    ( $start:ident ) => {
        #[cfg(feature = "timing")]
        let $start = unsafe { $crate::dpdk::rte_rdtsc() };
    };
}

/// Records number of CPU cycles since `start`.
#[allow(unused_macros)]
macro_rules! timer_elapsed_cycles {
    ( $timers:expr, $timer:expr, $start:ident ) => {
        #[cfg(feature = "timing")]
        $timers.record($timer, unsafe { $crate::dpdk::rte_rdtsc() } - $start);
    };
}

/// Records number of nanoseconds since `start`.
#[allow(unused_macros)]
macro_rules! timer_elapsed_nanos {
    ( $timers:expr, $timer:expr, $start:ident ) => {
        #[cfg(feature = "timing")]
        $timers.record($timer, unsafe {
            ($crate::dpdk::rte_rdtsc() - $start) as f64 / (rte_get_tsc_hz() as f64 / 1e9)
        } as u64);
    };
}

/// Record a value to timer.
#[allow(unused_macros)]
macro_rules! timer_record {
    ( $timers:expr, $timer:expr, $time:expr ) => {
        #[cfg(feature = "timing")]
        $timers.record($timer, $time);
    };
}
