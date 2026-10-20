use ruda_core::rand::{SeedableRng, StdRng, CheckpointRng};
use ruda_core::stub::Mutex;

/// Type alias for the RNG used by Flex.
pub type HostRng = StdRng;

/// Global seed storage for reproducible random number generation.
/// Uses Mutex for thread-safe RNG state management.
pub(crate) static SEED: Mutex<Option<CheckpointRng>> = Mutex::new(None);

/// Fallback RNG when `SEED` is empty (consumed or never set).
///
/// The seeding flow is: `Backend::seed()` stores a `FlexRng` in `SEED`. Random
/// ops (`float_random`, `int_random`) call `SEED.lock().take()`, consuming it for
/// that op and falling back to this function for subsequent calls. This function
/// delegates to ruda_core's own entropy source.
pub(crate) fn get_seeded_rng() -> CheckpointRng {
    ruda_core::rand::get_checkpoint_rng()
}

pub fn seed(seed: u64) {
    let rng = CheckpointRng::seed_from_u64(seed);
    let mut seed_lock = SEED.lock().unwrap();
    *seed_lock = Some(rng);
}

pub fn rng_state() -> alloc::vec::Vec<u8> {
    let mut state = SEED.lock().unwrap();
    ruda_core::rand::checkpoint_rng(state.get_or_insert_with(get_seeded_rng))
}

pub fn restore_rng_state(bytes: &[u8]) -> Result<(), &'static str> {
    let rng = ruda_core::rand::restore_rng(bytes)?;
    *SEED.lock().unwrap() = Some(rng);
    Ok(())
}
