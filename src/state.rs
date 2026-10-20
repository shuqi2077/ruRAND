use ruda_core::{rand::{CheckpointRng, get_checkpoint_rng}, stub::Mutex};
use rand::{RngExt, SeedableRng};

static SEED: Mutex<Option<CheckpointRng>> = Mutex::new(None);

pub fn seed(seed: u64) {
    let rng = CheckpointRng::seed_from_u64(seed);
    let mut seed = SEED.lock().unwrap();
    *seed = Some(rng);
}

pub fn get_seeds() -> [u32; 4] {
    let mut seed = SEED.lock().unwrap();
    let mut rng = match seed.take() {
        Some(rng_seeded) => rng_seeded,
        None => get_checkpoint_rng(),
    };
    let mut seeds: Vec<u32> = Vec::with_capacity(4);
    for _ in 0..4 {
        seeds.push(rng.random());
    }
    *seed = Some(rng);

    seeds.try_into().unwrap()
}

pub fn rng_state() -> Vec<u8> {
    let mut state = SEED.lock().unwrap();
    ruda_core::rand::checkpoint_rng(state.get_or_insert_with(get_checkpoint_rng))
}

pub fn restore_rng_state(bytes: &[u8]) -> Result<(), &'static str> {
    let rng = ruda_core::rand::restore_rng(bytes)?;
    *SEED.lock().unwrap() = Some(rng);
    Ok(())
}
