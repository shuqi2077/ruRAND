#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod state;
mod tensor;

pub use state::{HostRng, seed, rng_state, restore_rng_state};
pub use tensor::{float_random, int_random};
