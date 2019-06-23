#!/bin/sh

set -ex

# RUSTFLAGS=''

cargo clean
cargo update
cargo build --release --frozen
