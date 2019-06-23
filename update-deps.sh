#!/bin/sh

set -ex

git submodule update --init
cargo vendor ./vendor
