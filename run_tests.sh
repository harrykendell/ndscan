#!/bin/bash
set -e
cd "$(dirname "$0")"
NDSCAN_SKIP_GUI=1 ARTIQ_ROOT=./.github/artiq-emulator nix develop --accept-flake-config ./.github/artiq-emulator --command python test/unittest_timeout.py discover -v test
