# change to this directory and run NDSCAN_SKIP_GUI=1 ARTIQ_ROOT=./.github/artiq-emulator nix develop ./.github/artiq-emulator --command python -m unittest discover -v test

#!/bin/bash
set -e
cd "$(dirname "$0")"
NDSCAN_SKIP_GUI=1 ARTIQ_ROOT=./.github/artiq-emulator nix develop ./.github/artiq-emulator --command python -m unittest discover -v test