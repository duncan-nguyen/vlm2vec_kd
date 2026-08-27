#!/bin/bash
# Kept as the entry point the README has always referenced. The old body was a
# serial wget/unzip chain over all 47.2 GB of MMEB-train; download_mmeb.py does
# the same job in parallel, resumably, and can fetch only the subsets a given
# experiment needs.
#
#   python scripts/data/download_mmeb.py --preset cls        # 11.9 GB
#   python scripts/data/download_mmeb.py --preset grounding  #  3.6 GB
#   python scripts/data/download_mmeb.py --preset vqa        # 16.6 GB
#   python scripts/data/download_mmeb.py --preset ret        # 14.0 GB
#   python scripts/data/download_mmeb.py --for scripts/train/rebuttal/rebuttal_hierd_grounding.sh
#
set -e
python scripts/data/download_mmeb.py --preset all "$@"
