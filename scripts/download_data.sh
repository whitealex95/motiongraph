#!/usr/bin/env bash
# Download the G1-retargeted LAFAN1 clips (CSV, 30 FPS) from HuggingFace.
# Source: lvhaidong/LAFAN1_Retargeting_Dataset (public mirror of unitreerobotics set).
set -euo pipefail

REPO="lvhaidong/LAFAN1_Retargeting_Dataset"
OUT="$(cd "$(dirname "$0")/.." && pwd)/data/g1"
BASE="https://huggingface.co/datasets/${REPO}/resolve/main/g1"
mkdir -p "$OUT"

# Locomotion subset used by the demos; pass "all" to grab every G1 clip.
CLIPS=(walk1_subject1 walk1_subject2 walk1_subject5 walk2_subject1 walk3_subject1 \
       walk3_subject2 walk4_subject1 run1_subject2 run1_subject5 run2_subject1 \
       sprint1_subject2 sprint1_subject4)

if [[ "${1:-}" == "all" ]]; then
  echo "Fetching full G1 file list from HF API..."
  mapfile -t CLIPS < <(curl -sL "https://huggingface.co/api/datasets/${REPO}/tree/main/g1?recursive=1" \
    | python3 -c "import sys,json;[print(x['path'].split('/')[-1][:-4]) for x in json.load(sys.stdin) if x['path'].endswith('.csv')]")
fi

echo "Downloading ${#CLIPS[@]} clips -> $OUT"
for c in "${CLIPS[@]}"; do
  dst="$OUT/${c}.csv"
  [[ -s "$dst" ]] && { echo "  skip $c (exists)"; continue; }
  echo "  get  $c"
  curl -fsSL "${BASE}/${c}.csv" -o "$dst"
done
echo "Done. $(ls "$OUT" | wc -l) files in $OUT"
