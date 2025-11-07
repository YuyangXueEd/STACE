#!/usr/bin/env bash
# Simple helper to run ESD unlearning for a list of concepts.

set -euo pipefail

# Resolve repository root relative to this script.
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
ESD_SCRIPT="${REPO_ROOT}/../external/esd/esd_sd.py"

# Configure training defaults.
MODEL="stable-diffusion"
VARIANT="${VARIANT:-xattn}"
ITERATIONS="${ITERATIONS:-200}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.0}"
NEGATIVE_GUIDANCE="${NEGATIVE_GUIDANCE:-2.0}"
HF_DEVICE="${HF_DEVICE:-cuda:0}"

# Concepts to erase.
CONCEPTS=(
  "airplane"
  "dog"
  "bird"
  "micky mouse"
  "pikachu"
  "superman"
  "nudity"
  "violence"
  "Pablo Picasso"
  "Andy Warhol"
  "Van Gogh"
)

# Map friendly variant names to the underlying ESD script flag.
case "${VARIANT}" in
  xattn) TRAIN_METHOD="esd-x" ;;
  noxattn) TRAIN_METHOD="esd-u" ;;
  full|selfattn) TRAIN_METHOD="esd-all" ;;
  esd-x|esd-u|esd-all|esd-x-strict) TRAIN_METHOD="${VARIANT}" ;;
  *)
    echo "Unsupported variant '${VARIANT}'. Valid options: xattn, noxattn, full, selfattn." >&2
    exit 1
    ;;
esac

OUTPUT_ROOT="${REPO_ROOT}/../data/unlearned_models/esd/${ITERATIONS}/${MODEL}"
mkdir -p "${OUTPUT_ROOT}"

echo "Using ESD script at: ${ESD_SCRIPT}"
echo "Saving checkpoints under: ${OUTPUT_ROOT}"
echo "Running with variant '${VARIANT}' -> train method '${TRAIN_METHOD}'"

for concept in "${CONCEPTS[@]}"; do
  concept_slug="${concept// /_}"
  concept_dir="${OUTPUT_ROOT}/${concept_slug}"

  mkdir -p "${concept_dir}"
  echo "[$(date --iso-8601=seconds)] Starting unlearning for concept: ${concept}"

  if ! "${PYTHON_BIN}" "${ESD_SCRIPT}" \
    --erase_concept "${concept}" \
    --train_method "${TRAIN_METHOD}" \
    --iterations "${ITERATIONS}" \
    --lr "${LEARNING_RATE}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --negative_guidance "${NEGATIVE_GUIDANCE}" \
    --save_path "${concept_dir}" \
    --device "${HF_DEVICE}"; then
      echo "[$(date --iso-8601=seconds)] Failed to erase concept '${concept}'. See logs above." >&2
      continue
  fi

  echo "[$(date --iso-8601=seconds)] Finished unlearning for concept: ${concept}"
done

echo "All requested concepts processed."
