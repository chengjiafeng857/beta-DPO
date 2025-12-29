#!/bin/bash

# Exit on error
set -e

# Auto-shutdown logic
POD_ID="${RUNPOD_POD_ID:-}"
AUTO_SHUTDOWN=true
RUN_SFT=true
RUN_DPO=true
cleanup() {
    if [[ -n "$POD_ID" && "$AUTO_SHUTDOWN" == "true" ]]; then
        echo "[cleanup] Training finished (or script exited). Stopping pod $POD_ID..."
        runpodctl stop pod "$POD_ID" || true
    else
        echo "[cleanup] Skipping auto-shutdown (POD_ID unset or AUTO_SHUTDOWN=false)."
    fi
}
on_interrupt() {
    echo "[cleanup] Interrupt received; skipping auto-shutdown."
    AUTO_SHUTDOWN=false
    exit 130
}
trap cleanup EXIT
trap on_interrupt INT

echo "Starting training initialization..."

# Training config (override with env vars if needed)
MODEL_NAME="${MODEL_NAME:-llama32-1b}"
DATASET_NAME="${DATASET_NAME:-hh}"
DATA_FRACTION="${DATA_FRACTION:-0.5}"
SFT_DATA_FRACTION="${SFT_DATA_FRACTION:-$DATA_FRACTION}"
LOSS_BETA="${LOSS_BETA:-0.1}"
BASE_EXP_NAME="${BASE_EXP_NAME:-${EXP_NAME:-llama32_1b}}"
SFT_EXP_NAME="${SFT_EXP_NAME:-${BASE_EXP_NAME}_sft_runpod}"
DPO_EXP_NAME="${DPO_EXP_NAME:-${BASE_EXP_NAME}_dpo_runpod}"
TRAINER="${TRAINER:-BasicTrainer}"
LOCAL_DIRS="${LOCAL_DIRS:-/scr-ssd,/scr,.cache}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -noautoshut)
            AUTO_SHUTDOWN=false
            shift
            ;;
        -sft)
            RUN_SFT=true
            RUN_DPO=false
            shift
            ;;
        -dpo)
            RUN_SFT=false
            RUN_DPO=true
            shift
            ;;
        *)
            echo "Error: Unknown flag '$1'"
            AUTO_SHUTDOWN=false
            exit 1
            ;;
    esac
done

# 1. Create venv using uv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment '.venv' using uv..."
    uv venv
else
    echo "Virtual environment '.venv' already exists."
fi

# 2. Activate venv
echo "Activating virtual environment..."
source .venv/bin/activate

# 3. Install dependencies using uv
echo "Installing dependencies..."
# If uv.lock exists, use sync (fastest and most reliable)
if [ -f "uv.lock" ]; then
    echo "Found uv.lock. Syncing environment..."
    uv sync || { echo "Error: uv sync failed."; AUTO_SHUTDOWN=false; exit 1; }
# If pyproject.toml exists but no lock, install from it
elif [ -f "pyproject.toml" ]; then
    echo "Found pyproject.toml. Installing..."
    uv pip install . || { echo "Error: uv pip install . failed."; AUTO_SHUTDOWN=false; exit 1; }
# Fallback to requirements.txt
elif [ -f "requirements.txt" ]; then
    echo "Found requirements.txt. Installing..."
    uv pip install -r requirements.txt || { echo "Error: uv pip install requirements.txt failed."; AUTO_SHUTDOWN=false; exit 1; }
else
    echo "Error: No dependency file found (uv.lock, pyproject.toml, or requirements.txt)."
    AUTO_SHUTDOWN=false
    exit 1
fi

# 4. HuggingFace Login
echo "Logging in to Hugging Face..."
# 4. HuggingFace Login
echo "Logging in to Hugging Face..."
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set."
    echo "Please export HF_TOKEN='your_token_here' before running this script."
    AUTO_SHUTDOWN=false
    exit 1
fi
uv run huggingface-cli login --token "$HF_TOKEN"

# 5. WandB Login
echo "Logging in to WandB..."
if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY environment variable is not set."
    echo "Please export WANDB_API_KEY='your_key_here' before running this script."
    AUTO_SHUTDOWN=false
    exit 1
fi
uv run wandb login "$WANDB_API_KEY"

# 6. Compute dataset examples for SFT and DPO
echo "Computing dataset sizes for '$DATASET_NAME'..."
eval "$(DATASET_NAME="$DATASET_NAME" DATA_FRACTION="$DATA_FRACTION" SFT_DATA_FRACTION="$SFT_DATA_FRACTION" uv run python - <<'PY'
import contextlib
import os
import sys
from preference_datasets import get_dataset

dataset_name = os.environ["DATASET_NAME"]
fraction = float(os.environ["DATA_FRACTION"])
sft_fraction = float(os.environ["SFT_DATA_FRACTION"])
with contextlib.redirect_stdout(sys.stderr):
    data = get_dataset(dataset_name, "train", silent=False)
total_prompts = len(data)
total_pairs = sum(len(v["pairs"]) for v in data.values())
sft_examples = max(1, int(total_prompts * sft_fraction))
dpo_examples = max(1, int(total_pairs * fraction))
print(f"SFT_N_EXAMPLES={sft_examples}")
print(f"DPO_N_EXAMPLES={dpo_examples}")
PY
)"
echo "Using SFT n_examples=$SFT_N_EXAMPLES (fraction=$SFT_DATA_FRACTION)"
echo "Using DPO n_examples=$DPO_N_EXAMPLES (fraction=$DATA_FRACTION)"

# 7. Run SFT
# Detect number of GPUs
GPUS=$(uv run python -c "import torch; print(torch.cuda.device_count())")
echo "Detected $GPUS GPUs."

if [[ "$RUN_SFT" == "true" ]]; then
    echo "Running SFT..."
    SFT_ARGS=(
      "model=$MODEL_NAME"
      "datasets=[$DATASET_NAME]"
      "loss=sft"
      "exp_name=$SFT_EXP_NAME"
      "trainer=$TRAINER"
      "n_examples=$SFT_N_EXAMPLES"
      "sample_during_eval=false"
    )
    uv run python -u train.py "${SFT_ARGS[@]}"
fi

if [[ "$RUN_SFT" == "true" || "$RUN_DPO" == "true" ]]; then
    echo "Resolving SFT checkpoint..."
    if [[ -n "$SFT_CHECKPOINT" ]]; then
        echo "Using provided SFT checkpoint: $SFT_CHECKPOINT"
    else
        SFT_CHECKPOINT=$(LOCAL_DIRS="$LOCAL_DIRS" SFT_EXP_NAME="$SFT_EXP_NAME" uv run python - <<'PY'
import glob
import os
from utils import get_local_dir

exp_name = os.environ["SFT_EXP_NAME"]
prefixes = os.environ["LOCAL_DIRS"].split(",")
base_dir = get_local_dir(prefixes)
pattern = os.path.join(base_dir, f"{exp_name}_*/LATEST/policy.pt")
candidates = glob.glob(pattern)
if not candidates:
    raise SystemExit(f"No SFT checkpoint found for exp_name={exp_name} in {base_dir}")
latest = max(candidates, key=os.path.getmtime)
print(latest)
PY
        )
        echo "Using SFT checkpoint: $SFT_CHECKPOINT"
    fi
fi

if [[ "$RUN_SFT" == "true" ]]; then
    SFT_RUN_DIR=$(LOCAL_DIRS="$LOCAL_DIRS" SFT_EXP_NAME="$SFT_EXP_NAME" uv run python - <<'PY'
import glob
import os
from utils import get_local_dir

exp_name = os.environ["SFT_EXP_NAME"]
prefixes = os.environ["LOCAL_DIRS"].split(",")
base_dir = get_local_dir(prefixes)
pattern = os.path.join(base_dir, f"{exp_name}_*")
candidates = [p for p in glob.glob(pattern) if os.path.isdir(p)]
if not candidates:
    raise SystemExit(f"No SFT run dir found for exp_name={exp_name} in {base_dir}")
latest = max(candidates, key=os.path.getmtime)
print(latest)
PY
    )
    echo "Using SFT run dir: $SFT_RUN_DIR"
fi

if [[ "$RUN_DPO" == "true" ]]; then
    if [[ -z "$SFT_CHECKPOINT" ]]; then
        echo "Error: SFT_CHECKPOINT is required to run DPO."
        AUTO_SHUTDOWN=false
        exit 1
    fi
    echo "Running DPO in regular mode..."
    DPO_ARGS=(
      "model=$MODEL_NAME"
      "datasets=[$DATASET_NAME]"
      "loss=dpo"
      "loss.beta=$LOSS_BETA"
      "loss.mode_loss=mean"
      "exp_name=$DPO_EXP_NAME"
      "trainer=$TRAINER"
      "n_examples=$DPO_N_EXAMPLES"
      "sample_during_eval=false"
      "model.archive=$SFT_CHECKPOINT"
    )
    uv run python -u train.py "${DPO_ARGS[@]}"
fi

echo "Training initialization sequence completed."

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory at $OUTPUT_DIR..."
    mkdir -p "$OUTPUT_DIR"
fi
if [ ! -w "$OUTPUT_DIR" ]; then
    echo "Error: $OUTPUT_DIR is not writable."
    AUTO_SHUTDOWN=false
    exit 1
fi

if [[ "$RUN_DPO" == "true" ]]; then
    DPO_RUN_DIR=$(LOCAL_DIRS="$LOCAL_DIRS" DPO_EXP_NAME="$DPO_EXP_NAME" uv run python - <<'PY'
import glob
import os
from utils import get_local_dir

exp_name = os.environ["DPO_EXP_NAME"]
prefixes = os.environ["LOCAL_DIRS"].split(",")
base_dir = get_local_dir(prefixes)
pattern = os.path.join(base_dir, f"{exp_name}_*")
candidates = [p for p in glob.glob(pattern) if os.path.isdir(p)]
if not candidates:
    raise SystemExit(f"No DPO run dir found for exp_name={exp_name} in {base_dir}")
latest = max(candidates, key=os.path.getmtime)
print(latest)
PY
    )
    echo "Using DPO run dir: $DPO_RUN_DIR"
fi

echo "Copying final outputs to $OUTPUT_DIR..."
if [[ "$RUN_SFT" == "true" ]]; then
    cp -R "$SFT_RUN_DIR" "$OUTPUT_DIR"/
fi
if [[ "$RUN_DPO" == "true" ]]; then
    cp -R "$DPO_RUN_DIR" "$OUTPUT_DIR"/
fi
echo "Copied run dirs to $OUTPUT_DIR"
