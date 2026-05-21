#!/usr/bin/env bash
set -euo pipefail

# Run in the root of anhnda/FPRAG on branch `rtn`.
# W4G128 RTN baseline and RTN+JS/heuristic flip for:
#   - Meta-Llama-3.1-8B
#   - Qwen2.5-7B
#
# Usage:
#   bash fprag_rtn_w4g128.sh --ppl
#   bash fprag_rtn_w4g128.sh --lm-eval
#   bash fprag_rtn_w4g128.sh --ppl --lm-eval
#   bash fprag_rtn_w4g128.sh --tune-flip --ppl --lm-eval

BITS=4
GROUP_SIZE=128
KNEE="${KNEE:-0.0}"
MAX_FLIP="${MAX_FLIP:-0.05}"
KNEE_VALUES="${KNEE_VALUES:-0.0 0.01 0.02 0.03 0.04 0.05}"
MAX_FLIP_VALUES="${MAX_FLIP_VALUES:-0.01 0.02 0.03 0.04 0.05}"
START_KNEE="${START_KNEE:-}"
START_FLIP="${START_FLIP:-}"
N_CALIB="${N_CALIB:-128}"
CALIB_DATASET="${CALIB_DATASET:-c4}"
PPL_SAMPLES="${PPL_SAMPLES:-500}"
TASKS="${TASKS:-arc_challenge,arc_easy,boolq,piqa,rte}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-./hf_datasets_cache_rtn_w4g128}"
WANDB_PROJECT="${WANDB_PROJECT:-fprag_rtn_w4g128}"
RUN_PPL=0
RUN_LM_EVAL=0
TUNE_FLIP=0
FORCE="${FORCE:-0}"
SAVE_CONTINUE_TAR="${SAVE_CONTINUE_TAR:-0}"
LOG_QUANTIZE_WANDB="${LOG_QUANTIZE_WANDB:-0}"
DELETE_FLIP_AFTER_EVAL="${DELETE_FLIP_AFTER_EVAL:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ppl) RUN_PPL=1; shift ;;
    --lm-eval) RUN_LM_EVAL=1; shift ;;
    --tune-flip) TUNE_FLIP=1; shift ;;
    --force) FORCE=1; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ "$N_CALIB" != "128" ]]; then
  echo "N_CALIB must stay 128 for RTN flip tuning; got N_CALIB=$N_CALIB." >&2
  exit 2
fi

if [[ ! -f rtn.py || ! -f rtn_js_xl.py || ! -f compare_slicing.py ]]; then
  echo "Run this script inside anhnda/FPRAG branch rtn." >&2
  exit 1
fi

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_DATASETS_CACHE
unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE

resolve_hf_token() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf '%s' "$HF_TOKEN"
    return
  fi
  if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    printf '%s' "$HUGGINGFACE_HUB_TOKEN"
    return
  fi
  if [[ -n "${HF_HUB_TOKEN:-}" ]]; then
    printf '%s' "$HF_HUB_TOKEN"
    return
  fi
  local default_hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
  if [[ -f "$default_hf_home/token" ]]; then
    tr -d '\n' < "$default_hf_home/token"
    return
  fi
  if [[ -f "$HOME/.cache/huggingface/token" ]]; then
    tr -d '\n' < "$HOME/.cache/huggingface/token"
  fi
}

HF_AUTH_TOKEN="$(resolve_hf_token || true)"
if [[ -n "$HF_AUTH_TOKEN" ]]; then
  export HF_TOKEN="$HF_AUTH_TOKEN"
  export HUGGINGFACE_HUB_TOKEN="$HF_AUTH_TOKEN"
  export HF_HUB_TOKEN="$HF_AUTH_TOKEN"
fi

prepare_hf_home() {
  local hf_home="$1"
  mkdir -p "$hf_home"
  if [[ -n "${HF_AUTH_TOKEN:-}" ]]; then
    printf '%s' "$HF_AUTH_TOKEN" > "$hf_home/token"
    chmod 600 "$hf_home/token" || true
  fi
}

mkdir -p quantized_models logs dataset_cache calibration_cache eval_results_rtn_w4g128 artifacts "$HF_DATASETS_CACHE"

MODELS=(
  "meta-llama/Meta-Llama-3.1-8B"
  "Qwen/Qwen2.5-7B"
)

if [[ -n "${MODEL_PATH:-}" ]]; then
  MODELS=("$MODEL_PATH")
fi

slugify() {
  local model="$1"
  local slug="${model##*/}"
  slug="${slug//./p}"
  slug="${slug//-/_}"
  echo "$slug"
}

model_artifact_complete() {
  local model_dir="$1"
  [[ -f "$model_dir/config.json" ]] || return 1
  [[ -f "$model_dir/model.safetensors" ]] && return 0
  [[ -f "$model_dir/pytorch_model.bin" ]] && return 0
  [[ -f "$model_dir/model.safetensors.index.json" ]] && compgen -G "$model_dir/*.safetensors" >/dev/null && return 0
  [[ -f "$model_dir/pytorch_model.bin.index.json" ]] && compgen -G "$model_dir/pytorch_model*.bin" >/dev/null && return 0
  compgen -G "$model_dir/model-*.safetensors" >/dev/null && return 0
  return 1
}

remove_incomplete_artifact() {
  local model_dir="$1"
  if [[ -d "$model_dir" ]] && ! model_artifact_complete "$model_dir"; then
    echo "Removing incomplete artifact without model weights: $model_dir"
    rm -rf "$model_dir"
  fi
}

ensure_wandb_helper() {
  cat > wandb_log_rtn.py <<'PY'
import argparse
import json
import os
import re
from pathlib import Path

try:
    import wandb
except ImportError:
    wandb = None

TASKS = [
    "arc_challenge",
    "arc_easy",
    "boolq",
    "piqa",
    "rte",
]

def parse_ppl(path):
    metrics = {}
    text = Path(path).read_text(errors="ignore")
    for line in text.splitlines():
        match = re.match(r"^(WikiText-2|C4)\s+([0-9.]+)\s+([0-9.]+)\s+([-+0-9.]+)%\s+(\S+)", line.strip())
        if not match:
            continue
        dataset, flip_ppl, raw_ppl, delta_pct, winner = match.groups()
        dataset = dataset.replace("WikiText-2", "WikiText2")
        metrics[f"ppl/{dataset}/rtn_flip"] = float(flip_ppl)
        metrics[f"ppl/{dataset}/rtn"] = float(raw_ppl)
        metrics[f"ppl/{dataset}/delta_pct"] = float(delta_pct)
        metrics[f"ppl/{dataset}/flip_wins"] = 1.0 if winner.lower().startswith("heur") else 0.0
    return metrics

def find_result_json(path):
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        candidates = sorted(candidate for candidate in p.rglob("*.json") if candidate.is_file())
        return candidates
    return []

def parse_lm_eval(path):
    metrics = {}
    result_files = find_result_json(path)
    if not result_files:
        return {}, None
    for p in result_files:
        try:
            payload = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        results = payload.get("results", payload)
        if not isinstance(results, dict):
            continue
        for task, item in results.items():
            if not isinstance(item, dict):
                continue
            for key in ("acc,none", "acc", "acc_norm,none", "acc_norm"):
                value = item.get(key)
                if isinstance(value, (int, float)):
                    metrics[f"lm_eval/{task}"] = float(value)
                    break
    values = list(metrics.values())
    if values:
        metrics["lm_eval/avg"] = sum(values) / len(values)
    return metrics, result_files[0]

def add_file(run, path, artifact_type):
    if path is None:
        return
    p = Path(path)
    if not p.exists():
        return
    wandb.save(str(p))
    artifact = wandb.Artifact(f"{run.name}-{artifact_type}", type=artifact_type)
    artifact.add_file(str(p), name=p.name)
    run.log_artifact(artifact)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=["quantize", "ppl", "lm_eval"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--bits", type=int, required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--knee", type=float, default=0.0)
    parser.add_argument("--max-flip", type=float, default=0.05)
    parser.add_argument("--log-path")
    parser.add_argument("--json-path")
    parser.add_argument("--project", default=os.getenv("WANDB_PROJECT", "fprag_rtn"))
    args = parser.parse_args()

    if wandb is None or os.getenv("WANDB_MODE") == "disabled":
        return

    run = wandb.init(
        project=args.project,
        name=args.name,
        job_type=args.kind,
        tags=[args.kind, f"bits:{args.bits}", f"model:{args.model_slug}"],
        config={
            "bits": args.bits,
            "group_size": args.group_size,
            "knee": args.knee,
            "max_flip": args.max_flip,
            "model_slug": args.model_slug,
        },
    )
    metrics = {}
    result_json = None
    if args.kind == "quantize":
        metrics["quantize/done"] = 1.0
    if args.kind == "ppl" and args.log_path:
        metrics.update(parse_ppl(args.log_path))
    if args.kind == "lm_eval" and args.json_path:
        parsed, result_json = parse_lm_eval(args.json_path)
        metrics.update(parsed)
    if metrics:
        wandb.log(metrics)
    add_file(run, args.log_path, "log")
    add_file(run, result_json or args.json_path, "eval-json")
    run.finish()

if __name__ == "__main__":
    main()
PY
}

log_wandb() {
  python wandb_log_rtn.py "$@" --project "$WANDB_PROJECT" || true
}

ensure_piqa_helper() {
  cat > piqa_manual_eval.py <<'PY'
import argparse
import json
import os
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PIQA_URLS = [
    "https://huggingface.co/datasets/lighteval/piqa/resolve/main/plain_text/validation-00000-of-00001.parquet",
    "https://huggingface.co/datasets/regisss/piqa/resolve/main/data/validation-00000-of-00001.parquet",
]


def download_piqa(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "piqa_validation.parquet"
    if target.exists() and target.stat().st_size > 0:
        return target

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_HUB_TOKEN")
    last_error = None
    for url in PIQA_URLS:
        tmp = target.with_suffix(".tmp")
        try:
            request = urllib.request.Request(url)
            if token:
                request.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(request) as response, open(tmp, "wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            tmp.replace(target)
            return target
        except Exception as exc:
            last_error = exc
            if tmp.exists():
                tmp.unlink()
    raise RuntimeError(f"Could not download PIQA validation parquet: {last_error}")


def load_rows(cache_dir: Path):
    path = download_piqa(cache_dir)
    table = pq.read_table(path)
    return table.to_pylist()


def continuation_score(model, tokenizer, prompt: str, continuation: str, device: str):
    prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids[0]
    cont_ids = tokenizer(" " + continuation, add_special_tokens=False, return_tensors="pt").input_ids[0]
    input_ids = torch.cat([prompt_ids, cont_ids], dim=0).unsqueeze(0).to(device)
    cont_len = cont_ids.numel()
    if cont_len == 0:
        return float("-inf"), float("-inf")

    with torch.no_grad():
        logits = model(input_ids).logits[:, :-1, :]
        targets = input_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        cont_log_probs = token_log_probs[:, -cont_len:]
        total = cont_log_probs.sum().item()
    return total, total / cont_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--cache-dir", default="./dataset_cache/piqa_manual")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    rows = load_rows(Path(args.cache_dir))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        device_map={"": args.device} if args.device.startswith("cuda") else None,
        trust_remote_code=True,
    )
    model.eval()

    correct = 0
    correct_norm = 0
    total = 0
    for row in tqdm(rows, desc="Manual PIQA"):
        goal = row.get("goal")
        sol1 = row.get("sol1")
        sol2 = row.get("sol2")
        label = int(row.get("label"))
        prompt = f"Question: {goal}\nAnswer:"
        s1, s1n = continuation_score(model, tokenizer, prompt, sol1, args.device)
        s2, s2n = continuation_score(model, tokenizer, prompt, sol2, args.device)
        correct += int((0 if s1 > s2 else 1) == label)
        correct_norm += int((0 if s1n > s2n else 1) == label)
        total += 1

    payload = {
        "results": {
            "piqa": {
                "acc,none": correct / total,
                "acc_norm,none": correct_norm / total,
            }
        },
        "versions": {"piqa": "manual-parquet"},
        "n-shot": {"piqa": 0},
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["results"]["piqa"], indent=2))


if __name__ == "__main__":
    main()
PY
}

run_quantize_model() {
  local model="$1"
  local slug="$2"
  local rtn_dir="./quantized_models/${slug}_rtn_w${BITS}g${GROUP_SIZE}"
  local flip_dir="./quantized_models/${slug}_rtn_js_xl_w${BITS}g${GROUP_SIZE}_k${KNEE}_f${MAX_FLIP}"
  local rtn_log="logs/${slug}_rtn_w${BITS}g${GROUP_SIZE}.log"
  local flip_log="logs/${slug}_rtn_js_xl_w${BITS}g${GROUP_SIZE}_k${KNEE}_f${MAX_FLIP}.log"

  remove_incomplete_artifact "$rtn_dir"
  if [[ "$FORCE" == "1" ]] || ! model_artifact_complete "$rtn_dir"; then
    python rtn.py \
      --model-path "$model" \
      --bits "$BITS" \
      --group-size "$GROUP_SIZE" \
      --output-dir "$rtn_dir" \
      2>&1 | tee "$rtn_log"
  else
    echo "Skip existing RTN artifact: $rtn_dir"
  fi
  if [[ "$LOG_QUANTIZE_WANDB" == "1" ]]; then
    log_wandb --kind quantize --name "${slug}_rtn_w${BITS}g${GROUP_SIZE}" --model-slug "$slug" --bits "$BITS" --group-size "$GROUP_SIZE" --log-path "$rtn_log"
  fi

  remove_incomplete_artifact "$flip_dir"
  if [[ "$FORCE" == "1" ]] || ! model_artifact_complete "$flip_dir"; then
    python rtn_js_xl.py \
      --model-path "$model" \
      --bits "$BITS" \
      --group-size "$GROUP_SIZE" \
      --n-calib "$N_CALIB" \
      --calib-dataset "$CALIB_DATASET" \
      --cache-dir "./calibration_cache" \
      --knee-tolerance "$KNEE" \
      --max-flip-percent "$MAX_FLIP" \
      --output-dir "$flip_dir" \
      2>&1 | tee "$flip_log"
  else
    echo "Skip existing RTN+Flip artifact: $flip_dir"
  fi
  if [[ "$LOG_QUANTIZE_WANDB" == "1" ]]; then
    log_wandb --kind quantize --name "${slug}_rtn_js_xl_w${BITS}g${GROUP_SIZE}_k${KNEE}_f${MAX_FLIP}" --model-slug "$slug" --bits "$BITS" --group-size "$GROUP_SIZE" --knee "$KNEE" --max-flip "$MAX_FLIP" --log-path "$flip_log"
  fi
}

run_ppl_model() {
  local slug="$1"
  local rtn_dir="./quantized_models/${slug}_rtn_w${BITS}g${GROUP_SIZE}"
  local flip_dir="./quantized_models/${slug}_rtn_js_xl_w${BITS}g${GROUP_SIZE}_k${KNEE}_f${MAX_FLIP}"
  local log_path="logs/${slug}_compare_ppl_w${BITS}g${GROUP_SIZE}_k${KNEE}_f${MAX_FLIP}.log"
  local ran_ppl=0

  if [[ "$FORCE" == "1" || ! -f "$log_path" ]]; then
    python compare_slicing.py \
      --heuristic-path "$flip_dir" \
      --standard-path "$rtn_dir" \
      --n-samples "$PPL_SAMPLES" \
      --cache-dir "./dataset_cache" \
      2>&1 | tee "$log_path"
    ran_ppl=1
  else
    echo "Skip existing PPL log: $log_path"
  fi
  if [[ "$ran_ppl" == "1" ]]; then
    log_wandb --kind ppl --name "${slug}_ppl_w${BITS}g${GROUP_SIZE}_k${KNEE}_f${MAX_FLIP}" --model-slug "$slug" --bits "$BITS" --group-size "$GROUP_SIZE" --knee "$KNEE" --max-flip "$MAX_FLIP" --log-path "$log_path"
  fi
}

eval_one() {
  local name="$1"
  local path="$2"
  local slug="$3"
  local tokenizer_path="$4"
  local output="./eval_results_rtn_w${BITS}g${GROUP_SIZE}/${name}"
  local log_path="logs/${name}_lm_eval.log"
  local task
  local task_output
  local task_log
  local task_cache
  local task_hf_home
  local task_datasets_cache
  local status
  local ran_lm_eval=0
  local log_initialized=0
  local -a task_list

  mkdir -p "$output"
  IFS=',' read -ra task_list <<< "$TASKS"

  for task in "${task_list[@]}"; do
    task_output="$output/${task}_results.json"
    task_log="logs/${name}_${task}_lm_eval.log"
    task_cache="${HF_DATASETS_CACHE}/${name}/${task}"
    task_hf_home="${task_cache}/hf_home"
    task_datasets_cache="${task_cache}/datasets"
    if [[ "$FORCE" != "1" && -e "$task_output" ]]; then
      echo "Skip existing lm_eval result: $task_output"
      continue
    fi
    if [[ "$log_initialized" != "1" ]]; then
      : > "$log_path"
      log_initialized=1
    fi
    ran_lm_eval=1
    mkdir -p "$task_hf_home" "$task_datasets_cache"
    prepare_hf_home "$task_hf_home"

    if [[ "$task" == "piqa" ]]; then
      set +e
      HF_HOME="$task_hf_home" HF_DATASETS_CACHE="$task_datasets_cache" python piqa_manual_eval.py \
        --model-path "$path" \
        --tokenizer-path "$tokenizer_path" \
        --output-json "$task_output" \
        --cache-dir "$task_cache/piqa_data" \
        --device cuda:0 \
        2>&1 | tee "$task_log"
      status="${PIPESTATUS[0]}"
      set -e
      cat "$task_log" >> "$log_path"
      if [[ "$status" != "0" ]]; then
        echo "manual PIQA failed for $name" | tee -a "$log_path"
        return "$status"
      fi
      continue
    fi

    set +e
    HF_HOME="$task_hf_home" HF_DATASETS_CACHE="$task_datasets_cache" python -m lm_eval --model hf \
      --model_args pretrained="$path",tokenizer="$tokenizer_path",trust_remote_code=True \
      --tasks "$task" \
      --device cuda:0 \
      --batch_size auto \
      --output_path "$task_output" \
      2>&1 | tee "$task_log"
    status="${PIPESTATUS[0]}"
    set -e
    cat "$task_log" >> "$log_path"

    if [[ "$status" != "0" ]]; then
      echo "lm_eval failed for $name task=$task. Clearing task HF datasets cache and retrying once..." | tee -a "$log_path"
      rm -rf "$task_cache"
      mkdir -p "$task_hf_home" "$task_datasets_cache"
      prepare_hf_home "$task_hf_home"
      HF_HOME="$task_hf_home" HF_DATASETS_CACHE="$task_datasets_cache" python -m lm_eval --model hf \
        --model_args pretrained="$path",tokenizer="$tokenizer_path",trust_remote_code=True \
        --tasks "$task" \
        --device cuda:0 \
        --batch_size auto \
        --output_path "$task_output" \
        2>&1 | tee "$task_log"
      cat "$task_log" >> "$log_path"
    fi
  done
  if [[ "$ran_lm_eval" == "1" ]]; then
    log_wandb --kind lm_eval --name "${name}_lm_eval" --model-slug "$slug" --bits "$BITS" --group-size "$GROUP_SIZE" --knee "$KNEE" --max-flip "$MAX_FLIP" --log-path "$log_path" --json-path "$output"
  else
    echo "Skip W&B lm_eval log: no new tasks for $name"
  fi
}

run_lm_eval_model() {
  local slug="$1"
  local tokenizer_path="$2"
  local rtn_dir="./quantized_models/${slug}_rtn_w${BITS}g${GROUP_SIZE}"
  local flip_dir="./quantized_models/${slug}_rtn_js_xl_w${BITS}g${GROUP_SIZE}_k${KNEE}_f${MAX_FLIP}"
  local rtn_name="${slug}_rtn_w${BITS}g${GROUP_SIZE}"
  if [[ "$FORCE" != "1" ]] && lm_eval_outputs_complete_for_name "$rtn_name"; then
    echo "Skip completed RTN baseline lm_eval/logging: $rtn_name"
  else
    eval_one "$rtn_name" "$rtn_dir" "$slug" "$tokenizer_path"
  fi
  eval_one "${slug}_rtn_js_xl_w${BITS}g${GROUP_SIZE}_k${KNEE}_f${MAX_FLIP}" "$flip_dir" "$slug" "$tokenizer_path"
}

cleanup_flip_artifact_if_requested() {
  local slug="$1"
  local flip_dir="./quantized_models/${slug}_rtn_js_xl_w${BITS}g${GROUP_SIZE}_k${KNEE}_f${MAX_FLIP}"

  if [[ "$DELETE_FLIP_AFTER_EVAL" != "1" ]]; then
    return 0
  fi
  if [[ "$RUN_PPL" != "1" && "$RUN_LM_EVAL" != "1" ]]; then
    echo "DELETE_FLIP_AFTER_EVAL=1 ignored because neither --ppl nor --lm-eval was requested."
    return 0
  fi
  if [[ -d "$flip_dir" ]]; then
    echo "Deleting heavy RTN+Flip artifact after eval/logging: $flip_dir"
    rm -rf "$flip_dir"
  fi
}

lm_eval_outputs_complete_for_name() {
  local name="$1"
  local output="./eval_results_rtn_w${BITS}g${GROUP_SIZE}/${name}"
  local task
  local task_output
  local -a task_list

  IFS=',' read -ra task_list <<< "$TASKS"
  for task in "${task_list[@]}"; do
    task_output="$output/${task}_results.json"
    if [[ ! -f "$task_output" ]]; then
      return 1
    fi
  done
  return 0
}

config_outputs_complete() {
  local slug="$1"
  local have_requested_output=0
  local ppl_log="logs/${slug}_compare_ppl_w${BITS}g${GROUP_SIZE}_k${KNEE}_f${MAX_FLIP}.log"
  local flip_name="${slug}_rtn_js_xl_w${BITS}g${GROUP_SIZE}_k${KNEE}_f${MAX_FLIP}"

  if [[ "$FORCE" == "1" ]]; then
    return 1
  fi
  if [[ "$RUN_PPL" == "1" ]]; then
    have_requested_output=1
    [[ -f "$ppl_log" ]] || return 1
  fi
  if [[ "$RUN_LM_EVAL" == "1" ]]; then
    have_requested_output=1
    lm_eval_outputs_complete_for_name "$flip_name" || return 1
  fi
  [[ "$have_requested_output" == "1" ]]
}

run_model_config() {
  local model="$1"
  local slug="$2"

  echo "================================================================"
  echo "MODEL=$model"
  echo "SLUG=$slug"
  echo "BITS=$BITS GROUP_SIZE=$GROUP_SIZE KNEE=$KNEE MAX_FLIP=$MAX_FLIP"
  echo "================================================================"
  if config_outputs_complete "$slug"; then
    echo "Skip completed RTN+Flip config outputs: k=$KNEE f=$MAX_FLIP"
    return 0
  fi
  run_quantize_model "$model" "$slug"
  if [[ "$RUN_PPL" == "1" ]]; then
    run_ppl_model "$slug"
  fi
  if [[ "$RUN_LM_EVAL" == "1" ]]; then
    run_lm_eval_model "$slug" "$model"
  fi
  cleanup_flip_artifact_if_requested "$slug"
}

run_model_tune_grid() {
  local model="$1"
  local slug="$2"
  local knee
  local flip
  local resume_gate_open=1
  local resume_point_found=0
  local -a knee_grid
  local -a flip_grid

  read -r -a knee_grid <<< "$KNEE_VALUES"
  read -r -a flip_grid <<< "$MAX_FLIP_VALUES"

  if [[ -n "$START_KNEE" || -n "$START_FLIP" ]]; then
    if [[ -z "$START_KNEE" || -z "$START_FLIP" ]]; then
      echo "Both START_KNEE and START_FLIP must be set for tune resume." >&2
      exit 2
    fi
    for knee in "${knee_grid[@]}"; do
      for flip in "${flip_grid[@]}"; do
        if [[ "$knee" == "$START_KNEE" && "$flip" == "$START_FLIP" ]]; then
          resume_point_found=1
        fi
      done
    done
    if [[ "$resume_point_found" != "1" ]]; then
      echo "Invalid resume point: START_KNEE=$START_KNEE START_FLIP=$START_FLIP" >&2
      echo "Valid KNEE_VALUES: ${knee_grid[*]}" >&2
      echo "Valid MAX_FLIP_VALUES: ${flip_grid[*]}" >&2
      exit 2
    fi
    resume_gate_open=0
  fi

  for knee in "${knee_grid[@]}"; do
    for flip in "${flip_grid[@]}"; do
      if [[ "$resume_gate_open" == "0" ]]; then
        if [[ "$knee" == "$START_KNEE" && "$flip" == "$START_FLIP" ]]; then
          resume_gate_open=1
        else
          echo "Skipping before resume point: k=$knee f=$flip"
          continue
        fi
      fi
      KNEE="$knee"
      MAX_FLIP="$flip"
      run_model_config "$model" "$slug"
    done
  done
}

ensure_wandb_helper
ensure_piqa_helper

for model in "${MODELS[@]}"; do
  slug="$(slugify "$model")"
  if [[ "$TUNE_FLIP" == "1" ]]; then
    run_model_tune_grid "$model" "$slug"
  else
    run_model_config "$model" "$slug"
  fi
done

if [[ "$SAVE_CONTINUE_TAR" == "1" ]]; then
  tar -czf "artifacts/fprag_rtn_w${BITS}g${GROUP_SIZE}_continue_minimal.tar.gz" \
    quantized_models "eval_results_rtn_w${BITS}g${GROUP_SIZE}" logs calibration_cache dataset_cache \
    2>/dev/null || true
  echo "Done. Continue artifact: artifacts/fprag_rtn_w${BITS}g${GROUP_SIZE}_continue_minimal.tar.gz"
else
  echo "Done. SAVE_CONTINUE_TAR=0, skipped tar to avoid filling disk."
fi
