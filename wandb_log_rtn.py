import argparse
import json
import os
import re
from pathlib import Path

try:
    import wandb
except ImportError:
    wandb = None


def directory_size_bytes(path):
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    for item in root.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def parse_ppl(path):
    metrics = {}
    text = Path(path).read_text(errors="ignore")
    for line in text.splitlines():
        match = re.match(
            r"^(WikiText-2|C4)\s+([0-9.]+)\s+([0-9.]+)\s+([-+0-9.]+)%\s+(\S+)",
            line.strip(),
        )
        if not match:
            continue
        dataset, flip_ppl, raw_ppl, delta_pct, winner = match.groups()
        dataset = dataset.replace("WikiText-2", "WikiText2")
        metrics[f"ppl/{dataset}/candidate"] = float(flip_ppl)
        metrics[f"ppl/{dataset}/baseline"] = float(raw_ppl)
        metrics[f"ppl/{dataset}/delta_pct"] = float(delta_pct)
        metrics[f"ppl/{dataset}/candidate_wins"] = 1.0 if winner.lower().startswith("heur") else 0.0
    return metrics


def find_result_json(path):
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(candidate for candidate in p.rglob("*.json") if candidate.is_file())
    return []


def parse_lm_eval(path):
    metrics = {}
    for result_file in find_result_json(path):
        try:
            payload = json.loads(result_file.read_text())
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
    values = [value for key, value in metrics.items() if key.startswith("lm_eval/")]
    if values:
        metrics["lm_eval/avg"] = sum(values) / len(values)
    return metrics


def add_file(run, path, artifact_type):
    if path is None:
        return
    p = Path(path)
    if not p.exists() or not p.is_file():
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
    parser.add_argument("--knee", type=float, default=None)
    parser.add_argument("--max-flip", type=float, default=None)
    parser.add_argument("--log-path")
    parser.add_argument("--json-path")
    parser.add_argument("--model-dir")
    parser.add_argument("--project", default=os.getenv("WANDB_PROJECT", "fprag_rtn"))
    args = parser.parse_args()

    if wandb is None or os.getenv("WANDB_MODE") == "disabled":
        return

    config = {
        "bits": args.bits,
        "group_size": args.group_size,
        "model_slug": args.model_slug,
    }
    if args.knee is not None:
        config["knee"] = args.knee
    if args.max_flip is not None:
        config["max_flip"] = args.max_flip

    run = wandb.init(
        project=args.project,
        name=args.name,
        job_type=args.kind,
        tags=[args.kind, f"bits:{args.bits}", f"model:{args.model_slug}"],
        config=config,
    )

    metrics = {}
    if args.kind == "quantize":
        metrics["quantize/done"] = 1.0
        if args.model_dir:
            metrics["quantize/model_dir_exists"] = 1.0 if Path(args.model_dir).exists() else 0.0
            metrics["quantize/model_dir_gb"] = directory_size_bytes(args.model_dir) / (1024 ** 3)
    elif args.kind == "ppl" and args.log_path:
        metrics.update(parse_ppl(args.log_path))
    elif args.kind == "lm_eval" and args.json_path:
        metrics.update(parse_lm_eval(args.json_path))

    if metrics:
        wandb.log(metrics)
    add_file(run, args.log_path, "log")
    add_file(run, args.json_path, "eval-json")
    run.finish()


if __name__ == "__main__":
    main()
