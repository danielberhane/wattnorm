"""Train + calibrate the detector on the configured splits, log to MLflow, register as staging.

uv run python scripts/train.py --config configs/default.yaml
"""

import argparse
from datetime import UTC, datetime

import mlflow

from smartbuilding.config import Env, load_config
from smartbuilding.data.splits import temporal_split
from smartbuilding.pipeline import load_dataset, train_detector
from smartbuilding.registry import log_and_register


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--alias", default="staging")
    ap.add_argument("--local-out", default=None, help="also save the artifact directory here")
    args = ap.parse_args()

    cfg = load_config(args.config)
    mlflow.set_tracking_uri(Env().mlflow_tracking_uri)
    mlflow.set_experiment("smartbuilding")

    data = load_dataset(cfg, cfg.splits.train[0], cfg.splits.calibrate[1])
    parts = temporal_split(data, {"train": cfg.splits.train, "calibrate": cfg.splits.calibrate})
    version = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    detector, metrics = train_detector(
        parts["train"], parts["calibrate"], cfg.model, cfg.rules, cfg.site.tz, version
    )
    for k, v in metrics.items():
        print(f"{k:22s} {v:.4f}" if isinstance(v, float) else f"{k:22s} {v}")

    params = {
        **cfg.model.model_dump(),
        **{f"rule_{k}": v for k, v in cfg.rules.model_dump().items()},
        "train": "→".join(cfg.splits.train),
        "calibrate": "→".join(cfg.splits.calibrate),
    }
    mv = log_and_register(detector, params, metrics, alias=args.alias, run_name=f"train-{version}")
    print(f"registered {mv.name} v{mv.version} → alias '{args.alias}'")
    if args.local_out:
        detector.save(args.local_out)
        print(f"saved to {args.local_out}")


if __name__ == "__main__":
    main()
