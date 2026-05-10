from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
for path in (ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from grid_search import InterventionConfig, load_grid_from_json, run_experiment, run_grid_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intervention experiment runner")
    parser.add_argument("--mode", choices=["single", "grid"], default="grid")
    parser.add_argument("--output-dir", type=Path, default=Path("results/grid_search/concat_onion"))
    parser.add_argument("--state-mode", choices=["c", "h", "both", "concat"], default="c")
    parser.add_argument("--scale-param", choices=["per_pos", "onion", "linear", "one_scale"], default="onion")
    parser.add_argument("--pretrained-embedding", default=False, action="store_true")
    parser.add_argument("--freeze-embedding", default=False, action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--max-seq-len", type=int, default=20)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "grid":
        grid = load_grid_from_json(ROOT / "grid_config.json")
        run_grid_search(grid, args.output_dir)
        return

    config = InterventionConfig(
        model_name="Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1",
        weights_path=str(Path("../reproduce/weights/1024_75.pth")),
        state_mode=args.state_mode,
        scale_param=args.scale_param,
        pretrained_embedding=args.pretrained_embedding,
        freeze_embedding=args.freeze_embedding,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        num_epochs=args.num_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        max_seq_len=args.max_seq_len,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    run_experiment(config, args.output_dir, verbose=True)


if __name__ == "__main__":
    main()
