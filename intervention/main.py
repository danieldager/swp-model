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

from experiment import run_experiment
from grid_search import InterventionConfig, load_grid_from_json, run_grid_search, run_grid_search_parallel, update_grid_summary_row, make_summary_row, make_run_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intervention experiment runner")
    parser.add_argument("--mode", choices=["single", "grid"], default="grid")
    parser.add_argument("--save_dir", type=str, default="results/grid_search")
    return parser.parse_args()


def main() -> None:
    
    args = parse_args()
    output_dir = Path(args.save_dir)
    if args.mode == "grid":
        grid = load_grid_from_json(ROOT / "grid_config.json")
        run_grid_search_parallel(grid, output_dir, skip_existing=False, n_jobs=-1)
        return

    config = InterventionConfig(
        model_name="Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1",
        weights_path=str(Path("../reproduce/weights/1024_75.pth")),
        state_mode='c',
        scale_param='per_pos',
        pretrained_embedding=True,
        train_embedding=True,
        teacher_forcing=False,
        train_all_pos=False,
        learning_rate=1e-3,
        batch_size=128,
        hidden_size=128,
        num_epochs=200,
        patience=7,
        min_delta=1e-6,
        max_seq_len=20,
        val_ratio=0.05,
        seed=42,
    )

    run_dir = output_dir / make_run_name(config)
    start_time = os.times()
    history = run_experiment(config, run_dir, verbose=True)
    end_time = os.times()
    print(f"grid search completed in {end_time[4] - start_time[4]:.2f} seconds")
    update_grid_summary_row(make_summary_row(config, run_dir, history), output_dir)


if __name__ == "__main__":
    main()
