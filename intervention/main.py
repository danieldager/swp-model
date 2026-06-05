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
from grid_search import InterventionConfig, load_grid_from_json, run_grid_search,  update_grid_summary_row, make_summary_row, make_run_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intervention experiment runner")
    parser.add_argument("--mode", choices=["single", "grid"], default="grid")
    parser.add_argument("--save_dir", type=str, default="results/dataset_conditions")
    # parser.add_argument("--skip_existing", action="store_true", default=True)
    # parser.add_argument("--v", action="store_true", help="Print detailed logs during execution")
    return parser.parse_args()

# per_pos: 78%,  spiral_decel:0.7244, spiral lie: 70, onion without rot: 69, spiral rope:68
def main() -> None:
    
    args = parse_args()
    output_dir = Path(args.save_dir)
    if args.mode == "grid":
        grid = load_grid_from_json(ROOT / "grid_config.json")
        run_grid_search(grid, output_dir, skip_existing=False,verbose=True)
        return

    config = InterventionConfig(
        model_name="Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1",
        weights_path=str(Path("../reproduce/weights/1024_75.pth")),
        state_mode='c',
        scale_param='per_pos',
        embedding_init='pretrained',
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
        dataset_type='pseudo-real',
        lexicality_col='Lexicality',
    )

    run_dir = output_dir / make_run_name(config)
    history = run_experiment(config, run_dir, verbose=True)
    update_grid_summary_row(make_summary_row(config, run_dir, history), output_dir)


if __name__ == "__main__":
    main()
