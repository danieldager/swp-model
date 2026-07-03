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
    parser.add_argument("--save_dir", type=str, default="results/final_grid_search_repeat_only")
    # parser.add_argument("--skip_existing", action="store_true", default=True)
    # parser.add_argument("--v", action="store_true", help="Print detailed logs during execution")
    return parser.parse_args()


def main() -> None:
    
    args = parse_args()
    output_dir = Path(args.save_dir)
    if args.mode == "grid":
        grid = load_grid_from_json(ROOT / "grid_config.json")
        run_grid_search(grid, output_dir, skip_existing=True, verbose=True)
        return

    config = InterventionConfig(
        model_name="Ua_LSTM_h128_l1_v42_d0.0_t0.0_s1",
        weights_path=str(Path("../reproduce/weights/1024_75.pth")),
        state_mode='c', # 'h', 'concat', 'c'
        scale_param='low_rank-8', # 'low_rank-n', 'onion','expo_decay', 'spiral_rope', 'per_pos', 'spiral_lie','linear'
        embedding_init='delta_c_mean', #'none', 'delta_{state/h/c}_{mean/median}', 'pretrained'
        dataset_type='real-real', # 'modified-source', 'source-modified', 'real-real'
        check_repeat=True,
        check_cv=False,
        check_n_gram=0, # 0 for no n-gram check, otherwise 2 for bi-gram, 3 for trigram
        train_embedding=False,
        teacher_forcing=False,
        train_all_pos=False,
        learning_rate=1e-3,
        batch_size=32,
        hidden_size=128,
        num_epochs=200,
        patience=7,
        min_delta=1e-6,
        max_seq_len=20,
        val_ratio=0.05, # not for real-real since we will use the original val set
        seed=42,
    )

    
    run = "check_repeat+cv+bi-" + make_run_name(config)
    run_dir = output_dir / run
    history = run_experiment(config, run_dir, verbose=True)
    update_grid_summary_row(make_summary_row(config, run_dir, history), output_dir)


if __name__ == "__main__":
    main()
