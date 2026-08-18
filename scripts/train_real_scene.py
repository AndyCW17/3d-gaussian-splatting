"""
Trains a Gaussian scene on real COLMAP data from the command line.

Run inside Docker (needs GPU):
    PYTHONPATH=src python3 scripts/train_real_scene.py \
        data/raw/mug/sparse_txt data/raw/mug outputs/mug_training
"""

import sys
from splatproj.pipeline import run_pipeline


def main():
    sparse_folder, images_folder, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]

    def on_iteration(iteration, loss_value, model):
        if iteration % 100 == 0:
            print(f"iter {iteration:5d} | loss {loss_value:.4f} | n_gaussians {len(model)}")

    ply_path = run_pipeline(
        sparse_folder, images_folder, output_dir,
        num_iterations=3000, device="cuda", on_iteration=on_iteration,
    )

    print(f"Done. Exported PLY: {ply_path}")


if __name__ == "__main__":
    main()