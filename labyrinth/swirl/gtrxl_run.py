import numpy as np
from pathlib import Path

# Import ONLY callable utilities
from gtrxl_func import TrainConfig, train


def make_npz_from_emissions(folder: str) -> str:
    """
    Load emissions500new.npy and create a lightweight npz wrapper
    for GTrXL training.
    """
    folder = Path(folder)

    emissions_path = folder / "emissions500new.npy"
    out_npz = folder / "labyrinth_gtrxl.npz"

    emissions = np.load(emissions_path)
    xs = emissions[:, :, 0].astype(np.int64)
    acs = emissions[:, :, 1].astype(np.int64)

    np.savez_compressed(out_npz, xs=xs, acs=acs)

    print("=== Dataset summary ===")
    print(f"xs shape:  {xs.shape}")
    print(f"acs shape: {acs.shape}")
    print(f"# trajectories: {xs.shape[0]}")
    print(f"trajectory length T: {xs.shape[1]}")
    print(f"total tokens (N*T): {xs.shape[0] * xs.shape[1]}")
    print("=======================")

    return str(out_npz)


def quick_sanity_run(npz_path: str):
    """
    Short run to check:
    - loss decreases
    - next-state accuracy > random
    """
    print("\nRunning quick sanity run...")

    cfg = TrainConfig(
        npz_path=npz_path,
        out_dir="runs/gtrxl_sanity",
        epochs=3,
        batch_size=32,
        chunk_len=64,
        horizon_H=20,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        dropout=0.1,
        lambda_occ=0.5,
    )

    train(cfg)


def overfit_test(npz_path: str):
    """
    Overfitting test on small batches.
    This SHOULD overfit if the model is correct.
    """
    print("\nRunning overfitting test...")

    cfg = TrainConfig(
        npz_path=npz_path,
        out_dir="runs/gtrxl_overfit",
        epochs=30,
        batch_size=8,
        chunk_len=64,
        horizon_H=10,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        dropout=0.0,      # disable dropout to allow overfitting
        lambda_occ=0.5,
    )

    train(cfg)


if __name__ == "__main__":
    folder = "../data"

    # Step 1: create npz wrapper
    npz_path = make_npz_from_emissions(folder)

    # Step 2: quick sanity run
    quick_sanity_run(npz_path)

    # Step 3: overfitting test
    overfit_test(npz_path)
