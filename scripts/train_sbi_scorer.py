"""SBI Scorer training script.

Generates training data via BasiliskTwin (or ToySimulator fallback),
trains amortized NPE posteriors per fault family, and saves to
models/sbi/<family>/{posterior,calibration,ppc}.

Usage:
    python scripts/train_sbi_scorer.py --families friction,dropout,stiction
    python scripts/train_sbi_scorer.py --n-sims 10000 --families friction
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from domain import FaultParameter, SimMapping
from twin.simulator import ToySimulator
from evaluate.sbi_scorer import _extract_summary_stats

MODELS_DIR = ROOT / "models" / "sbi"


FAULT_FAMILIES = {
    "friction": {
        "param": "friction",
        "prior_low": 0.0,
        "prior_high": 2.0,
    },
    "dropout": {
        "param": "dropout_rate",
        "prior_low": 0.0,
        "prior_high": 0.1,
    },
    "stiction": {
        "param": "stiction_rate",
        "prior_low": 0.0,
        "prior_high": 0.05,
    },
}


def generate_training_data(
    family: str,
    n_sims: int = 1000,
    duration_s: int = 2000,
    twin_cls: type = ToySimulator,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (theta, summary_stats) training pairs for a fault family.

    Returns:
        thetas: (n_sims,) array of sampled parameters
        stats: (n_sims, n_features) array of summary statistics
    """
    config = FAULT_FAMILIES[family]
    rng = np.random.default_rng(seed)

    thetas = rng.uniform(config["prior_low"], config["prior_high"], size=n_sims)
    stats_list = []

    for i, theta in enumerate(thetas):
        mapping = SimMapping(
            subsystem="reaction_wheel",
            fault_params=(FaultParameter(config["param"], float(theta)),),
        )
        twin = twin_cls()
        twin.configure(mapping)
        sim = twin.run(duration_s=duration_s, seed=seed + i)
        summary = _extract_summary_stats(sim)
        stats_list.append(summary)

        if (i + 1) % 100 == 0:
            print(f"  [{family}] Generated {i + 1}/{n_sims} simulations")

    return thetas, np.array(stats_list)


def train_family(
    family: str,
    n_sims: int = 1000,
    seed: int = 42,
) -> Path:
    """Train an amortized posterior for a single fault family.

    When sbi is available, trains a neural posterior estimator.
    Otherwise, saves the training data for later use.
    """
    print(f"Training family: {family} ({n_sims} simulations)")
    family_dir = MODELS_DIR / family
    family_dir.mkdir(parents=True, exist_ok=True)

    # Generate training data
    thetas, stats = generate_training_data(family, n_sims=n_sims, seed=seed)

    # Save raw training data
    np.savez(family_dir / "training_data.npz", thetas=thetas, stats=stats)

    try:
        import torch
        import sbi.inference  # type: ignore[import-untyped]
        from sbi.inference import SNPE  # type: ignore[import-untyped]
        from sbi.utils import BoxUniform  # type: ignore[import-untyped]

        config = FAULT_FAMILIES[family]

        # Define prior
        prior = BoxUniform(
            low=torch.tensor([config["prior_low"]]),
            high=torch.tensor([config["prior_high"]]),
        )

        # Prepare tensors
        theta_tensor = torch.tensor(thetas, dtype=torch.float32).unsqueeze(-1)
        stats_tensor = torch.tensor(stats, dtype=torch.float32)

        # Train NPE
        inference = SNPE(prior=prior)
        inference.append_simulations(theta_tensor, stats_tensor)
        density_estimator = inference.train()
        posterior = inference.build_posterior(density_estimator)

        # Save posterior
        with open(family_dir / "posterior.pkl", "wb") as f:
            pickle.dump(posterior, f)

        print(f"  ✓ Trained NPE posterior for {family}")

    except ImportError:
        print(f"  ⚠ sbi not installed. Saved training data only for {family}")
        print(f"    Install with: pip install sbi torch")

    return family_dir


def main():
    parser = argparse.ArgumentParser(description="Train SBI scorer posteriors")
    parser.add_argument("--families", type=str, default="friction,dropout,stiction",
                        help="Comma-separated fault families to train")
    parser.add_argument("--n-sims", type=int, default=1000,
                        help="Number of simulations per family")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    families = [f.strip() for f in args.families.split(",")]

    for family in families:
        if family not in FAULT_FAMILIES:
            print(f"Unknown family: {family}. Available: {list(FAULT_FAMILIES.keys())}")
            continue
        train_family(family, n_sims=args.n_sims, seed=args.seed)

    print(f"\nTraining complete. Models saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
