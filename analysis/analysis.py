import numpy as np

import fire
import json
from pathlib import Path

from featureviz.analysis.simple_statistics import FeatureStatistics



def main(
    features_path: str,
    output_dir: str = "outputs/simple_statistics",
    sample_ratio: float = 0.1,
    num_samples_iso: int = 5000
):
    """
    CLI to run full statistical analysis on extracted features.
    
    Args:
        features_path: Path to the .h5 file containing extracted features.
        output_dir: Where to save the resulting JSON stats.
        sample_ratio: Fraction of the dataset to use for mean/std calculation.
        num_samples_iso: Number of vectors to sample for Isotropy and Normality tests.
    """
    features_path = Path(features_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Could not find features at {features_path}")

    stats_engine = FeatureStatistics(str(features_path))
    
    # 1. Run Analysis
    results = {}
    
    # Global Mean/Std/Dead Channels
    global_stats = stats_engine.compute_global_stats(sample_ratio=sample_ratio)
    results["dead_channels_count"] = len(global_stats["dead_channels"])
    results["dead_channels_indices"] = global_stats["dead_channels"].tolist()
    results["channel_mean_avg"] = float(np.mean(global_stats["mean"]))
    results["channel_std_avg"] = float(np.mean(global_stats["std"]))

    # Isotropy
    results["isotropy_score"] = float(stats_engine.compute_isotropy_score(num_samples=num_samples_iso))

    # Normality
    norm_test = stats_engine.test_normality(num_samples=num_samples_iso)
    results["normality"] = norm_test

    # 2. Save Results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    json_name = f"{features_path.stem}_stats.json"
    with open(out_path / json_name, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n--- Summary for {features_path.name} ---")
    print(f"Dead Channels: {results['dead_channels_count']}")
    print(f"Isotropy:      {results['isotropy_score']:.4f}")
    print(f"Gaussian:      {results['normality']['is_gaussian']}")
    print(f"Results saved to: {out_path / json_name}")

if __name__ == "__main__":
    fire.Fire(main)