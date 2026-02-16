import numpy as np
import h5py
from scipy import stats
from typing import Dict
from tqdm import tqdm

class FeatureStatistics:
    """
    Computes global statistics and distribution properties of latent features.
    Uses streaming algorithms to handle datasets larger than RAM.
    """
    def __init__(self, features_path: str):
        self.features_path = features_path
        
    def compute_global_stats(self, sample_ratio: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Computes Mean, Std, and Min/Max for each channel.
        
        Args:
            sample_ratio: Fraction of data to use for speed (default 10%).
            
        Returns:
            Dict containing 'mean', 'std', 'min', 'max' arrays of shape (C,).
        """
        print(f"--- Computing Global Statistics (Sampling {sample_ratio*100}%) ---")
        
        # Welford's online algorithm for mean and variance
        n = 0
        mean = None
        M2 = None
        min_val = None
        max_val = None

        with h5py.File(self.features_path, "r") as f:
            # Gather all frame keys
            all_keys = []
            for scene_id in f.keys():
                if scene_id == "metadata": continue
                for frame_key in f[scene_id].keys():
                    all_keys.append((scene_id, frame_key))
            
            # Shuffle and sample
            np.random.shuffle(all_keys)
            subset_size = int(len(all_keys) * sample_ratio)
            selected_keys = all_keys[:subset_size]

            for scene_id, frame_key in tqdm(selected_keys, desc="Scanning Features"):
                # Load feature: (1, T, L, C) or (1, L, C)
                feat = f[scene_id][frame_key][:]
                
                # Flatten to (N_tokens, C)
                # We combine spatial and temporal dimensions for global stats
                if feat.ndim == 4: # (B, T, L, C)
                    feat = feat.reshape(-1, feat.shape[-1])
                elif feat.ndim == 3: # (B, L, C)
                    feat = feat.reshape(-1, feat.shape[-1])
                
                # Initialize accumulators on first batch
                if mean is None:
                    C = feat.shape[-1]
                    mean = np.zeros(C, dtype=np.float64)
                    M2 = np.zeros(C, dtype=np.float64)
                    min_val = np.full(C, np.inf)
                    max_val = np.full(C, -np.inf)

                # Update Min/Max
                batch_min = feat.min(axis=0)
                batch_max = feat.max(axis=0)
                min_val = np.minimum(min_val, batch_min)
                max_val = np.maximum(max_val, batch_max)

                # Update Mean and Variance (Welford's)
                # This is numerically stable
                for x in feat:
                    n += 1
                    delta = x - mean
                    mean += delta / n
                    delta2 = x - mean
                    M2 += delta * delta2

        variance = M2 / (n - 1)
        std_dev = np.sqrt(variance)
        
        return {
            "mean": mean,
            "std": std_dev,
            "min": min_val,
            "max": max_val,
            "dead_channels": np.where(std_dev < 1e-6)[0]
        }

    def compute_isotropy_score(self, num_samples: int = 5000) -> float:
        """
        Measures how "Isotropic" the feature space is.
        Isotropy = 1.0 means the covariance matrix is Identity (perfectly spherical).
        
        Metric: Ratio of Min Eigenvalue to Max Eigenvalue of Covariance Matrix.
        """
        print("--- Computing Isotropy Score ---")
        # Load a random subset of features
        data = self._load_random_subset(num_samples)
        
        # Center the data
        data = data - np.mean(data, axis=0)
        
        # Compute Covariance Matrix
        cov = np.cov(data, rowvar=False)
        
        # Compute Eigenvalues
        eigvals = np.linalg.eigvalsh(cov)
        
        # Isotropy ratio
        min_eig = eigvals.min()
        max_eig = eigvals.max()
        
        # Avoid division by zero
        if max_eig == 0: return 0.0
            
        isotropy = min_eig / max_eig
        print(f"Isotropy Score (Min/Max Eigenvalue): {isotropy:.4f}")
        return isotropy

    def test_normality(self, num_samples: int = 1000) -> Dict[str, float]:
        """
        Tests if the features follow a Gaussian distribution.
        Uses Kolmogorov-Smirnov test on the projection of features.
        
        Returns:
            p-value of the test (Higher = More likely Gaussian).
        """
        print("--- Testing Normality (Kolmogorov-Smirnov) ---")
        data = self._load_random_subset(num_samples)
        
        # We test normality on random projections to avoiding testing every channel
        # If X is Gaussian, any linear projection v^T X is 1D Gaussian.
        
        # Project data onto a random vector
        C = data.shape[1]
        random_direction = np.random.randn(C)
        random_direction /= np.linalg.norm(random_direction)
        
        projected = data @ random_direction
        
        # Normalize to standard normal (0, 1) for comparison
        projected = (projected - projected.mean()) / projected.std()
        
        # KS Test against standard normal
        stat, p_value = stats.kstest(projected, 'norm')
        
        result = {
            "ks_statistic": stat,
            "p_value": p_value,
            "is_gaussian": bool(p_value > 0.05) # Standard alpha
        }
        print(f"Normality Test: p-value={p_value:.4f} ({'Gaussian' if result['is_gaussian'] else 'Non-Gaussian'})")
        return result

    def _load_random_subset(self, num_samples: int) -> np.ndarray:
        """Helper to load exactly N random feature vectors."""
        collected = []
        with h5py.File(self.features_path, "r") as f:
            all_scenes = list(f.keys())
            if "metadata" in all_scenes: all_scenes.remove("metadata")
            
            while len(collected) < num_samples:
                scene = np.random.choice(all_scenes)
                frames = list(f[scene].keys())
                frame = np.random.choice(frames)
                
                feat = f[scene][frame][:] # (1, T, L, C)
                feat = feat.reshape(-1, feat.shape[-1]) # (N, C)
                
                # Take a random patch from this frame
                idx = np.random.randint(0, len(feat))
                collected.append(feat[idx])
                
                if len(collected) >= num_samples:
                    break
                    
        return np.array(collected)