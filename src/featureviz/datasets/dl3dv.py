import os
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional
from featureviz.core.dataset import BaseDataset

class DL3DVDataset(BaseDataset):
    """
    Dataset wrapper for DL3DV-10K.
    Structure: root_dir / scene_id / images_4 / frame_XXXXX.png
    """
    def __init__(
        self, 
        root_dir: str, 
        separation: int = 1, 
        shuffle: bool = False, 
        transform: Optional[Any] = None,
        processor: Optional[Any] = None,
        clip_len: int = 1  # Set > 1 for temporal models like V-JEPA
    ):
        self.clip_len = clip_len
        # The parent constructor calls self._load_samples() and handles separation/shuffling
        super().__init__(root_dir, separation, shuffle, transform, processor)

    def _load_samples(self) -> List[Dict[str, Any]]:
        """
        Scans the DL3DV directory structure.
        Returns a list of dicts containing frame paths and metadata.
        """
        samples = []
        root_path = Path(self.root_dir)
        
        # DL3DV-10K/1K/ contains scene hash folders
        scene_dirs = [d for d in root_path.iterdir() if d.is_dir()]
        
        for scene_dir in scene_dirs:
            # We look for images in 'images_4' (downsampled) or 'images'
            img_dir = scene_dir / "images_4"
            if not img_dir.exists():
                img_dir = scene_dir / "images"
            
            if not img_dir.exists():
                continue

            # Load transforms.json if available for camera metadata
            transform_path = scene_dir / "transforms.json"
            scene_metadata = {}
            if transform_path.exists():
                with open(transform_path, 'r') as f:
                    scene_metadata = json.load(f)

            # Get all frames, sorted numerically
            frames = sorted(list(img_dir.glob("frame_*.png")))
            
            for i in range(len(frames) - self.clip_len + 1):
                samples.append({
                    "frame_paths": frames[i : i + self.clip_len],
                    "scene_id": scene_dir.name,
                    "scene_metadata": scene_metadata,
                    "frame_idx": i
                })
        
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        frame_paths = sample["frame_paths"]
        
        frames_np = []
        frames_tensor = []

        for p in frame_paths:
            # Load as RGB
            img_pil = Image.open(p).convert("RGB")
            
            # Apply transforms (standardizing to Tensor)
            if self.transform:
                img_tensor = self.transform(img_pil)
            else:
                # Fallback to basic tensor conversion if no transform provided
                img_np = np.array(img_pil)
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            
            frames_tensor.append(img_tensor)

        # Handle temporal vs image output shapes
        if self.clip_len == 1:
            final_tensor = frames_tensor[0]
        else:
            final_tensor = torch.stack(frames_tensor) # (T, 3, H, W)

        if self.processor:
            final_tensor = self.processor(final_tensor)[0] # (3, T, H, W)
            final_tensor = final_tensor.permute(1, 0, 2, 3) # (3, T, H, W) -> (T, 3, H, W)

        return {
            "pixels": final_tensor,
            "metadata": {
                "scene_id": sample["scene_id"],
                "frame_idx": sample["frame_idx"],
                "paths": [str(p) for p in frame_paths]
            }
        }