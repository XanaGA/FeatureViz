import os
import torch
import h5py
import fire
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision import transforms as T

# Local imports
from featureviz.models.vjepa2 import VJEPA2Model
from featureviz.datasets.dl3dv import DL3DVDataset

def main(
    dataset_path: str,
    output_path: str,
    model_type: str = "vjepa2_vit_large",
    batch_size: int = 1,
    separation: int = 1,
    device: str = "cuda",
    limit_scenes: int = -1
):
    """
    Extracts features from a model and saves them to an HDF5 file.
    Supports checkpointing: if a scene/frame exists in the output file, it is skipped.
    """

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Model
    # Note: In a full library, you'd use a factory pattern here
    if "vjepa" in model_type:
        model = VJEPA2Model(model_name=model_type, device=device)
    else:
        raise ValueError(f"Model {model_type} not supported yet.")

    # 2. Initialize Dataset
    # V-JEPA expects clips. For now, we assume a default clip length of 16 for temporal models.
    transform = T.ToTensor()
    clip_len = 16 if model.is_temporal else 1
    dataset = DL3DVDataset(
        root_dir=dataset_path,
        separation=separation,
        clip_len=clip_len,
        transform= transform,
        processor= model.processor if hasattr(model, "processor") else None
    )

    print(f"Extracting {limit_scenes} out of {len(dataset)} samples...")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 3. Open HDF5 File (Append mode for checkpointing)
    with h5py.File(output_path, "a") as h5_file:
        # Store global metadata
        if "metadata" not in h5_file:
            meta = h5_file.create_group("metadata")
            meta.attrs["model_name"] = model_type
            meta.attrs["separation"] = separation
            meta.attrs["clip_len"] = clip_len

        print(f"Starting extraction for {len(dataset)} samples...")

        count = 0
        for batch in tqdm(loader, desc="Extracting Features"):
            images = batch["pixels"].to(device)
            metadata = batch["metadata"]
            
            # Checkpointing Logic: 
            # We use scene_id and frame_idx to create a unique path in the H5 file
            scene_id = metadata["scene_id"][0]
            frame_idx = metadata["frame_idx"][0].item()
            
            h5_path = f"{scene_id}/frame_{frame_idx:05d}"
            
            if h5_path in h5_file:
                continue # Skip if already processed

            # Inference
            with torch.no_grad():
                features = model.forward_features(images)
                # Move to CPU and convert to float16 to save significant disk space
                features = features.cpu().to(torch.float16).numpy()

            # Save to H5
            # Create group structure if it doesn't exist
            group = h5_file.require_group(scene_id)
            group.create_dataset(
                f"frame_{frame_idx:05d}", 
                data=features, 
                compression="gzip", 
                compression_opts=4
            )
            count += 1
            if limit_scenes > 0 and count >= limit_scenes:
                break

    print(f"Done! Features saved to {output_path}")

if __name__ == "__main__":
    fire.Fire(main)