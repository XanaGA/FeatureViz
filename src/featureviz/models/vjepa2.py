import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional
from featureviz.core.model import BaseModel

class VJEPA2Model(BaseModel):
    """
    Implementation of the V-JEPA2 model for feature extraction.
    Supports Large, Huge, and Giant ViT backbones.
    """
    
    # Model dimensions for V-JEPA2 variants
    ARCH_CONFIG = {
        "vjepa2_vit_large": 1024,
        "vjepa2_vit_huge": 1280,
        "vjepa2_vit_giant": 1536,
        "vjepa2_vit_giant_384": 1536,
    }

    def __init__(self, model_name: str = "vjepa2_vit_large", device: str = "cuda"):
        super().__init__(model_name, device)
        self._feature_blocks = {}
        self._hooks = []

    @property
    def is_temporal(self) -> bool:
        return True

    @property
    def num_channels(self) -> int:
        return self.ARCH_CONFIG.get(self.model_name, 1024)

    def _load_model(self) -> nn.Module:
        """Loads the model and preprocessor from TorchHub."""
        print(f"--- Loading {self.model_name} from facebookresearch/vjepa2 ---")
        model, predictor = torch.hub.load('facebookresearch/vjepa2', self.model_name)
        self.processor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_preprocessor')
        return model

    def _get_hook(self, name):
        def hook(model, input, output):
            # V-JEPA outputs are often (B, T, L, C) or (B, L, C)
            self._feature_blocks[name] = output.detach()
        return hook

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x shape: (B, T, 3, H, W)
        Returns: (B, T_out, L, C) where L is spatial tokens
        """
        # V-JEPA expects (B, T, 3, H, W) or specific tubelet dimensions
        # We permute to match the model's expected 3D conv input
        if x.ndim == 5:
            x = x.permute(0, 2, 1, 3, 4) # B, C, T, H, W
            
        with torch.no_grad():
            # Most V-JEPA Hub models have a .trunk or .forward method 
            # that returns the latent tokens
            # features = self.model.get_vision_features(x)
            features = self.model(x)
        return features

    def forward_intermediate(self, x: torch.Tensor, layers: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """
        Extracts features from specific ViT blocks.
        If layers is None, returns a subset of blocks (e.g., every 4th layer).
        """
        self._feature_blocks = {}
        self._hooks = []
        
        # Identify the blocks in the ViT trunk
        # V-JEPA typically uses model.blocks
        blocks = self.model.blocks
        
        if layers is None:
            # Default to extracting 4 roughly equidistant layers
            total = len(blocks)
            layers = [total // 4, total // 2, (3 * total) // 4, total - 1]

        # Register hooks
        for idx in layers:
            name = f"block_{idx}"
            self._hooks.append(blocks[idx].register_forward_hook(self._get_hook(name)))

        try:
            _ = self.forward_features(x)
        finally:
            # Always remove hooks to prevent memory leaks
            for h in self._hooks:
                h.remove()

        return self._feature_blocks