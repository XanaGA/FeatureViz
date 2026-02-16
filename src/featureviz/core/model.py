import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional, List, Dict

class BaseModel(nn.Module, ABC):
    """
    Standardized interface for SOTA Vision Models.
    Supports both static images and temporal video sequences, 
    including intermediate layer extraction.
    """
    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__()
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

    @property
    @abstractmethod
    def is_temporal(self) -> bool:
        """Returns True if the model expects video-like input (B, T, 3, H, W)."""
        pass

    @abstractmethod
    def _load_model(self) -> nn.Module:
        """Logic to load the model (TorchHub, HF, etc.)."""
        pass

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the final latent representation.
        
        Args:
            x: Tensor of shape (B, 3, H, W) OR (B, T, 3, H, W)
            
        Returns:
            Tensor: 
                - Image: (B, C, H, W) or (B, L, C)
                - Video: (B, T_out, C, H, W) or (B, T_out, L, C)
        """
        pass

    @abstractmethod
    def forward_intermediate(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from multiple intermediate layers.
        
        Returns:
            Dict mapping layer names/indices to their feature tensors.
        """
        pass

    def get_feature_metadata(self, input_res: Tuple[int, int] = (224, 224), t_dim: int = 8) -> Dict:
        """
        Probes the model to determine output spatial and temporal resolution.
        """
        with torch.no_grad():
            # Create dummy input based on model type
            shape = (1, t_dim, 3, *input_res) if self.is_temporal else (1, 3, *input_res)
            dummy_input = torch.zeros(shape).to(self.device)
            feat = self.forward_features(dummy_input)
            
            metadata = {
                "shape": feat.shape,
                "channels": feat.shape[2] if feat.ndim == 5 else feat.shape[-1], # Simplified check
                "is_temporal": self.is_temporal
            }
            
            # Logic for spatial resolution (handling L=H*W or L=H*W+1)
            # This can be expanded to return (T_out, H_out, W_out)
            return metadata

    @property
    def num_channels(self) -> int:
        """Returns the bottleneck feature dimension."""
        raise NotImplementedError