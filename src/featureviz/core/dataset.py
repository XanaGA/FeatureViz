import torch
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

class BaseDataset(Dataset, ABC):
    """
    Abstract Base Class for datasets.
    Handles the 'separation' and 'shuffle' logic for sequences.
    """
    def __init__(
        self, 
        root_dir: str, 
        separation: int = 1, 
        shuffle: bool = False,
        transform: Optional[Any] = None
    ):
        self.root_dir = root_dir
        self.separation = separation
        self.shuffle = shuffle
        self.transform = transform
        
        # This should be populated by the child class
        self.samples: List[Any] = self._load_samples()
        
        if self.shuffle:
            np.random.shuffle(self.samples)
            
        # Apply separation (stride)
        if self.separation > 1:
            self.samples = self.samples[::self.separation]

    @abstractmethod
    def _load_samples(self) -> List[Any]:
        """Logic to scan directories and return a list of file paths or indices."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        return len(self.samples)

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dictionary containing:
        - "image": torch.Tensor (transformed for the model)
        - "viz_image": np.ndarray (original image for plotting)
        - "metadata": dict (filename, frame index, etc.)
        """
        pass

    def get_dataloader(self, batch_size: int = 1, shuffle: bool = False, num_workers: int = 4) -> DataLoader:
        """Standardizes the creation of a PyTorch DataLoader."""
        return DataLoader(
            self, 
            batch_size=batch_size, 
            shuffle=shuffle, # Shuffling not intra sample
            num_workers=num_workers,
            pin_memory=True
        )