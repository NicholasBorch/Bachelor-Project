# Dataset definitions for HAM10000 classification.
from pathlib import Path
from typing import Dict
import torch
from PIL import Image
from torch.utils.data import Dataset


class HamTensorDataset(Dataset):
    # Minimal HAM10000 dataset returning (image_tensor, label_index, image_id)
    # Images are loaded into memory once at init to avoid repeated disk I/O

    def __init__(self, df, images_dir: Path, c2i: Dict[str, int], tfm):
        self.df         = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.c2i        = c2i
        self.tfm        = tfm

        # Pre-load all images into memory once
        print(f"    Caching {len(self.df)} images...", flush=True)
        self.images = []
        for idx in range(len(self.df)):
            image_id = str(self.df.iloc[idx]["image_id"])
            img = Image.open(images_dir / f"{image_id}.jpg").convert("RGB")
            self.images.append(img)
        print(f"    Done caching.", flush=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row      = self.df.iloc[idx]
        image_id = str(row["image_id"])
        y        = int(self.c2i[str(row["dx"])])
        return self.tfm(self.images[idx]), y, image_id