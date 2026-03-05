# Dataset definitions for HAM10000 classification.

from pathlib import Path
from typing import Dict

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class HamTensorDataset(Dataset):
    # Minimal HAM10000 dataset returning (image_tensor, label_index, image_id)
    def __init__(self, df, images_dir: Path, c2i: Dict[str, int], tfm):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.c2i = c2i
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        y = int(self.c2i[str(row["dx"])])
        img = Image.open(self.images_dir / f"{image_id}.jpg").convert("RGB")
        return self.tfm(img), y, image_id