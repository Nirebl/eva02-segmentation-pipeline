import pathlib
from glob import glob

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


# -----------------------
# Dataset
# -----------------------
class SegDataset(Dataset):
    def __init__(self, root, split="train", image_size=448):
        self.img_dir = pathlib.Path(root) / split / "images"
        self.mask_dir = pathlib.Path(root) / split / "masks"
        self.img_paths = sorted(glob(str(self.img_dir / "*.jpg")))
        assert len(self.img_paths) > 0, f"No images in {self.img_dir}"

        # аугменты: для train — посильнее, для val — только resize/normalize
        if split == "train":
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=image_size),
                A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    scale=(0.9, 1.1),
                    rotate=(-15, 15),
                    border_mode=cv2.BORDER_CONSTANT,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    fill=0,
                    fill_mask=0,
                    p=0.5
                ),

                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=image_size),
                A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

        self.split = split
        self.image_size = image_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        ip = pathlib.Path(self.img_paths[idx])
        mp = self.mask_dir / (ip.stem + ".png")

        img = cv2.imread(str(ip))[:, :, ::-1]  # BGR->RGB
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)

        # Маски приводим к {0,1}
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)

        aug = self.tf(image=img, mask=mask)
        x = aug["image"]  # (3,H,W), float32
        y = aug["mask"][None, ...].float()  # (1,H,W), float32 in {0,1}
        return x, y

