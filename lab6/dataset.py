from pathlib import Path

from pydantic import BaseModel
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image, decode_image


class Image(BaseModel):
    path: Path
    label: int


class MyDataset(Dataset):
    def __init__(self, folder: str, transform=None):
        self.folder = Path(folder)
        self.transform = transform
        self.images = list()
        self.classes = list()
        for i, sub in enumerate(self.folder.iterdir()):
            self.classes.append(sub.name)
            for image in sub.iterdir():
                self.images.append(Image(
                    path=image,
                    label=i,
                ))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        label = image.label
        tensor = decode_image(image.path, ImageReadMode.RGB)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label

