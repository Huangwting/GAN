from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import os
from PIL import Image


class DataSetFace(Dataset):
    def __init__(self, root, transforms=None):
        imgs = []
        for path in os.listdir(root):
            imgs.append(os.path.join(root, path))

        self.imgs = imgs
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            self.transforms = T.Compose([
                T.Resize(64),
                T.CenterCrop(64),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]

        data = Image.open(img_path)
        if data.mode != "RGB":
            data = data.convert("RGB")
        data = self.transforms(data)
        return data

    def __len__(self):
        return len(self.imgs)
