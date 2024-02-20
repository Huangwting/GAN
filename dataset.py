import torchvision.datasets as TD
from torch.utils.data import DataLoader
from torchvision import transforms as TF

image_size = 64
batch_size = 128
n_workers = 2

# 加载数据集
data_face = TD.ImageFolder('face', transform= TF.Compose(
    [
        TF.Resize(image_size),  # 64 * 64 * 3
        TF.CenterCrop(image_size),
        TF.ToTensor(),  # to [0,1]
        TF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
))

face_loader = DataLoader(data_face, batch_size=batch_size, shuffle=True, num_workers=n_workers)
invTrans = TF.Compose(
    [
        TF.Normalize(mean=[0., 0., 0.], std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
        TF.Normalize(mean=[-0.5, -0.5, -0.5], std=[1., 1., 1.])
    ]
)
