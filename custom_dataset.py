import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from typing import Tuple


train_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def inverse_transform_fn(x):
    x = x.cpu()
    # apply sigmoid function to x
    x = 1 / (1 + torch.exp(-x))
    min, max = x.min(), x.max()
    y = (x - min) / (max - min)
    if y.min() < 0 or y.max() > 1:
        raise ValueError('Invalid range after the inverse transform')
    y = y.permute(1, 2, 0).numpy()
    return y
inverse_transform = transforms.Lambda(lambda x: inverse_transform_fn(x))


class CustomSubset(Dataset):
    def __init__(self, dataframe, train:bool = False):
        super(CustomSubset, self).__init__()
        self.dataframe = dataframe
        self.train = train

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noisy_path = self.dataframe.iloc[idx, 0]
        gt_path = self.dataframe.iloc[idx, 1]

        noisy_image = Image.open(noisy_path).convert('RGB')
        gt_image = Image.open(gt_path).convert('RGB')

        if self.train:
            noisy_image = train_transform(noisy_image)
            gt_image_transformed = train_transform(gt_image)
            gt_image = val_transform(gt_image)
            return noisy_image, gt_image_transformed, gt_image
        else:
            noisy_image = val_transform(noisy_image)
            gt_image = val_transform(gt_image)
            return noisy_image, gt_image


class CustomDataset():
    def __init__(self, dataframe, split_ratios=(0.8, 0.1, 0.1), random_state=42):
        if isinstance(dataframe, str):
            dataframe = pd.read_csv(dataframe)
        train_df = dataframe.sample(frac=split_ratios[0], random_state=random_state)
        val_test_df = dataframe.drop(train_df.index)
        val_df = val_test_df.sample(frac=0.5)
        test_df = val_test_df.drop(val_df.index)
        self.train = CustomSubset(train_df, train=True)
        self.val = CustomSubset(val_df)
        self.test = CustomSubset(test_df)

    def get_loaders(self, batch_size=32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_loader = DataLoader(self.train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(self.test, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # visualize the dataset
    dataset = CustomDataset('dataset.csv')
    for i in range(30):
        _, _, _, noisy_path, gt_path = dataset.train[i]
        noisy = plt.imread(noisy_path)
        gt = plt.imread(gt_path)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'Pair {i + 1}')
        axes[0].imshow(noisy)
        axes[0].set_title('Noisy Image\n' + noisy_path)
        axes[0].axis('off')
        axes[1].imshow(gt)
        axes[1].set_title('Clean Image\n' + gt_path)
        axes[1].axis('off')
        plt.show()


