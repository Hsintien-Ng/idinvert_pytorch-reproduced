import os
import numpy as np
from torch.utils import data
from torchvision import transforms as trans
from PIL import Image


class CelebaHQ(data.Dataset):

    def __init__(self, dataset_args, train=True):
        self.name = 'CelebaHQ'
        self.data_root = dataset_args.data_root
        self.train_root = os.path.join(self.data_root, 'train')
        self.val_root = os.path.join(self.data_root, 'val')
        self.train_list = self.collect_image(self.train_root)
        self.val_list = self.collect_image(self.val_root)
        self.transform = trans.ToTensor()
        self.max_val = dataset_args.max_val
        self.min_val = dataset_args.min_val
        self.train = train
        self.size = dataset_args.size

    def collect_image(self, root):
        image_path_list = []
        for split in os.listdir(root):
            split_root = os.path.join(root, split)
            for file_name in sorted(os.listdir(split_root)):
                file_path = os.path.join(split_root, file_name)
                image_path_list.append(file_path)
        return image_path_list

    def read_image(self, path):
        img = Image.open(path)
        return img

    def resize_image(self, image, size=256):
        img = image.resize((size, size))
        return img

    def __getitem__(self, index):
        if self.train:
            image_path = self.train_list[index]
        else:
            image_path = self.val_list[index]
        img = self.read_image(image_path)
        img = self.resize_image(img, size=self.size)
        img = self.transform(img)
        img = img * (self.max_val - self.min_val) + self.min_val
        return img

    def __len__(self):
        if self.train:
            return len(self.train_list)
        else:
            return len(self.val_list)


if __name__ == '__main__':
    class Config:
        data_root = '/mnt/ssd2/xintian/datasets/celeba_hq/'
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        size = 256
    config = Config()
    dataset = CelebaHQ(config)
    for i, data in enumerate(dataset):
        print(data.shape, data.max(), data.min())
        break
    print(dataset.__len__())