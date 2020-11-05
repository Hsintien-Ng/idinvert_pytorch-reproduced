import os
import numpy as np
from torch.utils import data
from torchvision import transforms as trans
from PIL import Image
from torch.utils.data import DataLoader
ignored_dir = ['wild','dog'] # 采集猫脸和狗脸作为动物数据集
class Afhq(data.Dataset):

    def __init__(self, dataset_args, train=True):
        self.name = 'afhq'
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

    def collect_image(self, root=None):
        if root is None:
            root = self.data_root

        image_path_list = []
        for split in os.listdir(root):
            if split in ignored_dir:
                continue
            print(split)
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

    def get_batch_index(self,batch_size):
        return np.random.choice(range(self.__len__()), batch_size)

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

class RandomSampler(object):
  """random sampler to yield a mini-batch of indices."""
  def __init__(self, batch_size, dataset, drop_last=False):
    self.dataset = dataset
    self.batch_size = batch_size
    self.num_imgs = len(dataset)
    self.drop_last = drop_last
 
  def getbatch(self):
    indices = np.random.permutation(self.num_imgs)
    batch = []
    for i in indices:
      batch.append(i)
      if len(batch) == self.batch_size:
        yield batch
        batch = []
    ## if images not to yield a batch
    if len(batch)>0 and not self.drop_last:
      yield batch
 
 
  def __len__(self):
    if self.drop_last:
      return self.num_imgs // self.batch_size
    else:
      return (self.num_imgs + self.batch_size - 1) // self.batch_size


if __name__ == '__main__':
    class Config:
        data_root = '/4T/huanzhang/afhq/'
        max_val = 255
        min_val = 0
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        size = 256
    config = Config()
    dataset = Afhq(config,train=True)
    randomsample = RandomSampler(8,dataset)
    print(randomsample.getbatch())
    print(dataset.collect_image)
    dataloader = DataLoader(dataset,batch_size = 8,shuffle = False)
    idx = 0
    for id,item in enumerate(dataloader):
        idx = idx + 1
    print(idx)
    # for i, data in enumerate(dataset):
    #     print(data.shape, data.max(), data.min())
    #     break
    print(dataset.__len__())