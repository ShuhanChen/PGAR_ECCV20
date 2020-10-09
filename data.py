import os
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, dep_root, trainsize):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.depths = [dep_root + f for f in os.listdir(dep_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor()])
        self.dep_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.493], [0.231])])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        dep = self.gray_loader(self.depths[index])
        image, gt, dep = self.cv_random_flip(image, gt, dep)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        dep = self.dep_transform(dep)
        return image, gt, dep

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def gray_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def cv_random_flip(self, img, gt, dep):
        if np.random.randint(2)==0:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
            dep = dep.transpose(Image.FLIP_LEFT_RIGHT)
        return img, gt, dep

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, dep_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):

    dataset = SalObjDataset(image_root, gt_root, dep_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, dep_root, testsize):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')
                    or f.endswith('.png') or f.endswith('.bmp')]
        self.depths = [dep_root + f for f in os.listdir(dep_root) if f.endswith('.png')
                    or f.endswith('.bmp')]
        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.img_transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.dep_transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.493], [0.231])])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        depth = self.gray_loader(self.depths[self.index])
        img_size = (image.size[1], image.size[0])
        image = self.img_transform(image).unsqueeze(0)
        depth = self.dep_transform(depth).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, depth, img_size, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def gray_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

