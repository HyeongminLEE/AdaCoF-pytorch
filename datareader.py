import numpy as np
from os import listdir
from PIL import Image
from os.path import join, isdir
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random


def cointoss(p):
    return random.random() < p


class DBreader_Vimeo90k(Dataset):
    def __init__(self, db_dir, random_crop=None, resize=None, augment_s=True, augment_t=True):
        db_dir += '/sequences'
        self.random_crop = random_crop
        self.augment_s = augment_s
        self.augment_t = augment_t

        transform_list = []
        if resize is not None:
            transform_list += [transforms.Resize(resize)]

        transform_list += [transforms.ToTensor()]

        self.transform = transforms.Compose(transform_list)

        self.folder_list = [(db_dir + '/' + f) for f in listdir(db_dir) if isdir(join(db_dir, f))]
        self.triplet_list = []
        for folder in self.folder_list:
            self.triplet_list += [(folder + '/' + f) for f in listdir(folder) if isdir(join(folder, f))]

        self.triplet_list = np.array(self.triplet_list)
        self.file_len = len(self.triplet_list)

    def __getitem__(self, index):
        rawFrame0 = Image.open(self.triplet_list[index] + "/im1.png")
        rawFrame1 = Image.open(self.triplet_list[index] + "/im2.png")
        rawFrame2 = Image.open(self.triplet_list[index] + "/im3.png")

        if self.random_crop is not None:
            i, j, h, w = transforms.RandomCrop.get_params(rawFrame1, output_size=self.random_crop)
            rawFrame0 = TF.crop(rawFrame0, i, j, h, w)
            rawFrame1 = TF.crop(rawFrame1, i, j, h, w)
            rawFrame2 = TF.crop(rawFrame2, i, j, h, w)

        if self.augment_s:
            if cointoss(0.5):
                rawFrame0 = TF.hflip(rawFrame0)
                rawFrame1 = TF.hflip(rawFrame1)
                rawFrame2 = TF.hflip(rawFrame2)
            if cointoss(0.5):
                rawFrame0 = TF.vflip(rawFrame0)
                rawFrame1 = TF.vflip(rawFrame1)
                rawFrame2 = TF.vflip(rawFrame2)

        frame0 = self.transform(rawFrame0)
        frame1 = self.transform(rawFrame1)
        frame2 = self.transform(rawFrame2)

        if self.augment_t:
            if cointoss(0.5):
                return frame2, frame1, frame0
            else:
                return frame0, frame1, frame2
        else:
            return frame0, frame1, frame2

    def __len__(self):
        return self.file_len
