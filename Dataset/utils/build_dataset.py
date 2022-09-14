'''
Build datasets for SVRC from images. 
All data are sorted. 
Use samplers in dataloader. 
'''


from PIL import Image
from torch.utils.data import (
    Dataset, DataLoader, random_split, SequentialSampler, RandomSampler, BatchSampler
)
import pandas as pd


class SVRCDataset_Old(Dataset):
    def __init__(self, image_path: list, image_class: list, transform = None):
        self.image_path = image_path
        self.image_class = image_class
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item): #can add more rules to pick data
        img = Image.open(self.image_path[item])
        label = self.image_class[item]
        if self.transform is not None:
            img = self.transform(img)

        return img, label


class CheckDataset(Dataset):
    def __init__(self, images: list=None, labels: list=None, transform = None):
        self.inited = False
        self.image_path = images
        self.image_class = labels
        self.transform = transform

    def init(self, image_base):
        # sth
        self.inited = True

    def check(func):
        ''' A decorator to check dataset status '''
        def check_inited(self, *args, **kwargs):
            assert self.inited, 'Dataset not initialized'
            print(args, kwargs)
            return func(self, *args, **kwargs)
        return check_inited

    @check
    def __len__(self):
        return len(self.image_path)

    @check
    def __getitem__(self, item): #can add more rules to pick data
        img = Image.open(self.image_path[item])
        label = self.image_class[item]
        if self.transform is not None:
            img = self.transform(img)

        return img, label

