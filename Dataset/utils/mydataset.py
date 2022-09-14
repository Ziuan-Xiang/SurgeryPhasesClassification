'''
Build data structure for network feeding.
'''

from torch.utils.data import Dataset
from PIL import Image

class SVRCDataset(Dataset):
    '''
    Input data structure for SVRCNet,
    '''
    def __init__(self, image_path: list, image_class: list=None, transform=None):
        self.image_path = image_path
        self.image_class = image_class
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item): #can add more rules to pick data
        img = Image.open(self.image_path[item])
        if self.image_class is not None:
            label = self.image_class[item]
        if self.transform is not None:
            img = self.transform(img)

        return {'feature': img, 'label': label} if self.image_class is not None else {'feature': img}