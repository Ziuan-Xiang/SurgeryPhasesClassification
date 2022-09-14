'''
Hernia Dataset and processing utils.
'''

from torch.utils.data import Dataset, DataLoader

import os
import glob
from PIL import Image
from bisect import bisect
import random

import pandas as pd


class PhaseMapper:
    ''' Phase Mapper '''

    def __init__(self, all_labels_merged):
        '''
        Args
        ----
            all_labels_merged: a csv with columns [id,labels,merge]
                id -> int: id of the labels
                labels -> str: names of labels
                merge -> str: to which label name the current one is merged to, can be:
                    - NaN: empty
                    - a str contained in labels: this label will be statically merged as indicated
                    - "PREV": this label will be dynamically merged to whichever comes before
        '''

        all_labels_merged = pd.read_csv(all_labels_merged)

        # check column values
        idx = all_labels_merged['merge'].dropna().isin(
            all_labels_merged['labels'].values.tolist() + ['PREV']
        )
        if not idx.all():
            fail_idx = idx[idx == False].index.values
            raise ValueError('Invalid values to merge to in line {}: {}'.format(
                fail_idx, all_labels_merged.loc[fail_idx, 'merge'].values
            ))

        all_labels_merged.drop(all_labels_merged[all_labels_merged['merge'] == 'PREV'].index.values, inplace=True)
        self.names = all_labels_merged[pd.isna(all_labels_merged['merge'])][['labels']].reset_index(drop=True)

        # static maps
        all_labels_merged['merge'].fillna(all_labels_merged['labels'], inplace=True)
        all_labels_merged['to'] = all_labels_merged['merge'].apply(
            lambda x: self.names[self.names['labels'] == x].index[0]
        )
        self.map_static = dict(all_labels_merged[['id', 'to']].values.astype(int).tolist())

        # previous label cache: default is transitionary idle
        self.prev_label = self.names[self.names['labels'] == 'transitionary idle'].index[0]


    def get_merged_labels(self):
        ''' Get a Dataframe of merged labels and index '''
        return self.names


    def __call__(self, label):
        # try getting the mapped value from static map
        # if failed, return prev_label
        self.prev_label = self.map_static.get(label, self.prev_label)
        return self.prev_label


class HerniaDataset(Dataset):
    '''
    Hernia Dataset

    Usage
    -----
        This is a map style dataset, whose data can be retrieved by:
        - dataset[index]
            index is an integer that lies between [0, num_frames_total)
        - dataset[vindex, findex]
            vindex is an integer that lies between [0, num_videos_total)
            findex is an integer that lies between [0, num_frames_each_video)

    Return
    ------
        a dict: {
            'image': img -> tensor, 
            'label': label -> int, 
            'findex': findex -> int: indicating the position of current frame in the video, 
            ...
        }
    '''

    def __init__(self, root, videos, transforms, class_map=None):
        '''
        Args
        ----
            root -> str: root of the dataset folder
            videos -> list: names of the videos (folders)
            transforms: data transforms (from PIL.Image to Tensor)
        '''

        super().__init__()
        self.paths = [os.path.join(root, video) for video in videos]
        self.labels = []
        self.images = []
        for path in self.paths:
            with open(os.path.join(path, 'labels.txt'), 'r') as f:
                self.labels.append(list(map(int, f.readlines())))
            self.images.append(sorted(glob.glob(os.path.join(path, '*.jpg'))))

        # compatibility check
        image_lens = [len(image) for image in self.images]
        label_lens = [len(label) for label in self.labels]
        assert image_lens == label_lens, (
            'Incompatible image lengths and label lengths'
             '\nimages: \n{} \nlabels: \n{}'.format(image_lens, label_lens)
        )

        self.transforms = transforms
        # the accumualted number of frames (labels) through all videos
        self.num_frames = label_lens
        self.update_cnts()

        self.class_map = class_map or (lambda x: x)


    def update_cnts(self):
        ''' Update frame counts after shuffle '''
        self.cnts = [self.num_frames[0]]
        for l in self.num_frames[1:]:
            self.cnts.append(self.cnts[-1] + l)


    def __len__(self):
        return self.cnts[-1]


    def __getitem__(self, index):
        if isinstance(index, int):
            vindex = bisect(self.cnts, index)
            findex = index if vindex == 0 else index - self.cnts[vindex - 1]

        elif isinstance(index, tuple) and len(index) == 2:
            vindex, findex = index

        else:
            raise ValueError('index must be either integer or a tuple of two integers')

        img = Image.open(self.images[vindex][findex])
        label = self.labels[vindex][findex]

        return {
            'feature': self.transforms(img), 
            'label': self.class_map(label), 
            'label_raw': label, 
            'findex': findex
        }


class RandomLoader(DataLoader):
    ''' Random-sampling loader (all frames are shuffled) '''

    def __init__(self, dataset, batch_size, num_workers=0, prefetch_factor=2, pin_memory=True):
        super().__init__(
            dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, prefetch_factor=2, pin_memory=pin_memory
        )


class SequentialLoader(DataLoader):
    ''' Sequential loader (all frames are sequential) '''

    def __init__(self, dataset, batch_size, num_workers=0, prefetch_factor=2, pin_memory=True):
        super().__init__(
            dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, prefetch_factor=2, pin_memory=pin_memory
        )


# unit test
if __name__ == '__main__':
    videos = ['RALIHR_surgeon01_fps01_{:04}'.format(i + 1) for i in range(5)]

    from torchvision import transforms
    trans = transforms.Compose([
        transforms.Resize((32, 32)), 
        transforms.ToTensor()
    ])

    dataset = HerniaDataset(
        'surgery_hernia_train_test/', # careful with the current directory
        videos, 
        transforms=trans
    )

    data = dataset[0]
    assert data['feature'].shape == (3, 32, 32), data['findex'] == 0
    print('HerniaDataset passed')

    # random-sampling loader
    loader = RandomLoader(dataset, 2)
    data = next(iter(loader))
    assert data['feature'].shape == (2, 3, 32, 32)
    print('Random-sampling loader passed')

    # mapper
    mapper = PhaseMapper('resnet/configs/all_labels_hernia_merged_7.csv')
    labels, target = range(14), [4, 4, 0, 0, 0, 1, 2, 3, 0, 4, 5, 5, 6, 6]
    result = list(map(mapper, labels))
    assert result == target
    print('PhaseMapper passed')
