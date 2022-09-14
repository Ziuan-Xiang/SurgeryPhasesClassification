'''
Get images and labels from selected videos.
'''

import os
from utils.sortimages import sort_images


def read(videos, image_base, ind_end:int, ind_start=0):
    '''
    Read image frames from videos.
    Input args:
        videos: Path of hernia surgical videos
        image_base: Path for saving image frames
        ind_end: Ending index of video sequences
        ind_start: Start index of video sequences
    Outputs:
        List of image paths and label paths
    '''
    image_paths = []
    labels = []
    # get 2 images and labels
    #dataset for test
    
    
    
    for video in sorted(videos[ind_start:ind_end]):
        #base = os.path.join(image_base, video.split('.')[0])
        base = image_base + '/' + video.split('.')[0]
        # image_paths.append(list(map(
        #     lambda img: os.path.join(base, img) if img.endswith('.png') else None, 
        #     os.listdir(base)
        # )))
        image_paths.append(list(map(
            lambda img: base + '/' + img if img.endswith('.png') else None, 
            os.listdir(base)
        )))

    image_paths = [list(filter(None, images)) for images in image_paths]
    image_paths = [list(sorted(images, key=sort_images)) for images in image_paths]

    labels = [
        [int(img.split('.')[0].split('-')[1]) for img in images]
        for images in image_paths
    ]

    return image_paths, labels