# -*- coding: utf-8 -*-
'''
Assign each image in data/images with corresponding labels. 
'''

import os
from datetime import datetime


def prepare_images(videos, image_base, labels_df, names_df, image_ext='png'):
    '''
    Prepare images for network feeding.
    Input args:
        videos, image_base: Path of videos and images
        labels_df, names_df: All labels for each frame, 14 label names
        image_ext: Suffix of images
    '''
    # convert time string to int
    # start time
    t0 = datetime(1900, 1, 1)
    def time2int(t):
        # if convertable
        return (
            # time given in MM:SS
            datetime.strptime(t, '%M:%S') - t0
        ).seconds if len(t.split(':')) == 2 else (
            # time given in HH:MM:SS
            datetime.strptime(t, '%H:%M:%S') - t0
        ).seconds

    # extract names
    for video in sorted(videos):
        images = [
            dir for dir in os.listdir(os.path.join(image_base, video.split('.')[0]))
            if dir.endswith(image_ext)
        ]
        # get df corresponding to current video
        video_df = labels_df.loc[labels_df['videoName'] == video.split('.')[0]]
        # add two columns
        video_df[['StartSec', 'EndSec']] = video_df[['Start', 'End']].applymap(time2int)
        # 这warning好烦但我懒得改了
        for image in images:
            # check paths
            # base = os.path.join(image_base, video.split('.')[0])
            base = image_base +'/' +video.split('.')[0]
            path = base + '/' + image
            # path = os.path.join(base, image)
            # if '-' in image:
            #     continue
            t = int(image.split('.')[0].split('-')[0])
            # select interval and remove tailing digits
            try:
                name = video_df[
                    (video_df['StartSec'] <= t) & (t <= video_df['EndSec'])
                ]['PhaseName'].iloc[0]
                # find correct integer labels
                label = names_df[names_df['labels'] == name].index[0]
            except:
                label = 14
            # incorperate label into filenames
            if '-' not in path:
                # avoid renaming twice
                #new_path = ''.join(path.split('.')[:-1]) + '-{}.'.format(label) + path.split('.')[-1]
                new_path = '{}-{}.{}'.format(''.join(path.split('.')[:-1]), label, path.split('.')[-1])
            else:
                #new_path = path.split('-')[0] + '-' + str(label) + '.' + path.split('-')[1].split('.')[1]
                new_path = '{}-{}.{}'.format(path.split('-')[0], label, path.split('-')[1].split('.')[1])
            # rename all files
            os.rename(path, new_path)
