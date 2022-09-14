'''
Sort images based on names.
'''

def sort_images(x):
    vid = int(x.split('_')[-1].split('/')[0])
    frame = int(x.split('/')[-1].split('-')[0])
    return vid*7200 + frame