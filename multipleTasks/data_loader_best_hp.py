import os
import random

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

INPUT_RESOLUTION = 224

"""Unused as it is not tensor"""
def tensor_rot_90(x):
    return x.flip(2).transpose(1,2)
def tensor_rot_90_digit(x):
	return x.transpose(1,2)

def tensor_rot_180(x):
	return x.flip(2).flip(1)
def tensor_rot_180_digit(x):
	return x.flip(2)

def tensor_rot_270(x):
	return x.transpose(1,2).flip(2)

"""End"""


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def load_image(path,do_flip,flip):
    img = Image.open(path)
    #img = img.flip(1)
    #img = np.array(list(reversed(img)))
    if do_flip:
        if flip:
            img = img.transpose(method=Image.FLIP_TOP_BOTTOM)
    return img.convert('RGB')


def make_sync_dataset(root, label, ds_name='synROD'):
    images = []

    with open(label, 'r') as labeltxt:
        for line in labeltxt:
            data = line.strip().split(' ')
            if not is_image_file(data[0]):
                continue
            if ds_name == 'ROD':
                path = os.path.join(root, '???-washington', data[0])
            else:
                path = os.path.join(root, data[0])

            if ds_name == 'synROD':
                #print("a")
                path_rgb = path.replace('***', 'rgb')
                path_depth = path.replace('***', 'depth')
            elif ds_name == 'ROD':
                #print("b")
                path_rgb = path.replace('***', 'crop')
                path_rgb = path_rgb.replace('???', 'rgb')
                path_depth = path.replace('***', 'depthcrop')
                path_depth = path_depth.replace('???', 'surfnorm')
            else:
                raise ValueError('Unknown dataset {}. Known datasets are synROD, ROD'.format(ds_name))
            gt = int(data[1])
            item = (path_rgb, path_depth, gt)
            images.append(item)
        return images


def get_relative_rotation(rgb_rot, depth_rot):
    rel_rot = rgb_rot - depth_rot
    if rel_rot < 0:
        rel_rot += 4
    assert rel_rot in range(4)
    return rel_rot





class MyTransform(object):

    def __init__(self, crop, flip):
        super(MyTransform, self).__init__()
        self.crop = crop
        self.flip = flip
        self.angles = [0, 90, 180, 270]

    def __call__(self, img, rot=None):
        img = TF.resize(img, [256, 256])
        img = TF.crop(img, self.crop[0], self.crop[1], INPUT_RESOLUTION, INPUT_RESOLUTION)
        """
        hflip invariant for natural scenes
        """
        if self.flip:
            img = TF.hflip(img)
        if rot is not None:
            img = TF.rotate(img, self.angles[rot])
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return img


class DatasetGeneratorMultimodal(Dataset):
    def __init__(self, root, label, ds_name='synROD', do_rot=False, do_flip=False, transform=None):
        imgs = make_sync_dataset(root, label, ds_name=ds_name)
        self.root = root
        self.label = label
        self.imgs = imgs
        self.transform = transform
        self.do_rot = do_rot
        """
        deciding to do a vertical flip
        """
        self.do_flip = do_flip

    def __getitem__(self, index):
        path_rgb, path_depth, target = self.imgs[index]
        """
        implementing labels for flipping
        """
        flip_rgb = bool(random.getrandbits(1))
        flip_depth = bool(random.getrandbits(1))

        img_rgb = load_image(path_rgb,self.do_flip,flip_rgb)
        img_depth = load_image(path_depth,self.do_flip,flip_depth)
        trans_rgb = None
        trans_depth = None

        # If a custom transform is specified apply that transform
        if self.transform is not None:
            img_rgb = self.transform(img_rgb)
            img_depth = self.transform(img_depth)
        else:  # Otherwise define a random one (random cropping, random horizontal flip)
            top = random.randint(0, 256 - INPUT_RESOLUTION)
            left = random.randint(0, 256 - INPUT_RESOLUTION)
            flip = bool(random.getrandbits(1)) #faster than random.choice([True, False])

            """
            Rotating the image and/or flipping -> implementation done in load_image
            """
            # Random multi-modal rotation is required
            if self.do_rot:
                # Define the rotation angle for both the modalities

                trans_rgb = random.choice([0, 1, 2, 3])
                trans_depth = random.choice([0, 1, 2, 3])


            transform = MyTransform([top, left], flip)
            # Apply the same transform to both modalities, rotating them if required
            #img_rgb = img_rgb.flip(1)
            #img_depth = img_depth.flip(1)
            img_rgb = transform(img_rgb, trans_rgb)
            img_depth = transform(img_depth, trans_depth)
            """
            if self.do_flip:
                    trans_rgb = img_rgb.flip(1)
                    trans_depth = img_depth.flip(1)
            """

        if self.do_rot and (self.transform is None):
            calculated_label = get_relative_rotation(trans_rgb, trans_depth)
            #if self.do_flip:
            #    if flip_rgb==flip_depth:
            #        calculated_label += 5
            
            if flip_rgb:
                calculated_label += 10
            if flip_depth:
                calculated_label += 100
            
            return img_rgb, img_depth, target, calculated_label
        return img_rgb, img_depth, target

    def __len__(self):
        return len(self.imgs)
