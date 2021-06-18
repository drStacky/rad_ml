import numpy as np
import random
from skimage.transform import resize
from torchvision import transforms as tfms
import torchvision.transforms.functional as tf


__all__ = [
    'Compose', 'ToTensor', 'Normalize', 'RandomDihedral',
    'RandomHorizontalFlip', 'RandomVerticalFlip', 'RandomRot90',
    'SoftenMask', 'Resize']


class Compose(tfms.Compose):
    def __call__(self, img, tar):
        for t in self.transforms:
            img, tar = t(img, tar)
        return img, tar


class ToTensor(tfms.ToTensor):
    def __call__(self, img, tar):
        return tf.to_tensor(img).double(), tf.to_tensor(tar).double()


class Normalize(tfms.Normalize):
    def __call__(self, img, tar):
        return tf.normalize(img.float(), self.mean, self.std), tar


class RandomDihedral:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, tar):
        if random.random() < self.p:
            img, tar = img[:,::-1].copy(), tar[:,::-1].copy()
        if random.random() < self.p:
            img, tar = img[::-1].copy(), tar[::-1].copy()
        if random.random() < self.p:
            img = np.rot90(img, axes=(0, 1)).copy()
            tar = np.rot90(tar, axes=(0, 1)).copy()

        return img, tar

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, tar):
        if random.random() < self.p:
            return img[:,::-1].copy(), tar[:,::-1].copy()
        else:
            return img, tar

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, tar):
        if random.random() < self.p:
            return img[::-1].copy(), tar[::-1].copy()
        else:
            return img, tar

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'


class RandomRot90:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, tar):
        if random.random() < self.p:
            return np.rot90(img, axes=(0, 1)).copy(), np.rot90(tar, axes=(0, 1)).copy()
        else:
            return img, tar

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'


class SoftenMask:
    """
    Clips target values to [percent * tar_max, (1-percent) * tar_max]
    """
    def __init__(self, pct=0.05, tar_max=1):
        self.pct = pct
        self.tar_max = tar_max

    def __call__(self, img, tar):
        tar = np.clip(tar, self.pct * self.tar_max, (1 - self.pct) * self.tar_max)
        return img, tar

    def __repr__(self):
        return self.__class__.__name__


class Resize:
    def __init__(self, sz=512):
        self.sz = sz

    def __call__(self, img, tar):
        img = resize(img, (self.sz, self.sz), order=1, anti_aliasing=True,
                     preserve_range=True, mode='constant', cval=0)
        tar = resize(tar, (self.sz, self.sz), order=0, anti_aliasing=False,
                     preserve_range=True, mode='constant', cval=0)

        return img, tar

    def __repr__(self):
        return self.__class__.__name__ + f'(sz={self.sz})'
