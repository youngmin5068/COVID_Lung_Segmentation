import torch as torch
import torchvision.transforms as transforms
from custom_transforms import *
import random 
import numpy as np

class customColorJitter(object):
    def __init__(self, SEED, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.seed = SEED

    def __call__(self, image):

        seed = self.seed
        torch.manual_seed(seed)

        # Apply random brightness, contrast, saturation, and hue adjustments
        image = transforms.functional.adjust_brightness(image, self.brightness)
        image = transforms.functional.adjust_contrast(image, self.contrast)
        image = transforms.functional.adjust_saturation(image, self.saturation)
        image = transforms.functional.adjust_hue(image, self.hue)

        return image

class customRandomHorizontalFlip(object):
    def __init__(self,SEED,p=0.5):
        self.p = p
        self.seed = SEED
    def __call__(self, image):
        # Set random seed
        seed = self.seed
        torch.manual_seed(seed)
        # Apply random transformation
        transform = transforms.RandomHorizontalFlip(p=self.p)
        image = transform(image)

        return image
    
class customCenterCrop(object):
    def __init__(self, size,SEED):
        self.size = size
        self.seed = SEED
    def __call__(self, image):
        seed = self.seed
        torch.manual_seed(seed)

        w, h = image.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return transforms.functional.crop(image, y1, x1, th, tw)
    

class customRandomRotate(object):
    def __init__(self, degrees,SEED):
        self.degrees = degrees
        self.seed = SEED
    def __call__(self, image):

        seed = self.seed
        torch.manual_seed(seed)
        # Generate random angle
        angle = transforms.RandomRotation.get_params((-self.degrees, self.degrees))
        # Apply random rotation
        image = transforms.functional.rotate(image, angle)

        return image
    
class CustomRandomCrop(object):
    def __init__(self, size,SEED):
        self.size = size
        self.seed = SEED
    def __call__(self, image):

        seed = self.seed
        torch.manual_seed(seed)
        w, h = image.size
        left = np.random.randint(0, w - self.size[0])
        top = np.random.randint(0, h - self.size[1])
        right = left + self.size[0]
        bottom = top + self.size[1]
        image = transforms.functional.crop(image, top, left, self.size[1], self.size[0])
        return image
    
class customCenterCrop(object):
    def __init__(self, size, SEED):
        self.size = size
        self.seed = SEED

    def __call__(self, image):
        seed = self.seed
        torch.manual_seed(seed)
        # Get dimensions of image
        width = image.size[0]
        height = image.size[1]

        # Calculate crop size
        crop_size = min(width, height, self.size)

        # Calculate crop position
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size

        # Apply center crop
        image = transforms.functional.crop(image, top, left, crop_size, crop_size)

        return image