
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from skimage import io, transform
from PIL import Image, ImageOps
import os
import numpy as np
import torchvision


class TrainSet(Dataset):
    def __init__(self, path='./train/Image/', transforms=None):
        self.path = path
        self.list = os.listdir(self.path)   
        
        self.transforms = transforms
        
    def __getitem__(self, index):
        # stuff
        image_path = './train/Image/'
        mask_path = './train/Mask/mask'
        image = Image.open(image_path+self.list[index])
        image = image.convert('L')
        mask = Image.open(mask_path+self.list[index])
        mask = mask.convert('L')
        image = np.array(image, dtype = np.long)
        size = (image.shape[0],image.shape[1])
        mask = torchvision.transforms.CenterCrop(size)(mask)
        mask = np.array(mask, dtype = np.long)
        min_img = np.min(image)
        max_img = np.max(image)
        min_arr = np.ones(image.shape)*min_img
        eps = 1e-10
        image = (image - min_arr)/(max_img-min_img+eps)
        min_mask = np.min(mask)
        max_mask = np.max(mask)
        min_arr = np.ones(mask.shape)*min_mask
        eps = 1e-10
        mask = (mask - min_arr)/(max_mask-min_mask+eps)
        for i in range(0,mask.shape[0]):
            for j in range(0,mask.shape[1]):
                if mask[i,j]>=0.8:
                    mask[i,j]=2.0
                elif mask[i,j]<=0.3:
                    mask[i,j]=0.0
                else:
                    mask[i,j]=1.0
        #mask = np.eye(3,dtype = 'uint8')[mask.astype(int)]
        #print(mask.shape)
        if self.transforms is not None:
            image = self.transforms(image)
            #print("shape of image",image.shape)
            mask = self.transforms(mask)
            #print("shape of mask",mask.shape)
        return (image, mask)

    def __len__(self):
        return len(self.list) # of how many data(images?) you have

    
class ValSet(Dataset):
    def __init__(self, path='./val/Image/', transforms=None):
        self.path = path
        self.list = os.listdir(self.path)
        
        self.transforms = transforms
        
    def __getitem__(self, index):
        # stuff
        image_path = './val/Image/'
        mask_path = './val/Mask/mask'
        image = Image.open(image_path+self.list[index])
        image = image.convert('L')
        mask = Image.open(mask_path+self.list[index])
        mask = mask.convert('L')
        image = np.array(image, dtype = np.long)
        size = (image.shape[0],image.shape[1])
        mask = torchvision.transforms.CenterCrop(size)(mask)
        mask = np.array(mask, dtype = np.long)
        min_img = np.min(image)
        max_img = np.max(image)
        min_arr = np.ones(image.shape)*min_img
        eps = 1e-10
        image = (image - min_arr)/(max_img-min_img+eps)
        min_mask = np.min(mask)
        max_mask = np.max(mask)
        min_arr = np.ones(mask.shape)*min_mask
        eps = 1e-10
        mask = (mask - min_arr)/(max_mask-min_mask+eps)
        for i in range(0,mask.shape[0]):
            for j in range(0,mask.shape[1]):
                if mask[i,j]>=0.8:
                    mask[i,j]=2.0
                elif mask[i,j]<=0.3:
                    mask[i,j]=0.0
                else:
                    mask[i,j]=1.0
        #mask = np.eye(3,dtype = 'uint8')[mask.astype(int)]
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return (image, mask)
        
    def __len__(self):
        return len(self.list)
        

