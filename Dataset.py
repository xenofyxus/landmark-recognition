# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:40:06 2019

@author: vaken
"""

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import time

class GoogleLandmarks(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = 'data/top100labels/' + self.image_arr[index] + '.jpg'
        # Open image
        img_as_img = Image.open(single_image_name)

        # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)
            
    def __len__(self):
        return self.data_len


GoogleLandmarksDataset =  GoogleLandmarks('data/labels2.csv')
    
