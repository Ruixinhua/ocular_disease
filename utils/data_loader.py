import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class OcularDataset(Dataset):
    def __init__(self, df_path, dataset_path, img_size, down_sampling=True, disease_labels=None):
        super().__init__()
        # load df file
        df = pd.read_csv(df_path)
        if not disease_labels:
            disease_labels = ["O", "M", "C", "H", "D", "A", "G"]
        # retrieve filenames of normal and disease cases, I treat all the cases that is not normal as disease
        normal_df = df.loc[df.labels == 'N']["filename"]
        # retrieve disease filenames
        disease_index = [label in disease_labels for label in df.labels.values]
        disease_df = df.loc[disease_index]["filename"]
        if down_sampling:
            # we want to make sure that the number of positive cases is equal to negative.
            if normal_df.count() < disease_df.count():
                # only select part of negative samples
                disease_df = df.loc[df.labels != 'N']["filename"].sample(normal_df.count(), random_state=42)
            else:
                # only select part of positive samples
                normal_df = df.loc[df.labels != 'N']["filename"].sample(disease_df.count(), random_state=42)
        normal = normal_df.values
        disease = disease_df.values
        # define labels of cases, set 1 for normal cases and 0 for disease cases
        self.labels = np.concatenate((np.ones_like(normal), np.zeros_like(disease)))
        # now we have a balanced dataset with 2873 normal cases and 2873 disease cases
        self.dataset_path = [os.path.join(dataset_path, file) for file in np.concatenate((normal, disease))]
        # normalize image
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # resize image to the img_size
        self.img_size = img_size

    def __getitem__(self, index):
        # read in image
        image = cv2.imread(self.dataset_path[index], cv2.IMREAD_COLOR)
        # resize image
        image = cv2.resize(image, (self.img_size, self.img_size))
        # swap color axis because numpy image is (H, W, C), but torch image is (C, H, W)
        image = image.transpose((2, 0, 1))
        # convert numpy array to torch tensor
        image = self.normalize(torch.tensor(image, dtype=torch.float))
        # image = torch.tensor(image, dtype=torch.float)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label

    def __len__(self):
        return len(self.dataset_path)


class OcularDataloader:
    def __init__(self, train_df_path, valid_df_path, dataset_path, img_size, down_sampling, batch_size, shuffle=True,
                 num_workers=0):
        self.train_dataset = OcularDataset(train_df_path, dataset_path, img_size, down_sampling)
        self.train_loader = DataLoader(self.train_dataset, batch_size, shuffle, num_workers=num_workers)
        self.valid_dataset = OcularDataset(valid_df_path, dataset_path, img_size, down_sampling)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size, shuffle, num_workers=num_workers)

# dataset = OcularDataset("../dataset/full_df.csv", "../dataset/preprocessed_images", 128)
# dataloader = OcularDataloader("../dataset/full_df.csv", "../dataset/preprocessed_images", 128, True, 144, True, 0.2)
# for img, l in dataloader:
#     print(img.shape, l)
