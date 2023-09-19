import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import json
import pickle




class DriveDataset(Dataset):
    def __init__(self, img_path, img_size, img_channels, labels, set_name, specialization=''):
        
        self.img_path = img_path
        self.img_size = img_size
        self.set_name = set_name
        self.num_labels = labels
        self.specialization = specialization
        self.length = len(os.listdir(self.img_path))
        self.loaded = False
        with open(f'{self.img_path}/../labels.json', 'r') as file:
            self.label_data = json.load(file)

        self.images = torch.zeros((self.length, img_channels, self.img_size, self.img_size), dtype=torch.float32)
        self.labels = torch.zeros((self.length, self.num_labels, 1, 1), dtype=torch.float32)

    def __getitem__(self, index):
        
        if not self.loaded:
            self.LoadInMemory()

        x_img = self.images[index]        
        x_mask = self.labels[index]

        return x_img, x_mask

    def __len__(self):
        return self.length

    def LoadInMemory(self):
        index = 0
        print(f"\nLoading {self.set_name} Set | {self.length} Images\n")
        for i in tqdm(range(self.length)):
            label = np.array([], dtype=np.uint8)
            while True:
                if os.path.exists(f"{self.img_path}/{index}.vec3d"):
                    with open(f"{self.img_path}/{index}.vec3d", 'rb') as file:
                        x_img = pickle.load(file)
                    for data in self.label_data:
                        if data['Image'] == f"{index}.vec3d":
                            for c in data['Label']:
                                label = np.append(label, int(c))
                            break
                    index += 1
                    break
                else:
                    index += 1

            """ Reading image """           
            self.images[i] = torch.from_numpy(x_img)   
            self.labels[i] = torch.from_numpy(label).unsqueeze(-1).unsqueeze(-1)

        self.loaded = True
