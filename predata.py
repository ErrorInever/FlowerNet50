import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data.dataset import Dataset


class FlowerDataset(Dataset):

    def __init__(self, data_dir, data_frame, transform=None):
        self.data_dir = data_dir
        self.data_frame = data_frame
        self.transform = transform

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.data_frame.iloc[idx, 0])
        img = Image.open(img_name).convert('RGB')
        label = self.data_frame.iloc[idx, 1]

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data_frame)


def prepare_data(data_dir, val_num=.6, test_num=.8):
    """
    dataset separation
    :param data_dir: path to data_dir
    :param val_num: value of size
    :param test_num: value of size
    :return: three dataframes with columns ['img_path', 'label']
    """
    data_frame = pd.DataFrame(columns=['img_path', 'label'])

    for flower in tqdm(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, flower)
        flowers = os.listdir(folder_path)
        for i in flowers:
            data_frame = data_frame.append(pd.DataFrame({
                'img_path': [os.path.join(flower, i)],
                'label': [flower]}), ignore_index=True)

    # Shuffling for randomness
    data_frame = data_frame.sample(frac=1.0).reset_index(drop=True)

    # dataset separation on [60%, 20%, 20%] of size
    train, validate, test = np.split(data_frame.sample(frac=1),
                                     [int(val_num * len(data_frame)),
                                      int(test_num * len(data_frame))])
    return train, validate, test
