import json
import numpy as np
from torch.utils.data import Dataset
import torch

"""
Create Dataset type for Dataloader from models
- Inputs:
    + A list of training data: data_x
    + A list of labels: data_y
- Outputs:
    + Dataset for Dataloader
"""
class USER_DATASET(Dataset):
    def __init__(self, data_x, data_y) -> None:
        super().__init__()
        self.user_data_x = np.array(data_x)
        self.user_data_y = np.array(data_y) 
    def __len__(self): 
        return self.user_data_x.shape[0]
    def __getitem__(self, idx):
        img = self.user_data_x[idx]
        label = self.user_data_y[idx]
        img = img.reshape(28, -1)
        img = np.expand_dims(img, 0)
        img = torch.Tensor(img)
        return img, label

"""
Get data for each user
- Inputs: 
    + An index of user
    + Train set or test set: True/False
- Outputs:
    + A list of training data: data_x
    + A list of labels: data_y
"""
def get_data_user(user_index, train=True):
    if train == True: dir = "train"
    else: dir = "test"
    file_path = f"data/FeMnist/{dir}/all_data_{user_index}_niid_01_keep_0_{dir}_9.json"
    f = open(file_path)
    data = json.load(f)
    user = data["users"]
    data_x = data["user_data"][user[0]]["x"]
    data_y = data["user_data"][user[0]]["y"]
    return data_x, data_y

def get_user_dataset(user_index):
    train_data_x, train_data_y = get_data_user(user_index=user_index, train=True)
    train_user_dataset = USER_DATASET(train_data_x, train_data_y)
    test_data_x, test_data_y = get_data_user(user_index=user_index, train=False)
    test_user_dataset = USER_DATASET(test_data_x, test_data_y)
    return train_user_dataset, test_user_dataset