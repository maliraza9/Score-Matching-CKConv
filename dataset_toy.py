import pickle
import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


# data = np.load("C:/runs/logs/diffmodel_ckconv/data/TS/2_LV_4cl_p_0.60_0.90_0.50_0.80_pn_0.03333_on_0.01000_tmax_5.npy", allow_pickle=True)[()]
# data = np.load("data/TS/2_LV_4cl_p_0.60_0.90_0.50_0.80_pn_0.03333_on_0.01000_tmax_5.npy", allow_pickle=True)[()]
data = np.load("data/TS/6_BRU_2cl_p_1.00_3.00_pn_0.03333_on_0.01000_tmax_10.npy", allow_pickle=True)[()]

def parse_id(id_, missing_ratio=0.1):
    observed_values = []
    observed_values  = id_  # this is the 51,2 SL
    observed_masks = ~np.isnan(observed_values) #python list of True of False in (48,35)
    # randomly set some percentage as ground-truth
    masks = observed_masks.reshape(-1).copy()  #python list of True of False in 48X35 = (1680,). just true for given data
    obs_indices = np.where(masks)[0].tolist()   #list of integer indexes python list (1680-nans,) after removing NaNs. all are observed

    #DO: use it for gt creation
    miss_indices = np.random.choice(
        obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
    )

    masks[miss_indices] = False  #this shape is (102,). assigns false to these miss_indices
    gt_masks = masks.reshape(observed_masks.shape) #this shape is (51,2), just reshaping back to normal
    observed_values = np.nan_to_num(observed_values) #includes NaN but now converted into NaN=0 or inf= +/- >>> number. NOT needed #but ok doesnt make any difference
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")
    #observed values dont matter, observed masks always on, gt is calculated
    return observed_values, observed_masks, gt_masks
#list is already available in form of dataset.


class Toy_Dataset(Dataset):
    def __init__(self, eval_length=51, use_index_list=None, missing_ratio=0.0, seed=0):
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []
        path = (
            "data/toy_missing" + str(missing_ratio) + "_seed" + str(seed) + ".pk"
        )
        # path = (
        #     "C:/runs/logs/diffmodel_ckconv/data/toy_missing" + str(missing_ratio) + "_seed" + str(seed) + ".pk"
        # )

        if os.path.isfile(path) == False:  # if datasetfile is none, create
            for id_ in data['simulation_list']: # do this for all 1024 rows as id_ in data for loop
                observed_values, observed_masks, gt_masks = parse_id(
                    id_, missing_ratio
                )
                self.observed_values.append(observed_values)
                self.observed_masks.append(observed_masks)
                self.gt_masks.append(gt_masks)
            self.observed_values = np.array(self.observed_values) #should be np array already. should be all values of all# 1024,51,2
            self.observed_masks = np.array(self.observed_masks)# 1024,51,2
            self.gt_masks = np.array(self.gt_masks)          # 1024,51,2

            # calc mean and std and normalize values
            # (it is the same normalization as Cao et al. (2018) (https://github.com/caow13/BRITS))
            tmp_values = self.observed_values.reshape(-1, 2)     #is this for 2 variables? so do # 1024X51, 2
            tmp_masks = self.observed_masks.reshape(-1, 2) # 1024X51, 2
            mean = np.zeros(2) #2 instead 35
            std = np.zeros(2)

            for k in range(2): #2 instead 35
                c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
                mean[k] = c_data.mean()
                std[k] = c_data.std()

            self.observed_values = (
                (self.observed_values - mean) / std * self.observed_masks
            )

            with open(path, "wb") as f:
                pickle.dump(
                    [self.observed_values, self.observed_masks, self.gt_masks], f
                )
        else:  # load datasetfile
            with open(path, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks = pickle.load(
                    f
                )
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=None, batch_size=16, missing_ratio=0.1):

    # only to obtain total length of dataset
    dataset = Toy_Dataset(missing_ratio=missing_ratio, seed=seed)
    indlist = np.arange(len(dataset))

    np.random.seed(seed)
    np.random.shuffle(indlist)

    # 5-fold test
    nfold = 2
    start = (int)(nfold * 0.2 * len(dataset))
    end = (int)((nfold + 1) * 0.2 * len(dataset))
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.seed(seed)
    np.random.shuffle(remain_index)
    num_train = (int)(len(dataset) * 0.7) #assumes num_train is less than last index of remain_index
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    dataset = Toy_Dataset(
        use_index_list=train_index, missing_ratio=missing_ratio, seed=seed
    )

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Toy_Dataset(
        use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Toy_Dataset(
        use_index_list=test_index, missing_ratio=missing_ratio, seed=seed
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader, valid_loader, test_loader


