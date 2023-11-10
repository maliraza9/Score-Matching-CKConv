import numpy as np

data = np.load('C:/runs/logs/synthetic_timeseries/LV_4cl_p_0.60_0.90_0.50_0.40_pn_0.03333_on_0.01000_tmax_5.npy', allow_pickle=True)[()]
print(data.keys())

SL = data['simulation_list']

# create data for 48 hours x 35 attributes
# observed_values = data['simulation_list']
# print(observed_values.shape)

indlist = np.arange(len(SL))

np.random.seed(1)
np.random.shuffle(indlist)


# 5-fold test: fix nfold = folds = 2 for all values: so indices 40-60% are used as test data.

nfold = 2
# print("dataset length is ",len(SL))
start = (int)(nfold * 0.2 * len(SL))
end = (int)((nfold + 1) * 0.2 * len(SL))
test_index = indlist[start:end]
remain_index = np.delete(indlist, np.arange(start, end))

np.random.seed(1)
np.random.shuffle(remain_index)
num_train = (int)(len(SL) * 0.7)  # assumes num_train is less than last index of remain_index
train_index = remain_index[:num_train] # returns indices from non_test_deleted index from original Dataset SL
valid_index = remain_index[num_train:] # returns indices from non_test_deleted index from original Dataset SL
#
# print("remain_index length is ", len(remain_index))
# print("train index is ", len(train_index))
# print("val index is ", len(valid_index))
# print("test index is ", len(test_index))



"""
some observed_values should be marked as isnan randomly.

depending on fold: folds available for each row: 1024 folds

set fold values as 0




"""
observed_values = SL[1]
missing_ratio = 0.1 #for gt value assignment
maskNan = 0.4

observed_masks = np.ones(observed_values.shape)
print("observed mask shape is ", observed_masks.shape)



# randomly set some percentage as ground-truth

masks = observed_masks.reshape(-1).copy()

obs_indices = masks.tolist()

print(len(obs_indices))
# miss_indices = np.random.choice(
#     obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
# )
# masks[miss_indices] = False
# gt_masks = masks.reshape(observed_masks.shape)
#
# observed_values = np.nan_to_num(observed_values)
# observed_masks = observed_masks.astype("float32")
# gt_masks = gt_masks.astype("float32")
#
# return observed_values, observed_masks, gt_masks
