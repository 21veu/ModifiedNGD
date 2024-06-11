import torch
import numpy as np    
# gamma_train = torch.rand(256).numpy()
# gamma_valid = torch.rand(64).numpy() 
# gamma_test  = torch.rand(64).numpy() 
# gamma_monte = torch.rand(1024).numpy()

# np.save('./data/without_perturb/gamma_train.npy', gamma_train)
# np.save('./data/without_perturb/gamma_test.npy', gamma_test)
# np.save('./data/without_perturb/gamma_valid.npy', gamma_valid)
# np.save('./data/without_perturb/gamma_monte.npy', gamma_monte)

# sigma = 'infty'
# if sigma != 'infty':
#     gamma_train = gamma_train*np.exp(-np.power(1-gamma_train,2)/sigma)
# np.save(f'./data/perturb/{sigma}/gamma_train.npy', gamma_train)
# np.save(f'./data/perturb/{sigma}/gamma_test.npy', gamma_test)
# np.save(f'./data/perturb/{sigma}/gamma_valid.npy', gamma_valid)
# np.save(f'./data/perturb/{sigma}/gamma_monte.npy', gamma_monte)

import pandas as pd
data_path='./data/htru2/'
data = pd.read_csv(data_path+'HTRU_2.csv').values   # np.array (17897,9)
print('If Nan? ', np.sum(np.isnan(data)))
max_value = np.max(data[:,:8], axis=0)
min_value = np.min(data[:,:8], axis=0)
data[:,:8] = (data[:,:8] - min_value) / (max_value - min_value)
# data = torch.tensor(data, device=device)
# train_data  = data[:14317, :8]                      # (14317,8)
# valid_data  = data[14317:16107, :8]                 # (1790,8)
# test_data   = data[16107:, :8]                      # (1790,8)
# train_label = data[:14317, -1].reshape(-1,1)        # (14317,1)
# valid_label = data[14317:16107, -1].reshape(-1,1)   # (1790,1) 
# test_label  = data[16107:, -1].reshape(-1,1)        # (1790,1)

np.random.shuffle(data)
# data = torch.tensor(data, device=device)
data_for_train = data[:14317,:]   
# positive_mask = (data_for_train[:,-1] == 1)
# negative_mask = ~positive_mask
# positive_data_for_train = data_for_train[positive_mask]
# negative_data_for_train = data_for_train[negative_mask]
# data_train = np.concatenate([positive_data_for_train[:64,:],negative_data_for_train[:448,:]],axis=0)
data_train = data_for_train
np.random.shuffle(data_train)
# data_train = torch.from_numpy(data_train).to(device)
# data = torch.from_numpy(data).to(device)
train_data  = data_train[:, :8]                     # (14317,8)
# for i in range(8):
#     print(torch.mean(train_data[:,i]))
# train_data  = train_data*torch.exp(-torch.pow(1-train_data,2)/5)
# print('\n perturbed \n')
# for i in range(8):
#     print(torch.mean(train_data[:,i]))
train_label = data_train[:, -1].reshape(-1,1)

# data_for_valid = data[14317:,:]   
# positive_mask = (data_for_valid[:,-1] == 1)
# negative_mask = ~positive_mask
# positive_data_for_valid = data_for_valid[positive_mask]
# negative_data_for_valid = data_for_valid[negative_mask]
# data_valid = np.concatenate([positive_data_for_valid[:32,:],negative_data_for_valid[:96,:]],axis=0)
data_for_valid = data[14317:16107,:]
data_valid = data_for_valid
np.random.shuffle(data_valid)
# data_valid = torch.from_numpy(data_valid).to(device)
valid_data  = data_valid[:, :8]                     # (14317,8)
valid_label = data_valid[:, -1].reshape(-1,1)

# data_test = np.concatenate([positive_data_for_valid[32:64,:],negative_data_for_valid[100:196,:]],axis=0)
data_for_test = data[16107:,:]
data_test = data_for_valid
np.random.shuffle(data_test)
# data_test = torch.from_numpy(data_test).to(device)
test_data  = data_test[:, :8]                     # (14317,8)
test_label = data_test[:, -1].reshape(-1,1)

np.save('./data/htru2/o/train_data.npy', train_data)
np.save('./data/htru2/o/valid_data.npy', valid_data)
np.save('./data/htru2/o/test_data.npy', test_data)
np.save('./data/htru2/o/train_label.npy', train_label)
np.save('./data/htru2/o/valid_label.npy', valid_label)
np.save('./data/htru2/o/test_label.npy', test_label)

