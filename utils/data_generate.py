import torch
import numpy as np 
import os   


def generate_synthetic():
    device='cpu'
    train_data = torch.rand(256, device=device)
    valid_data = torch.rand(64, device=device)
    test_data  = torch.rand(64, device=device)
    monte_data = torch.rand(1024, device=device)

    train_data = torch.stack([torch.cos(train_data), torch.sin(train_data)]).T
    valid_data  = torch.stack([torch.cos(valid_data), torch.sin(valid_data)]).T
    test_data  = torch.stack([torch.cos(test_data), torch.sin(test_data)]).T
    # monte_data = torch.stack([torch.cos(monte_data), torch.sin(monte_data)]).T
    train_label = (train_data[:,[0]]*train_data[:,[1]])
    valid_label  = (valid_data[:,[0]]*valid_data[:,[1]])
    test_label  = (test_data[:,[0]]*test_data[:,[1]])
    # monte_label = (monte_data[:,[0]]*monte_data[:,[1]])
    print('data shape', train_data.shape, train_label.shape)
    for u in [0.7,0.75,0.8,0.85,0.9,0.95]:
        save_path = f'./data/synthetic/perturbed_with_condition/u{u}/'
        if os.path.isdir(save_path) is False:
            os.mkdir(save_path)
        sigma = -0.25/np.log(u)
        perturbed_train_data = train_data*torch.exp(-torch.pow(1-train_data,2)/sigma)
        np.save(save_path+'train_data.npy', perturbed_train_data.numpy())
        np.save(save_path+'valid_data.npy', valid_data.numpy())
        np.save(save_path+'test_data.npy', test_data.numpy())
        np.save(save_path+'train_label.npy', train_label.numpy())
        np.save(save_path+'valid_label.npy', valid_label.numpy())
        np.save(save_path+'test_label.npy', test_label.numpy())
    for delta in range(11):
        save_path = f'./data/synthetic/perturbed_with_noise/10pm{delta}/'
        if os.path.isdir(save_path) is False:
            os.mkdir(save_path)
        # print('dsadas', np.power(10., -delta)*torch.randn(*train_data.shape))
        perturbed_train_data = train_data + np.power(10., -delta)*torch.randn(*train_data.shape)
        np.save(save_path+'train_data.npy', perturbed_train_data.numpy())
        np.save(save_path+'valid_data.npy', valid_data.numpy())
        np.save(save_path+'test_data.npy', test_data.numpy())
        np.save(save_path+'train_label.npy', train_label.numpy())
        np.save(save_path+'valid_label.npy', valid_label.numpy())
        np.save(save_path+'test_label.npy', test_label.numpy())
    save_path = f'./data/synthetic/original/'
    if os.path.isdir(save_path) is False:
        os.mkdir(save_path)
    np.save(save_path+'train_data.npy', train_data.numpy())
    np.save(save_path+'valid_data.npy', valid_data.numpy())
    np.save(save_path+'test_data.npy', test_data.numpy())
    np.save(save_path+'train_label.npy', train_label.numpy())
    np.save(save_path+'valid_label.npy', valid_label.numpy())
    np.save(save_path+'test_label.npy', test_label.numpy())




# np.save(f'./data/perturb/{sigma}/gamma_monte.npy', gamma_monte)

def generate_htru2():
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
    train_data  = data_train[:512, :8]                     # (14317,8)
    # for i in range(8):
    #     print(torch.mean(train_data[:,i]))
    # train_data  = train_data*torch.exp(-torch.pow(1-train_data,2)/5)
    # print('\n perturbed \n')
    # for i in range(8):
    #     print(torch.mean(train_data[:,i]))
    train_label = data_train[:512, -1].reshape(-1,1)

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
    valid_data  = data_valid[:128, :8]                     # (14317,8)
    valid_label = data_valid[:128, -1].reshape(-1,1)

    # data_test = np.concatenate([positive_data_for_valid[32:64,:],negative_data_for_valid[100:196,:]],axis=0)
    data_for_test = data[16107:,:]
    data_test = data_for_valid
    np.random.shuffle(data_test)
    # data_test = torch.from_numpy(data_test).to(device)
    test_data  = data_test[:128, :8]                     # (14317,8)
    test_label = data_test[:128, -1].reshape(-1,1)


    for delta in range(11):
        save_path = f'./data/htru2/perturbed_with_noise/10pm{delta}/'
        if os.path.isdir(save_path) is False:
            os.mkdir(save_path)
        perturbed_train_data = train_data + np.power(10., -delta)*np.random.randn(*train_data.shape)
        np.save(save_path+'train_data.npy', perturbed_train_data)
        np.save(save_path+'valid_data.npy', valid_data)
        np.save(save_path+'test_data.npy', test_data)
        np.save(save_path+'train_label.npy', train_label)
        np.save(save_path+'valid_label.npy', valid_label)
        np.save(save_path+'test_label.npy', test_label)
    
    save_path = f'./data/htru2/original'
    if os.path.isdir(save_path) is False:
        os.mkdir(save_path)
    np.save(save_path+'train_data.npy', train_data)
    np.save(save_path+'valid_data.npy', valid_data)
    np.save(save_path+'test_data.npy', test_data)
    np.save(save_path+'train_label.npy', train_label)
    np.save(save_path+'valid_label.npy', valid_label)
    np.save(save_path+'test_label.npy', test_label)

if __name__=='__main__':
    generate_synthetic()

