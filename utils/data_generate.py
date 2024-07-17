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



def generate_MNIST():
    # import json

    data_path='./data/MNIST/'
    # with open(data_path+'mnist_784_json.json')
    import pandas as pd
    
    raw_data = pd.read_csv(data_path+'mnist_784_csv.csv').values   # np.array (17897,9)
    print('If Nan? ', np.sum(np.isnan(raw_data)))
    print(raw_data.shape, raw_data[0])


    np.random.shuffle(raw_data)
    data  = raw_data[:, :-1]/255
    label = raw_data[:,-1]


    train_data  = data[:512]                     # (14317,8)
    train_label = (torch.nn.functional.one_hot(torch.from_numpy(label[:512]), num_classes=10)).numpy()
    print('shape check: ', train_label.shape, train_label[0])    #shape check:  (512, 10) [[0 0 1 0 0 0 0 0 0 0]]
    print('Value check: ', train_data[0])


    valid_data  = data[512:512+128]                     # (14317,8)
    valid_label = (torch.nn.functional.one_hot(torch.from_numpy(label[512:512+128]), num_classes=10)).numpy()


    test_data  = data[512+128: 512+256]                   # (14317,8)
    test_label = (torch.nn.functional.one_hot(torch.from_numpy(label[512+128: 512+256]), num_classes=10)).numpy()


    for delta in range(11):
        save_path = data_path+f'perturbed_with_noise/10pm{delta}/'
        if os.path.isdir(save_path) is False:
            os.makedirs(save_path)
        perturbed_train_data = train_data + np.power(10., -delta)*np.random.randn(*train_data.shape)
        np.save(save_path+'train_data.npy', perturbed_train_data)
        np.save(save_path+'valid_data.npy', valid_data)
        np.save(save_path+'test_data.npy', test_data)
        np.save(save_path+'train_label.npy', train_label)
        np.save(save_path+'valid_label.npy', valid_label)
        np.save(save_path+'test_label.npy', test_label)
    
    save_path = data_path+'original/'
    if os.path.isdir(save_path) is False:
        os.makedirs(save_path)
    np.save(save_path+'train_data.npy', train_data)
    np.save(save_path+'valid_data.npy', valid_data)
    np.save(save_path+'test_data.npy', test_data)
    np.save(save_path+'train_label.npy', train_label)
    np.save(save_path+'valid_label.npy', valid_label)
    np.save(save_path+'test_label.npy', test_label)


def generate_houseprice():
    # import json

    data_path='./data/housePrice/'
    # with open(data_path+'mnist_784_json.json')
    import pandas as pd

    tr = pd.read_csv(data_path+'train.csv')   # np.array (17897,9)

    # We could see columns "Alley, MasVnrType, PoolQC, Fence, MiscFeature" having very few non null values. So we need to remove these columns. FireplaceQu column has 770 non null valuees out of 1460 rows.
    tr = tr.drop(columns=["Alley", "MasVnrType", "PoolQC", "Fence","Id","MiscFeature"])
    # We have 690 null values in "FireplceQu" column, we can't replace all of them with mode, so we are removing this column.
    # Let us also drop some other unnecessary columns which does not have much effect on output
    # MoSold, does not have much effect
    tr = tr.drop(columns=["FireplaceQu","MoSold"])
    tr.dropna(thresh=2)
    # Let us find out the columns with null values
    x=[]
    for i in range(len(tr.columns)):
        for j in range(tr.shape[0]):
            if pd.isna(tr.iloc[j, i]):
                x.append(tr.columns[i])
                break
    #Let us fill the null values with strategy
    for a in x:
        if tr[a].dtype in [int, float]:  
            tr[a] = tr[a].fillna(tr[a].mean())
        else:
            tr[a] = tr[a].fillna(tr[a].mode()[0])
    #Replacing the columns with ordinal values.
    tr[['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','GarageQual','GarageCond']]=tr[['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','GarageQual','GarageCond']].replace(["Ex","Gd","TA","Fa","Po","NA"],[5,4,3,2,1,0])

    #The difference in the years responsible for the price rather than the year.

    tr['YR_YB']=tr['YearRemodAdd']-tr['YearBuilt']
    tr['YS_YR']=tr['YrSold']-tr['YearRemodAdd']
    tr['YS_GBY']=tr['YrSold']-tr['GarageYrBlt']

    tr.drop(columns=['YearRemodAdd','YearBuilt','YrSold','GarageYrBlt'])

    c=0
    for i in tr.columns:
        if tr[i].dtype==object:
            c+=1

    z=[]
    for i in tr.columns:
        if tr[i].dtypes=='object':
            z.append(i)
    
    dummies_df=pd.get_dummies(tr[z])
    dummies_df = dummies_df.astype(int)

    tr = pd.concat([tr, dummies_df], axis=1)

    tr=tr.drop(columns=z)

    X=tr.drop(columns=['SalePrice']) 
    y=np.array(tr['SalePrice'])
    corr_matrix=X.corr()
    columns=corr_matrix.columns

    #Let us find the features having correlation more than 0.8
    correlation_dict = {}
    for i in range(0, 240):
        correlated_features = []#Stores all the correlated feartures of i th column
        for j in range(i + 1, 240):
            if corr_matrix.loc[columns[i], columns[j]] > 0.8:
                correlated_features.append(columns[j])
        if correlated_features:#To store in dictionary only if there exsits correlated features for a feature
            correlation_dict[columns[i]] = correlated_features

    modified_values = [value[0] for value in correlation_dict.values()]
    X=X.drop(columns=modified_values)

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=125)  
    X_pca = pca.fit_transform(X_scaled)

    max_value = np.max(X_pca, axis=0)
    min_value = np.min(X_pca, axis=0)
    X_pca = ((X_pca - min_value) / (max_value - min_value) - 0.5) * 2

    max_value = np.max(y, axis=0)
    min_value = np.min(y, axis=0)
    y = (y - min_value) / (max_value - min_value)

    
    print('data check: ', X_pca.shape, y.shape)
    X_pca = X_pca[:(X_pca.shape[0]//256)*256]
    y     = y[:(y.shape[0]//256)*256]
    import sklearn
    from sklearn.model_selection import train_test_split
    train_data, X_tv, train_label, y_tv = train_test_split(X_pca, y.reshape(-1,1), test_size=0.4,random_state=18)
    valid_data, test_data, valid_label, test_label = train_test_split(X_tv, y_tv, test_size=0.5,random_state=18)

    print('data check: ', train_data.shape, train_label.shape, valid_data.shape, valid_label.shape, test_data.shape, test_label.shape)

    # te=pd.read_csv(data_path+'test.csv')

    # te=te.drop(columns=["Alley", "MasVnrType", "PoolQC", "Fence","Id","MiscFeature"])
    # te= te.drop(columns=["FireplaceQu","MoSold"])
    # te.dropna(thresh=2)

    # y1=[]
    # for i in range(len(te.columns)):
    #     for j in range(te.shape[0]):
    #         if pd.isna(te.iloc[j, i]):
    #             y1.append(te.columns[i])
    #             break

    # for a in y1:
    #     if te[a].dtype in [int, float]:  
    #         te[a] = te[a].fillna(te[a].mean())
    #     else:
    #         te[a] = te[a].fillna(te[a].mode()[0])
    
    # te[['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','GarageQual','GarageCond']]=tr2[['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','GarageQual','GarageCond']].replace(["Ex","Gd","TA","Fa","Po","NA"],[5,4,3,2,1,0])

    # te['YR_YB']=te['YearRemodAdd']-te['YearBuilt']
    # te['YS_YR']=te['YrSold']-te['YearRemodAdd']
    # te['YS_GBY']=te['YrSold']-te['GarageYrBlt']

    # te.drop(columns=['YearRemodAdd','YearBuilt','YrSold','GarageYrBlt'])

    # e=0
    # for i in te.columns:
    #     if te[i].dtype==object:
    #         e+=1

    # z1=[]
    # for i in te.columns:
    #     if te[i].dtypes=='object':
    #         z1.append(i)

    # dummies_df_te=pd.get_dummies(te[z1])
    # dummies_df_te = dummies_df_te.astype(int)

    # te = pd.concat([te, dummies_df_te], axis=1)

    # # Removing one-hot encoded columns
    # te=te.drop(columns=z)

    # te=te.drop(columns=modified_values)

    # scaler1 = StandardScaler()
    # X_scaled1 = scaler1.fit_transform(te)

    # pca1= PCA(n_components=125)  
    # X_pca_te = pca1.fit_transform(X_scaled1)

    # y_te = pd.read_csv(data_path+'sample_submission.csv')

    

    







    # train_data  = data[:512]                     # (14317,8)
    # train_label = (torch.nn.functional.one_hot(torch.from_numpy(label[:512]), num_classes=10)).numpy()
    # print('shape check: ', train_label.shape, train_label[0])    #shape check:  (512, 10) [[0 0 1 0 0 0 0 0 0 0]]
    # print('Value check: ', train_data[0])


    # valid_data  = data[512:512+128]                     # (14317,8)
    # valid_label = (torch.nn.functional.one_hot(torch.from_numpy(label[512:512+128]), num_classes=10)).numpy()


    # test_data  = data[512+128: 512+256]                   # (14317,8)
    # test_label = (torch.nn.functional.one_hot(torch.from_numpy(label[512+128: 512+256]), num_classes=10)).numpy()


    for delta in range(11):
        save_path = data_path+f'perturbed_with_noise/10pm{delta}/'
        if os.path.isdir(save_path) is False:
            os.makedirs(save_path)
        perturbed_train_data = train_data + np.power(10., -delta)*np.random.randn(*train_data.shape)
        np.save(save_path+'train_data.npy', perturbed_train_data)
        np.save(save_path+'valid_data.npy', valid_data)
        np.save(save_path+'test_data.npy', test_data)
        np.save(save_path+'train_label.npy', train_label)
        np.save(save_path+'valid_label.npy', valid_label)
        np.save(save_path+'test_label.npy', test_label)
    
    save_path = data_path+'original/'
    if os.path.isdir(save_path) is False:
        os.makedirs(save_path)
    np.save(save_path+'train_data.npy', train_data)
    np.save(save_path+'valid_data.npy', valid_data)
    np.save(save_path+'test_data.npy', test_data)
    np.save(save_path+'train_label.npy', train_label)
    np.save(save_path+'valid_label.npy', valid_label)
    np.save(save_path+'test_label.npy', test_label)

if __name__=='__main__':
    generate_houseprice()

