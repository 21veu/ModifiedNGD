# import sys
# sys.path.append("E:/Documents/A_Mathematics/Works/NG/\Modified_NGD/utils")
# print(sys.path)
import torch
import torch.nn.functional as F   # 激励函数的库
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from utils.modifiedNG import ModifiedNGD
from tqdm import tqdm
from utils.readData import read_dataset_H
from utils.modified_fisher_inverse import modified_Fisher_inverse
import copy
from mlp import MLP_H
import gc
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"


device = 'cuda'
# 定义全局变量
n_epochs = 200  # epoch 的数目
batch_size = 256  # 决定每次读取多少图片
perturb = 'perturb'
# perturb = 'no_perturb'
sigma = 1
train_loader,valid_loader,test_loader = read_dataset_H(batch_size=batch_size, data_path='./data/htru2/o')

# 训练神经网络
def train(model, mode='SGD', lr_decay=False): 
    #定义损失函数和优化器
    lossfunc = torch.nn.MSELoss()
    # lossfunc = torch.nn.BCEWithLogitsLoss()
    if mode == 'SGD':
        lr = 0.1
    else:
        lr = 1
    # 开始训练
    Train_loss = []
    Test_acc = []
    Valid_acc = []
    min_train_loss = 2**20
    max_test_acc = 0
    Train_mark = 2**20
    mark = 0
    counter = 0
    theoretical_loss_list = [0.] * n_epochs
    preserved_eigens_list = [0.] * n_epochs
    # initialize optimizer 
    if mode in ['NGD', 'Modified_NGD']:
        modify = True if mode == 'Modified_NGD' else False
        # print("\n MODIFY: ", modify)
        train_data = [data for data, _ in train_loader][:2]
        train_label = [label for _, label in train_loader][:2]
        valid_data = [data for data, _ in valid_loader][:2]
        valid_label = [label for _, label in valid_loader][:2]
        output=model(torch.cat(train_data,dim=0)).squeeze(1)
        y=torch.cat(train_label,dim=0)
        output_true=model(torch.cat(valid_data,dim=0)).squeeze(1)
        y_true=torch.cat(valid_label,dim=0)
        alpha = (output.detach() - y.detach()).clone()
        alpha_true = (output_true.detach() - y_true.detach()).clone()
        F_inverse_modified, preserved_eigens = modified_Fisher_inverse(model=model,
                output=output, 
                y=y,
                output_true=output_true, 
                y_true=y_true,
                alpha=alpha,
                alpha_true=alpha_true,
                modify=modify)
        model.zero_grad()
        optimizer = ModifiedNGD(params = model.parameters(), lr=lr, F_inverse_modified=F_inverse_modified)
    if mode == 'SGD':
        optimizer = torch.optim.SGD(params = model.parameters(), lr = lr)

    for epoch in tqdm(range(1, n_epochs+1)):
        # Modified_NGD optimizer
        train_loss = 0.
        for data, target in train_loader:
            optimizer.zero_grad()   # 清空上一步的残余更新参数值
            output = model(data)    # 得到预测值
            # print(target, output.shape)
            loss = lossfunc(output,target.reshape(output.shape))  # 计算两者的误差
            loss.backward(create_graph=False, retain_graph=False)         # 误差反向传播, 计算参数更新值

            optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
            # print(data.size(0))
            train_loss += loss.item()*data.size(0)

            # print(train_loss)
        train_loss = train_loss / len(train_loader.sampler)
        # if train_loss < min_train_loss:
        #     min_train_loss = train_loss
        print('Epoch:  {}  \tTraining Loss: {}'.format(epoch, train_loss))
        test_acc = test(model)
        Train_loss.append(train_loss)
        Test_acc.append(test_acc)
        valid_loss = valid(model)
        Valid_acc.append(valid_loss)
        if mode != 'SGD':
            # theoretical_loss_list[epoch-1] = (theoretical_loss)
            preserved_eigens_list[epoch-1] = (preserved_eigens)
        if (epoch<=25) or (epoch>25 and epoch % 10 == 1):
            if mode in ['NGD', 'Modified_NGD']:
                modify = True if mode == 'Modified_NGD' else False
                # print("\n MODIFY: ", modify)
                train_data = [data for data, _ in train_loader][:2]
                train_label = [label for _, label in train_loader][:2]
                valid_data = [data for data, _ in valid_loader][:2]
                valid_label = [label for _, label in valid_loader][:2]
                output=model(torch.cat(train_data,dim=0)).squeeze(1)
                y=torch.cat(train_label,dim=0)
                output_true=model(torch.cat(valid_data,dim=0)).squeeze(1)
                y_true=torch.cat(valid_label,dim=0)
                alpha = (output.detach() - y.detach()).clone()
                alpha_true = (output_true.detach() - y_true.detach()).clone()
                del F_inverse_modified, optimizer,  preserved_eigens
                gc.collect()
                torch.cuda.empty_cache()
                F_inverse_modified,  preserved_eigens = modified_Fisher_inverse(model=model, 
                output=output, 
                y=y,
                output_true=output_true, 
                y_true=y_true,
                alpha=alpha,
                alpha_true=alpha_true,
                modify=modify)
                model.zero_grad()
                optimizer = ModifiedNGD(params = model.parameters(), lr=lr, F_inverse_modified=F_inverse_modified)

        if lr_decay == True:
            if epoch>10:
                if train_loss >= Train_mark:
                    mark += 1
                Train_mark = train_loss
            if mark > 2 and epoch%5==0:
            # if epoch>10:
                mark = 0
                lr = lr*0.5
                print(50*'*'+'learning rate decay'+ 50*'*')
                if mode in ['NGD', 'Modified_NGD']:
                    modify = True if mode == 'Modified_NGD' else False
                    # print("\n MODIFY: ", modify)
                    train_data = [data for data, _ in train_loader][:2]
                    train_label = [label for _, label in train_loader][:2]
                    valid_data = [data for data, _ in valid_loader][:2]
                    valid_label = [label for _, label in valid_loader][:2]
                    output=model(torch.cat(train_data,dim=0)).squeeze(1)
                    y=torch.cat(train_label,dim=0)
                    output_true=model(torch.cat(valid_data,dim=0)).squeeze(1)
                    y_true=torch.cat(valid_label,dim=0)
                    alpha = (output.detach() - y.detach()).clone()
                    alpha_true = (output_true.detach() - y_true.detach()).clone()
                    del F_inverse_modified, optimizer,  preserved_eigens
                    gc.collect()
                    torch.cuda.empty_cache()
                    F_inverse_modified,  preserved_eigens = modified_Fisher_inverse(model=model, 
                            output=output, 
                            y=y,
                            output_true=output_true, 
                            y_true=y_true,
                            alpha=alpha,
                            alpha_true=alpha_true,
                            modify=modify)
                    model.zero_grad()
                    optimizer = ModifiedNGD(params = model.parameters(), lr=lr, F_inverse_modified=F_inverse_modified)
                if mode == 'SGD':
                    optimizer = torch.optim.SGD(params = model.parameters(), lr = lr)
                


        if test_acc > max_test_acc and epoch>=190:
            if os.path.isdir(f'checkpoint/htru2/o/{mode}/') == False:
                os.mkdir(f'checkpoint/htru2/o/{mode}/')
            max_test_acc = test_acc
            # torch.save(model.state_dict(), f'checkpoint/Modified_NGD/mlp_epoch{epoch}.pt')
            torch.save(model.state_dict(), f'checkpoint/htru2/o/{mode}/mlp_epoch{epoch}_seed{seed}.pt')
        # 每遍历一遍数据集，测试一下准确率
        torch.cuda.empty_cache()


    

    # NO cut
    if os.path.isdir(f'results_H/o/{mode}') == False:
        os.mkdir(f'results_H/o/{mode}')
    np.save(f'results_H/o/{mode}/train_loss_seed{seed}.npy', np.array(Train_loss))
    np.save(f'results_H/o/{mode}/test_acc_seed{seed}.npy', np.array(Test_acc))
    np.save(f'results_H/o/{mode}/validation_acc_seed{seed}.npy', np.array(Valid_acc))
    if mode != 'SGD':
        # np.save(f'results_H/{perturb}/{sigma}/{mode}/theoretical_loss{seed}.npy', np.array(theoretical_loss_list))
        np.save(f'results_H/o/{mode}/preserved_eigens{seed}.npy', np.array(preserved_eigens_list))

    # Cut small
    # if os.path.isdir(f'results_H/{perturb}/{sigma}/{mode}_cut_small') == False:
    #     os.mkdir(f'results_H/{perturb}/{sigma}/{mode}_cut_small')
    # np.save(f'results_H/{perturb}/{sigma}/{mode}_cut_small/train_loss_seed{seed}.npy', np.array(Train_loss))
    # np.save(f'results_H/{perturb}/{sigma}/{mode}_cut_small/test_acc_seed{seed}.npy', np.array(Test_acc))
    # np.save(f'results_H/{perturb}/{sigma}/{mode}_cut_small/validation_acc_seed{seed}.npy', np.array(Valid_acc))
    # if mode != 'SGD':
    #     # np.save(f'results_H/{perturb}/{sigma}/{mode}/theoretical_loss{seed}.npy', np.array(theoretical_loss_list))
    #     np.save(f'results_H/{perturb}/{sigma}/{mode}_cut_small/preserved_eigens{seed}.npy', np.array(preserved_eigens_list))

    # # Cut large
    # if os.path.isdir(f'results_H/{perturb}/{sigma}/{mode}_cut_large') == False:
    #     os.mkdir(f'results_H/{perturb}/{sigma}/{mode}_cut_large')
    # np.save(f'results_H/{perturb}/{sigma}/{mode}_cut_large/train_loss_seed{seed}.npy', np.array(Train_loss))
    # np.save(f'results_H/{perturb}/{sigma}/{mode}_cut_large/test_acc_seed{seed}.npy', np.array(Test_acc))
    # np.save(f'results_H/{perturb}/{sigma}/{mode}_cut_large/validation_acc_seed{seed}.npy', np.array(Valid_acc))
    # if mode != 'SGD':
    #     # np.save(f'results_H/{perturb}/{sigma}/{mode}/theoretical_loss{seed}.npy', np.array(theoretical_loss_list))
    #     np.save(f'results_H/{perturb}/{sigma}/{mode}_cut_large/preserved_eigens{seed}.npy', np.array(preserved_eigens_list))


# 在valid set上调试网络
def valid(model):
    # lossfunc = torch.nn.MSELoss()
    correct = 0
    with torch.no_grad():  # 训练集中不需要反向传播
        for data, target in valid_loader:
            data = data
            target = target
            output = model(data)    # 得到预测值
            prediction = torch.where(output>=0.5, 1, 0)
            prediction_mask = (prediction == target.reshape(output.shape))
            correct += int(torch.sum(prediction_mask))
        valid_acc = correct / len(valid_loader.sampler)
    # print('Accuracy of the network on the test images: %d %%' % (
    #     100 * correct / total))
    print('Valid Acc: ', valid_acc)
    return valid_acc

# 在数据集上测试神经网络
def test(model):
    correct = 0
    total = 0
    with torch.no_grad():  # 训练集中不需要反向传播
        for data, target in test_loader:
            data = data
            target = target
            output = model(data)    # 得到预测值
            # print(target, output.shape)
            prediction = torch.where(output>=0.5, 1, 0)
            prediction_mask = (prediction == target.reshape(output.shape))
            correct += int(torch.sum(prediction_mask))
            print(f'correct {int(torch.sum(prediction_mask))} vs. total {output.shape[0]}')
        test_acc = correct / len(test_loader.sampler)
    # print('Accuracy of the network on the test images: %d %%' % (
    #     100 * correct / total))
    print('Test Acc: ', test_acc)
    return test_acc


if __name__ == '__main__':
    # for i in range(1,6):
    # for i in range(6,11):
    # for i in range(11,13):
    # for i in range(21,24):
    # for i in range(16,21):
    # for i in range(1,21): 
    for i in range(1,3):
    # for i in range(3,5):
    # for i in range(5,7):
    # for i in range(7,9):
    # for i in range(9,11):

        
        global seed
        # seed = i
        # print('seed is ', seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        
        # mode = 'SGD'
        # # mode = 'NGD'
        # # mode = 'Modified_NGD'
        # print(40*"-", mode, 40*"-")
        # model = MLP_H().to(device)
        # train(model,mode, lr_decay=True)

        seed = i
        print('seed is ', seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        mode = 'NGD'
        print(40*"-", mode, 40*"-")
        model = MLP_H().to(device)
        train(model,mode, lr_decay=True)

        seed = i
        print('seed is ', seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        mode = 'Modified_NGD'
        print(40*"-", mode, 40*"-")
        model = MLP_H().to(device)  
        train(model,mode, lr_decay=True)

    # model = MLP_H()
    # model.load_state_dict(torch.load('numeric_experiments/checkpoint/mlp_epoch190.pt'))
    # model = model.to(device)
    




