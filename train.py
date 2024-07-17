import torch
import torch.nn.functional as F   
import numpy as np
from utils.modifiedNG import ModifiedNGD
from tqdm import tqdm
from utils.readData import read_dataset
from utils.modified_fisher_inverse import modified_Fisher_inverse
import copy
from utils.NNModel import *
import gc
import os
import argparse
import sys


parser = argparse.ArgumentParser()
parser.add_argument('-Day', '--date', type=str, default='Now')
parser.add_argument('-Ep', '--n_epochs', type=int, default=200)
parser.add_argument('-B', '--batch_size', type=int, default=512)
parser.add_argument('-PM', '--perturb_mode', type=str, choices=['original', 'noise', 'condition'], default='original')
parser.add_argument('-Dev', '--device', type=str, choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('-DN', '--data_name', type=str, default='synthetic')
parser.add_argument('-Sig', '--sigma', type=int, default=1)
parser.add_argument('-u', '--u', type=float, default=0.9)
parser.add_argument('-SGD', '--SGD', default=False, action='store_true')
parser.add_argument('-NGD', '--NGD', default=False, action='store_true')
parser.add_argument('-MNGD', '--MNGD', default=False, action='store_true')

args = parser.parse_args()

mydate = args.date
device = args.device
data_name = args.data_name
n_epochs = args.n_epochs 
batch_size = args.batch_size
perturb_mode = args.perturb_mode
sigma = args.sigma
u = args.u

mode = np.array(['SGD', 'NGD', 'MNGD'])
mode_idx = np.array([args.SGD, args.NGD, args.MNGD]).astype(bool)
mode = mode[mode_idx][0]
# print('MODE CHECK: ', mode, f'data/{mode}')


if perturb_mode == 'noise':
    file_path = f'{data_name}/perturbed_with_{perturb_mode}/10pm{sigma}/'
if perturb_mode == 'condition':
    file_path = f'{data_name}/perturbed_with_{perturb_mode}/u{u}'
if perturb_mode == 'original':
    file_path = f'{data_name}/{perturb_mode}/'

train_loader,valid_loader,test_loader = read_dataset(batch_size=batch_size, data_path=f'./data/{file_path}')
for data, target in train_loader:
    input_size = data.shape[1]
    print(data.shape, target.shape)
    break
lossfunc = torch.nn.MSELoss()
# lossfunc = torch.nn.CrossEntropyLoss()

class local_minima_detector():
    def __init__(self, window_length=5, threshold=1e-1):
        self.local_minima_window = np.array([],dtype=np.float64)
        self.window_length = window_length
        self.threshold = threshold

    def detector(self, loss_present):
        '''
        To detect the local minima with a window of length l.
        @param window_length: the length of window;
        @param threshold: the ratio coefficient.
        '''
        if self.local_minima_window.shape[0] < self.window_length:
            print('local minima detector shape: ', self.local_minima_window.shape)
            self.local_minima_window = np.append(self.local_minima_window, loss_present)
            return False
        else:
            self.local_minima_window[:-1] = self.local_minima_window[1:]
            self.local_minima_window[-1]  = loss_present
            local_mean = np.mean(self.local_minima_window)
            print('std: ', np.std(self.local_minima_window), '\nthres: ', local_mean*self.threshold)
            indicator = np.std(self.local_minima_window) < local_mean*self.threshold
            if indicator:
                self.local_minima_window = np.array([],dtype=np.float64)
                return True
            else:
                return False

detector = local_minima_detector(threshold=1e-3)

# 训练神经网络
def train(model, mode='SGD', lr_decay=False): 
    #定义损失函数和优化器
    
    # lossfunc = torch.nn.BCEWithLogitsLoss()
    if mode == 'SGD':
        lr = 0.1
    else:
        lr = 1.
    # 开始训练
    Train_loss = []
    Test_loss = []
    Valid_loss = []
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
    if mode in ['NGD', 'MNGD']:
        modify = True if mode == 'MNGD' else False
        # print("\n MODIFY: ", modify)
        train_data = [data for data, _ in train_loader]
        train_label = [label for _, label in train_loader]
        valid_data = [data for data, _ in valid_loader]
        valid_label = [label for _, label in valid_loader]
        output=model(torch.cat(train_data,dim=0)).squeeze(1)
        y=torch.cat(train_label,dim=0)
        print('OUTPUT CHECK: ', output.shape, torch.max(output), torch.min(output), '\nTARGET: ', y.shape, torch.max(y), torch.min(y))
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
        # print('TRAIN SHAPE CHECK: ', F_inverse_modified[0].shape, F_inverse_modified[1].shape)
        optimizer = ModifiedNGD(params = model.parameters(), lr=lr, F_inverse_modified=F_inverse_modified)
    if mode == 'SGD':
        optimizer = torch.optim.SGD(params = model.parameters(), lr = lr)

    for epoch in tqdm(range(1, n_epochs+1), file=sys.stdout):
        print('Epoch:  ', epoch)
        # MNGD optimizer
        train_loss = 0.
        if epoch == 1:
            train_acc, train_loss = test_train(model)
            Train_loss.append(train_loss)
            test_acc, test_loss = test(model)
            Test_acc.append(test_acc)
            Test_loss.append(test_loss)
            valid_acc, valid_loss= valid(model)
            Valid_acc.append(valid_loss)
            Valid_loss.append(valid_loss)
        for data, target in train_loader:
            optimizer.zero_grad()   # 清空上一步的残余更新参数值
            output = model(data)    # 得到预测值
            print('OUTPUT CHECK: ', torch.max(output), torch.min(output), '\nTARGET: ', torch.max(target), torch.min(target))
            loss = lossfunc(output,target.reshape(output.shape))  # 计算两者的误差
            loss.backward(create_graph=False, retain_graph=False)         # 误差反向传播, 计算参数更新值

            optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
            # print(data.size(0))
            train_loss += loss.item()*data.size(0)
            break

            # print(train_loss)
        train_loss = train_loss / len(train_loader.sampler)
        # if train_loss < min_train_loss:
        #     min_train_loss = train_loss
        print('Epoch:  {}  \nTraining Loss: {}'.format(epoch, train_loss))
        Train_loss.append(train_loss)
        test_acc, test_loss = test(model)
        Test_acc.append(test_acc)
        Test_loss.append(test_loss)
        valid_acc, valid_loss= valid(model)
        Valid_acc.append(valid_loss)
        Valid_loss.append(valid_loss)
        if mode != 'SGD':
            # theoretical_loss_list[epoch-1] = (theoretical_loss)
            # print('eigens shape check: ', preserved_eigens.shape)
            print('Preserved_eigens number check: ', np.sum(preserved_eigens))
            preserved_eigens_list[epoch-1] = (preserved_eigens)
        if (epoch<=20) or (epoch>20 and epoch % 10 == 1):
            if mode in ['NGD', 'MNGD']:
                modify = True if mode == 'MNGD' else False
                # print("\n MODIFY: ", modify)
                train_data = [data for data, _ in train_loader]
                train_label = [label for _, label in train_loader]
                valid_data = [data for data, _ in valid_loader]
                valid_label = [label for _, label in valid_loader]
                output=model(torch.cat(train_data,dim=0)).squeeze(1)
                y=torch.cat(train_label,dim=0)
                output_true=model(torch.cat(valid_data,dim=0)).squeeze(1)
                y_true=torch.cat(valid_label,dim=0)
                alpha = (output.detach() - y.detach()).clone()
                alpha_true = (output_true.detach() - y_true.detach()).clone()
                del F_inverse_modified, optimizer, preserved_eigens
                gc.collect()
                torch.cuda.empty_cache()
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

        if lr_decay == True:
            Mark = detector.detector(train_loss)
            if Mark:
            # if epoch>10:
                lr = lr*0.5
                print(50*'*'+'learning rate decay'+ 50*'*')
                if mode in ['NGD', 'MNGD']:
                    modify = True if mode == 'MNGD' else False
                    # print("\n MODIFY: ", modify)
                    train_data = [data for data, _ in train_loader]
                    train_label = [label for _, label in train_loader]
                    valid_data = [data for data, _ in valid_loader]
                    valid_label = [label for _, label in valid_loader]
                    output=model(torch.cat(train_data,dim=0)).squeeze(1)
                    y=torch.cat(train_label,dim=0)
                    output_true=model(torch.cat(valid_data,dim=0)).squeeze(1)
                    y_true=torch.cat(valid_label,dim=0)
                    alpha = (output.detach() - y.detach()).clone()
                    alpha_true = (output_true.detach() - y_true.detach()).clone()
                    del F_inverse_modified, optimizer, preserved_eigens
                    gc.collect()
                    torch.cuda.empty_cache()
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
                


        if test_acc > max_test_acc and epoch>=190:
            if os.path.isdir(f'checkpoint/{file_path}/{mode}') == False:
                os.makedirs(f'checkpoint/{file_path}/{mode}')
            max_test_acc = test_acc
            torch.save(model.state_dict(), f'checkpoint/{file_path}/{mode}/{mydate}_mlp_epoch{epoch}_seed{seed}.pt')
        # 每遍历一遍数据集，测试一下准确率
        torch.cuda.empty_cache()


    

    # NO cut
    if os.path.isdir(f'results/{file_path}/{mode}') == False:
        os.makedirs(f'results/{file_path}/{mode}')
    np.save(f'results/{file_path}/{mode}/{mydate}_train_loss_seed{seed}.npy', np.array(Train_loss))
    np.save(f'results/{file_path}/{mode}/{mydate}_test_loss_seed{seed}.npy', np.array(Test_loss))
    np.save(f'results/{file_path}/{mode}/{mydate}_valid_loss_seed{seed}.npy', np.array(Valid_loss))
    np.save(f'results/{file_path}/{mode}/{mydate}_test_acc_seed{seed}.npy', np.array(Test_acc))
    np.save(f'results/{file_path}/{mode}/{mydate}_validation_acc_seed{seed}.npy', np.array(Valid_acc))
    if mode != 'SGD':
        # np.save(f'results/{perturb}/{sigma}/2pm{delta}/{mode}/theoretical_loss{seed}.npy', np.array(theoretical_loss_list))
        np.save(f'results/{file_path}/{mode}/{mydate}_preserved_eigens{seed}.npy', np.array(preserved_eigens_list))

    # Cut small
    # if os.path.isdir(f'results/{perturb}/{sigma}/{mode}_cut_small') == False:
    #     os.makedirs(f'results/{perturb}/{sigma}/{mode}_cut_small')
    # np.save(f'results/{perturb}/{sigma}/{mode}_cut_small/train_loss_seed{seed}.npy', np.array(Train_loss))
    # np.save(f'results/{perturb}/{sigma}/{mode}_cut_small/test_acc_seed{seed}.npy', np.array(Test_acc))
    # np.save(f'results/{perturb}/{sigma}/{mode}_cut_small/validation_acc_seed{seed}.npy', np.array(Valid_acc))
    # if mode != 'SGD':
    #     # np.save(f'results/{perturb}/{sigma}/{mode}/theoretical_loss{seed}.npy', np.array(theoretical_loss_list))
    #     np.save(f'results/{perturb}/{sigma}/{mode}_cut_small/preserved_eigens{seed}.npy', np.array(preserved_eigens_list))

    # # Cut large
    # if os.path.isdir(f'results/{perturb}/{sigma}/{mode}_cut_large') == False:
    #     os.makedirs(f'results/{perturb}/{sigma}/{mode}_cut_large')
    # np.save(f'results/{perturb}/{sigma}/{mode}_cut_large/train_loss_seed{seed}.npy', np.array(Train_loss))
    # np.save(f'results/{perturb}/{sigma}/{mode}_cut_large/test_acc_seed{seed}.npy', np.array(Test_acc))
    # np.save(f'results/{perturb}/{sigma}/{mode}_cut_large/validation_acc_seed{seed}.npy', np.array(Valid_acc))
    # if mode != 'SGD':
    #     # np.save(f'results/{perturb}/{sigma}/{mode}/theoretical_loss{seed}.npy', np.array(theoretical_loss_list))
    #     np.save(f'results/{perturb}/{sigma}/{mode}_cut_large/preserved_eigens{seed}.npy', np.array(preserved_eigens_list))


# 在valid set上调试网络
def valid(model):
    # lossfunc = torch.nn.MSELoss()
    correct = 0
    valid_loss = 0.
    with torch.no_grad():  # 训练集中不需要反向传播
        for data, target in valid_loader:
            data = data
            target = target
            output = model(data)    # 得到预测值
            loss = lossfunc(output,target.reshape(output.shape))
            prediction = output
            prediction_mask = (prediction == target.reshape(output.shape))
            correct += int(torch.sum(prediction_mask))
            valid_loss += loss.item()*data.size(0)

        valid_loss = valid_loss / len(valid_loader.sampler)
            
        valid_acc = correct / len(valid_loader.sampler)
    # print('Accuracy of the network on the test images: %d %%' % (
    #     100 * correct / total))
    print('Valid Loss: ', valid_loss)
    print('Valid Acc: ', valid_acc)
    return valid_acc, valid_loss

# 在数据集上测试神经网络
def test(model):
    correct = 0
    total = 0
    test_loss = 0.
    with torch.no_grad():  # 训练集中不需要反向传播
        for data, target in test_loader:
            data = data
            target = target
            output = model(data)    # 得到预测值
            # print(target, output.shape)
            loss = lossfunc(output,target.reshape(output.shape))
            prediction = output
            prediction_mask = (prediction == target.reshape(output.shape))
            correct += int(torch.sum(prediction_mask))
            # print(f'correct {int(torch.sum(prediction_mask))} vs. total {output.shape[0]}')
            test_loss += loss.item()*data.size(0)

        test_loss = test_loss / len(test_loader.sampler)
        test_acc = correct / len(test_loader.sampler)
    # print('Accuracy of the network on the test images: %d %%' % (
    #     100 * correct / total))
    print('Test Loss: ', test_loss)
    print('Test Acc: ', test_acc)
    return test_acc, test_loss

def test_train(model):
    correct = 0
    total = 0
    test_loss = 0.
    with torch.no_grad():  # 训练集中不需要反向传播
        for data, target in train_loader:
            data = data
            target = target
            output = model(data)    # 得到预测值
            # print(target, output.shape)
            loss = lossfunc(output,target.reshape(output.shape))
            prediction = output
            prediction_mask = (prediction == target.reshape(output.shape))
            correct += int(torch.sum(prediction_mask))
            # print(f'correct {int(torch.sum(prediction_mask))} vs. total {output.shape[0]}')
            test_loss += loss.item()*data.size(0)

        test_loss = test_loss / len(train_loader.sampler)
        test_acc = correct / len(train_loader.sampler)
    # print('Accuracy of the network on the test images: %d %%' % (
    #     100 * correct / total))
    print('Test train Loss: ', test_loss)
    print('Test train Acc: ', test_acc)
    return test_acc, test_loss


if __name__ == '__main__':
    for i in range(1):
        global seed 

        seed = i
        print('seed is ', seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        print(40*"-", mode, 40*"-")
        model = MLP_M(input_size).to(device)  
        train(model,mode, lr_decay=True)

    # model = MLP_H()
    # model.load_state_dict(torch.load('numeric_experiments/checkpoint/mlp_epoch190.pt'))
    # model = model.to(device)
    




