import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os 
import scienceplots
# matplotlib.rcParams['text.usetex'] = True

# mode = 'SGD'
# mode = 'NGD'
# mode = 'modified_NGD'

# perturb = 'perturb'
# # perturb = 'without_perturb'
# seed = 0
# train_loss = np.load(f'results/{perturb}/{mode}/train_loss_seed{seed}.npy')
# test_loss = np.load(f'results/{perturb}/{mode}/test_loss_seed{seed}.npy')
# # print(train_loss)
# # print(test_loss)

# plt.plot(train_loss[20:], c = 'r', label = 'Train loss')
# plt.plot(test_loss[20:], c = 'b', label = 'Test loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend()
# plt.show()


# NGD_test_loss = np.load(f'results/{perturb}/NGD/test_loss_seed{seed}.npy')
# modified_NGD_test_loss = np.load(f'results/{perturb}/modified_NGD/test_loss_seed{seed}.npy')
# SGD_test_loss = np.load(f'results/{perturb}/SGD/test_loss_seed{seed}.npy')

# plt.plot(NGD_test_loss[20:], c = 'r', label = 'NGD Test loss')
# plt.plot(modified_NGD_test_loss[20:], c = 'b', label = 'Modified NGD Test loss')
# plt.plot(SGD_test_loss[20:], c = 'k', label = 'SGD Test loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend()
# plt.show()

def data_arrange(data_path='results/perturb', exclude = [], sigma=1):
    data_dict = {}
    for mode in ['modified_NGD', 'NGD', 'SGD']:
        for sets in ['train', 'test', 'validation']:
            loss_list = []
            for seed in range(1,21):
                if not (seed in exclude):
                    loss = np.load(data_path+f'/{sigma}/{mode}/{sets}_loss_seed{seed}.npy')
                    if (mode != 'SGD') and (sets =='test'):
                        print(f'sigma_{sigma} seed_{seed} mode_{mode} sets_{sets}', np.mean(loss[-10:]))
                    loss_list.append(loss)
            data_dict[f'{mode}_{sets}'] = np.array(loss_list)
            # print(np.array(loss_list).shape)
    return data_dict

def exclude(data_dict, start = 100, thres = 1.):
    nums = data_dict['SGD_test'].shape[1]
    assert start < nums
    ex = []
    for sets in ['test']:
        for mode in ['NGD']:
            data = data_dict[f'{mode}_{sets}']
            for seed in range(20):
                if sets == 'test':
                    if np.mean(data[seed,start:]) > thres:
                        ex.append(seed+1)
    return ex


def plot_fig(data_dict, start = 100, sigma=1):
    nums = data_dict['SGD_test'].shape[1]
    assert start < nums    
    with plt.style.context(['science', 'high-vis']):
        for sets in ['train', 'test', 'validation']:
            if sets == 'train':
                tt = 'Train'
            if sets == 'test':
                tt = 'Test'
            if sets == 'validation':
                tt = 'Validation'
            for mode in ['modified_NGD',  'NGD']:
                data = data_dict[f'{mode}_{sets}']
                print(data.shape[0])
                # # 早鼬
                # if mode == 'modified_NGD':
                #     c = (128/256,154/256,84/256)
                # if mode == 'NGD':
                #     c = (140/256,93/256,66/256)
                # if mode == 'SGD':
                #     c = (218/256,148/256,100/256)
                # 万叶
                if mode == 'modified_NGD':
                    # c = (170/256,221/256,214/256)
                    # c = (130/256,181/256,171/256)
                    c = (195/256,56/256,40/256)
                    # c = (142/256,141/256,50/256)
                    # c = (70/256,73/256,156/256)  #心海
                    mode ='Modified NGD'
                if mode == 'NGD':
                    # c = (170/256,221/256,214/256)
                    # c = (128/256,154/256,84/256)  # 早鼬
                    # c = (195/256,56/256,40/256)
                    c = (140/256,191/256,181/256)
                if mode == 'SGD':
                    c = (231/256,218/256,205/256)
                if mode == 'NGD_cut_small':
                    c = (70/256,73/256,156/256)
                    mode = 'NGD Cut Small'
                # 胡桃
                # if mode == 'modified_NGD':
                #     c = (201/256,71/256,55/256)
                #     mode ='Modified NGD'
                # if mode == 'NGD':
                #     c = (123/256,89/256,94/256)
                # if mode == 'SGD':
                #     c = (231/256,218/256,205/256)
                mu = np.mean(data[:, start:], axis=0)
                if sets =='test' and mode =='NGD':
                    for i in range(data.shape[0]):  
                        print(f'{i+1}', np.mean(data[i, start:]))
                plt.plot(range(start,nums), mu, c=c, label = f'{mode}')
                if data.shape[0] != 1:
                    std = 0.3*np.std(data[:, start:], axis=0)
                    plt.fill_between(range(start,nums),  mu - std, mu + std, color=c, alpha=0.2)
            plt.xlabel('Epochs')
            plt.ylabel('MSE loss')
            plt.title(f'{tt} Loss')
            plt.grid(True)
            plt.legend()
            plt.savefig(f"figures/{sigma}/{sets}_loss" + ".pdf", transparent=True, bbox_inches='tight', pad_inches=0)
            plt.show()

 
if __name__ == '__main__':
    # global sigma
    sigma = 'infty'
    start = 0
    # preserved_eigens_list = []
    # for seed in range(1,21):
    #     preserved_eigens_list.append(np.load(f'/home/yuyi/Documents/ModifiedNGD/results_H/Modified_NGD/preserved_eigens{seed}.npy'))
    # preserved_eigens_arr = np.stack(preserved_eigens_list,axis=0)     # (20, 500, 256)
    # average_preserved = np.mean(preserved_eigens_arr, axis=0)
    # average_preserved = np.mean(average_preserved, axis=0)
    # print(average_preserved.shape)
    # plt.plot(np.arange(1,average_preserved.shape[0]+1), average_preserved, linewidth=2)
    # print(np.sum(average_preserved))
    # preserved_eigens_list = []
    # for seed in range(1,21):
    #     preserved_eigens_list.append(np.load(f'/home/yuyi/Documents/ModifiedNGD/results_H/NGD/preserved_eigens{seed}.npy'))
    # preserved_eigens_arr = np.stack(preserved_eigens_list,axis=0)     # (20, 500, 256)
    # average_preserved = np.mean(preserved_eigens_arr, axis=0)
    # average_preserved = np.mean(average_preserved, axis=0)
    # print(average_preserved.shape)
    # plt.plot(np.arange(1,average_preserved.shape[0]+1), average_preserved, linewidth=2)
    # plt.ylim(0,1.1)
    # plt.grid(True)
    # plt.savefig('/home/yuyi/Documents/ModifiedNGD/figures/preserved_eigens_H.png')
    # print(np.sum(average_preserved))

    preserved_eigens_list = []
    for seed in range(1,21):
        # if seed not in [1,10,12]:#[1,5,10,12,14]:
        preserved_eigens_list.append(np.load(f'/home/yuyi/Documents/ModifiedNGD/results_H/Modified_NGD/test_acc_seed{seed}.npy'))
    preserved_eigens_arr = np.stack(preserved_eigens_list,axis=0)     # (20, 500, 256)
    print(preserved_eigens_arr.shape)
    average_preserved = np.mean(preserved_eigens_arr, axis=0)
    print(preserved_eigens_arr[:,-1])
    
    plt.plot(np.arange(1,average_preserved.shape[0]+1), average_preserved, linewidth=2, c='r', label='Modified NGD')
    preserved_eigens_list = []
    for seed in range(1,21):
        # if seed not in [1,10,12]:#[1,5,10,12,14]:
        preserved_eigens_list.append(np.load(f'/home/yuyi/Documents/ModifiedNGD/results_H/NGD/test_acc_seed{seed}.npy'))
    preserved_eigens_arr = np.stack(preserved_eigens_list,axis=0)     # (20, 500, 256)
    average_preserved = np.mean(preserved_eigens_arr, axis=0)
    print(preserved_eigens_arr[:,-1])
    plt.plot(np.arange(1,average_preserved.shape[0]+1), average_preserved, linewidth=2, c='b', label='NGD')
    # plt.ylim(0,1.1)
    plt.grid(True)
    plt.legend()
    plt.savefig('/home/yuyi/Documents/ModifiedNGD/figures/test_acc_H.png')
    
    # data_dict = data_arrange(sigma=sigma)
    # ex = exclude(data_dict,start=start,thres=0.5)
    # data_dict = data_arrange(exclude=ex, sigma=sigma)
    # plot_fig(data_dict, start=start, sigma=sigma)


    

    # sigma_list = [1, 1.5, 5, 10]
    # data_list  = [None]*len(sigma_list)
    # for i in range(len(sigma_list)):
    #         data_list[i] = data_arrange(sigma=sigma_list[i])
    #         ex = exclude(data_list[i],start=start,thres=0.5)
    #         data_list[i] = data_arrange(exclude=ex, sigma=sigma_list[i]) # data_list shape[4, 20-ex, 500-start]
    # with plt.style.context(['science', 'high-vis']):
    #     for sets in ['test']:
    #         if sets == 'train':
    #             tt = 'Train'
    #             c = (195/256,56/256,40/256)
    #         if sets == 'test':
    #             tt = 'Test'
    #             c = (100/256,26/256,17/256)
    #         if sets == 'validation':
    #             tt = 'Validation'
    #             c = (140/256,191/256,181/256)
    #         mu_list = []
    #         std_list = []
    #         for i in range(len(sigma_list)):
    #             data_dict = data_list[i]
    #             modified_NGD_data = data_dict[f'modified_NGD_{sets}']
    #             NGD_data = data_dict[f'NGD_{sets}']
    #             diff = np.mean(NGD_data[:,-10:], axis=1) - np.mean(modified_NGD_data[:,-10:],axis=1) # get mean for the last 10 epochs
    #             mu = np.mean(diff) # get mean of all seed
    #             std  = 0.1*np.std(diff) # get std of all seed
    #             mu_list.append(mu)
    #             std_list.append(std)
    #         mu_list = np.array(mu_list)
    #         std_list = np.array(std_list)

    #         plt.plot(sigma_list, mu_list, c=c, marker='.',linewidth=1, label = f'Difference of {sets} loss')
    #         plt.fill_between(sigma_list,  mu_list - std_list, mu_list + std_list, color=c, alpha=0.2)
    #     plt.xlabel(r'$\displaystyle\sigma^2$')
    #     plt.ylabel('Difference of MSE loss')
    #     plt.title(f'Loss Difference between NGD and Modified NGD')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.savefig(f"figures/diff_loss" + ".pdf", transparent=True, bbox_inches='tight', pad_inches=0)
    #     plt.show()


    #11111111111111111111111111111111111111111111111111111111
    # modified_NGD , NGD and theoretical_loss
    '''
    the comparision on modified NGD and theoretial loss with perturbation rate sigma^2 = 1
    '''
    #11111111111111111111111111111111111111111111111111111111
    # nums = data_dict['SGD_test'].shape[1]
    # theoretical_loss_list_modify = []
    # for seed in range(1,21):
    #     # theoretical_loss_list.append(np.mean(np.load(f'results/perturb/1/modified_NGD/theoretical_loss{seed}.npy')))
    #     theoretical_loss_list_modify.append(np.load(f'results/perturb/1/modified_NGD/theoretical_loss{seed}.npy'))
    # average_preserved = sum(preserved_eigens_list_modify)/len(preserved_eigens_list_modify)
    # preserved_eigens_list = np.array(preserved_eigens_list_modify)
    # average_preserved_modify = np.mean(preserved_eigens_list, axis=0)
    # preserved_eigens_list = []
    # # print(np.load(f'results/perturb/1/modified_NGD/preserved_eigens1.npy'))
    # for seed in range(1,21):
    #     # theoretical_loss_list.append(np.mean(np.load(f'results/perturb/1/modified_NGD/theoretical_loss{seed}.npy')))
    #     preserved_eigens_list.append(np.mean(np.load(f'results/perturb/1/NGD/preserved_eigens{seed}.npy'), axis=0))
    # average_preserved = sum(preserved_eigens_list)/len(preserved_eigens_list)
    # preserved_eigens_list = np.array(preserved_eigens_list)
    # average_preserved = np.mean(preserved_eigens_list, axis=0)
    # assert start < nums    
    # with plt.style.context(['science', 'high-vis']):
    #     for sets in ['train', 'test', 'validation']:
    #         if sets == 'train':
    #             tt = 'Train'
    #         if sets == 'test':
    #             tt = 'Test'
    #         if sets == 'validation':
    #             tt = 'Validation'
    #         for mode in ['modified_NGD', 'NGD']:
    #             data = data_dict[f'{mode}_{sets}']
    #             print(data.shape[0])
    #             # # 早鼬
    #             # if mode == 'modified_NGD':
    #             #     c = (128/256,154/256,84/256)
    #             # if mode == 'NGD':
    #             #     c = (140/256,93/256,66/256)
    #             # if mode == 'SGD':
    #             #     c = (218/256,148/256,100/256)
    #             # 万叶
    #             if mode == 'modified_NGD':
    #                 c = (140/256,191/256,181/256)
    #                 mode ='Modified NGD'
    #             if mode == 'NGD':
    #                 c = (195/256,56/256,40/256)
    #             if mode == 'SGD':
    #                 c = (231/256,218/256,205/256)
    #             mu = np.mean(data[:, start:], axis=0)
    #             if sets =='test' and mode =='NGD':
    #                 for i in range(data.shape[0]):
    #                     print(f'{i+1}', np.mean(data[i, start:]))
    #             plt.plot(range(start,nums), mu, c=c, label = f'{mode} {sets} loss')
    #             if data.shape[0] != 1:
    #                 std = 0.3*np.std(data[:, start:], axis=0)
    #                 plt.fill_between(range(start,nums),  mu - std, mu + std, color=c, alpha=0.2)
    #         plt.xlabel('Epochs')
    #         plt.ylabel('MSE loss')
    #         plt.title(f'{tt} Loss')
    #         plt.grid(True)
    #         plt.legend()
    #         plt.savefig(f"figures/{sigma}/{sets}_loss" + ".pdf", transparent=True, bbox_inches='tight', pad_inches=0)
    #         plt.show()



    #22222222222222222222222222222222222222222222222222222
    # modified NGD, cut NGD and theoretical loss
    #222222222222222222222222222222222222222222222222222222











    #333333333333333333333333333333333333333333333333333333
    # eigenvalues reserved comparision of modified NGD and cut NGD
    #333333333333333333333333333333333333333333333333333333
    # with plt.style.context(['science', 'high-vis']):
    #     preserved_eigens_list_modify = []
    #     for seed in range(1,21):
    #         # theoretical_loss_list.append(np.mean(np.load(f'results/perturb/1/modified_NGD/theoretical_loss{seed}.npy')))
    #         preserved_eigens_list_modify.append(np.mean(np.load(f'results/perturb/1/modified_NGD/preserved_eigens{seed}.npy'), axis=0))
    #     c = (195/256,56/256,40/256)
    #     preserved_eigens_list = np.array(preserved_eigens_list_modify)
    #     average_preserved_modify = np.mean(preserved_eigens_list, axis=0)
    #     plt.plot(average_preserved_modify,c=c, marker='.', markersize =5, label = 'Modified NGD')
    #     plt.fill_between(range(1,average_preserved_modify.shape[0]+1),  0, average_preserved_modify, color=c, alpha=0.2)
    #     preserved_eigens_list = []
    #     print(np.load(f'results/perturb/1/NGD_cut_small/preserved_eigens1.npy'))
    #     for seed in range(1,21):
    #         # theoretical_loss_list.append(np.mean(np.load(f'results/perturb/1/modified_NGD/theoretical_loss{seed}.npy')))
    #         preserved_eigens_list.append(np.mean(np.load(f'results/perturb/1/NGD_cut_small/preserved_eigens{seed}.npy'), axis=0))
    #     preserved_eigens_list = np.array(preserved_eigens_list)
    #     average_preserved = np.mean(preserved_eigens_list, axis=0)
    #     c = (70/256,73/256,156/256)
    #     plt.plot(average_preserved, c=c, marker='.', markersize =5, label = 'NGD Cut Small')
    #     plt.fill_between(range(1,average_preserved_modify.shape[0]+1),  0, average_preserved, color=c, alpha=0.2)
    #     plt.grid(True)
    #     plt.xlabel('Eigenvalue Index')
    #     plt.ylabel('Presevered Proportion')
    #     plt.title('Eigenvalues Preserved Proportion During Training')
    #     plt.legend()
    #     plt.savefig(f"figures/eigenvalue_preserved" + ".pdf", transparent=True, bbox_inches='tight', pad_inches=0)
    #     plt.show()