import torch
import numpy as np
import gc
import copy
from sys import getsizeof

def modified_Fisher_inverse(model, 
                 output :torch.Tensor, 
                 y:torch.Tensor,
                 output_true :torch.Tensor, 
                 y_true:torch.Tensor,
                 alpha:torch.Tensor,
                 alpha_true:torch.Tensor,
                 modify = True):
    """calulcates each layerwise component and returns a torch.Tensor representing the NTK
    
        parameters:
            model: a torch.nn.Module object. Must terminate to a single neuron output
            output: the final single neuron output of the model evaluated on some data
            y: the labels
            output: the final single neuron output of the model evaluated on true date (Here we use validation set)
            y: the labels of true data (Here we use validation set)

        returns:
            NTK: a torch.Tensor representing the emprirical neural tangent kernel of the model
    """
    threshold = 1e-2
    threshold2 = 1e4
    sigma2 = 1e-4

    device = y.device
    NTK = False
    
    # calculate the empirical Jacobian prepared for Fisher on training set 
    if len(output.shape) > 1:
        raise ValueError('y must be 1-D, but its shape is: {}'.format(output.shape))
    
    params_that_need_grad = []
    for param in model.parameters():
        if param.requires_grad:
            params_that_need_grad.append(param.requires_grad)
            #first set all gradients to not calculate, time saver
            param.requires_grad = False
        else:
            params_that_need_grad.append(param.requires_grad)
    J_list = []
    P_check = []
    for i,z in enumerate(model.parameters()):
        if not(params_that_need_grad[i]): #if it didnt need a grad, we can skip it.
            continue
        param = z
        # P_check.append(param.reshape(-1))
        # print('modief params shape: ', param.shape)
        param.requires_grad = True #we only care about this tensors gradients in the loop
        this_grad=[]
        for i in range(len(output)): #first dimension must be the batch dimension
            model.zero_grad()
            output[i].backward(create_graph=True)
            this_grad.append(param.grad.detach().clone().reshape(-1))
            # print('modief params shape: ', param.grad.shape)

        J_layer = torch.stack(this_grad).detach_() 
        # J_list.append(J_layer) 
        # print('\nJ_layer check| max: ', torch.max(J_layer), ' | min: ', torch.min(J_layer))
        J_list.append(J_layer)  
        # if (type(NTK) is bool) and not(NTK):
        #     NTK = J_layer @ J_layer.T # An extra transpose operation to my code for us to feel better
        # else:
        #     NTK += J_layer @ J_layer.T

        param.requires_grad = False
    # torch.save(torch.cat(P_check), 'P_check_in_modified.pt')
    J = torch.cat(J_list, dim=1).to(device)  # [N x P matrix] 
    # print('J shape check: ', J.shape)
    # J = copy.deepcopy(J.detach_().clone().cpu())
    # J = J.to(device)
    # J = copy.deepcopy(J)
    sample_num = J.shape[0]
    param_num = J.shape[1]
    #reset the model object to be how we started this function
    for i,param in enumerate(model.parameters()):
        if params_that_need_grad[i]:
            param.requires_grad = True #

    # calculate the svd decomposition of J
    with torch.no_grad():
        U, Lambda, Vh = torch.linalg.svd(J)
        U = U.detach().clone()
        Lambda = Lambda.detach().clone()
        V = (Vh.detach().clone()).T

    del Vh, J, J_list
    gc.collect()
    torch.cuda.empty_cache()

    # calculate the diagonal of empirical Fisher's eigenvalues
    Lambda2 = torch.pow(Lambda, 2)     # shape (N,)
    # print('Lambda2 shape check: ', Lambda2.shape)    
    #torch.tensor(sigma2, device=device) shape [sample_num], cut for the following computation since the rest elements will be mutiplied by zero
    # calculate the empirical gradient in function space projected onto eigenspace of NTK 
    uTa = (U.T @ alpha.detach().clone())   # shape (1, N)
    # print("U shape check: ", U.shape)    # shape (N, N)
    # print("alpha shape check: ", alpha.shape)   # shape (N,)
    # theoretical_loss = 0.
    # theoretical_loss = theoretical_loss -float(torch.sum(torch.pow(alpha, 2))/sample_num)

    del alpha
    gc.collect()
    torch.cuda.empty_cache()
    # calculate the empirical gradient in parameter space 
    # G_train = aTu * S/sample_num #shape [sample_num], cut for the following computation since the rest elements will be mutiplied by zero

    # Do same thing on the validation set representing for true data 
    if len(output_true.shape) > 1:
        raise ValueError('y must be 1-D, but its shape is: {}'.format(output_true.shape))
    
    J_true_list = []
    #how do we parallelize this operation across multiple gpus or something? that be sweet.
    
    for i,z in enumerate(model.parameters()):
        if not(params_that_need_grad[i]): #if it didnt need a grad, we can skip it.
            continue
        param = z
        param.requires_grad = True #we only care about this tensors gradients in the loop
        this_grad=[]
        for i in range(len(output_true)): #first dimension must be the batch dimension
            model.zero_grad()
            output_true[i].backward(create_graph=True)
            this_grad.append(param.grad.detach().reshape(-1).clone())
        J_true_layer = torch.stack(this_grad) # [N x P matrix] #this will go against our notation, but I'm not adding
        J_true_list.append(J_true_layer)
        # if (type(NTK) is bool) and not(NTK):
        #     NTK = J_layer @ J_layer.T # An extra transpose operation to my code for us to feel better
            # else:
        #     NTK += J_layer @ J_layer.T

        param.requires_grad = False
    J_true = torch.cat(J_true_list, dim=1).to(device).detach()

    del J_true_list
    gc.collect()
    torch.cuda.empty_cache()

    sample_num_t = J_true.shape[0]
    param_num_t  = J_true.shape[1]
    #reset the model object to be how we started this function
    for i,param in enumerate(model.parameters()):
        if params_that_need_grad[i]:
            param.requires_grad = True #
    
    with torch.no_grad():
        U_t ,Lambda_t, Vh = torch.linalg.svd(J_true)
        U_t = U_t.detach().clone()
        Lambda_t = Lambda_t.detach().clone()
        V_t = (Vh.detach().clone()).T

    uTa_t = (U_t.T @ alpha_true.detach().clone())
    del J_true, Vh, U_t, alpha_true
    gc.collect()
    torch.cuda.empty_cache()

    Lambda2_inverse = 1/Lambda2
    # print('\n max Lambda: ', torch.max(Lambda))
    # print('\n min Lambda: ', torch.min(Lambda))

    if modify == True:
        # calculate the modification criterion 
        '''
        Lambda[i] * (V.T @ Lambda_t**3 * uTa_t)[i]/ uTa[i] < 1/2
        '''
        
        # print('\n max Lambda_True: ', torch.max(Lambda_t))
        # print('\n min Lambda_True: ', torch.min(Lambda_t))
        criterion = (Lambda * (V.T @ (V_t[:,:Lambda_t.shape[0]]@(1/Lambda_t*uTa_t).reshape(-1,1))).squeeze()[:sample_num] / uTa.squeeze() < 1/2)
        # Lambda2_inverse = torch.where((criterion>0), 0, Lambda2_inverse)
        Lambda2 = torch.where((criterion>0), 0, Lambda2)
        
        
    if modify==False:
        # print('\n max Lambda: ', torch.max(Lambda))
        # print('\n min Lambda: ', torch.min(Lambda))
        # Cut small 
        # criterion = torch.cat([torch.zeros(74, device=device), torch.ones(Lambda.shape[0]-74, device=device)])
        # Cut large 
        # criterion = torch.cat([torch.ones(Lambda.shape[0]-74, device=device), torch.zeros(74, device=device)])
        # Lambda2_inverse = torch.where((criterion>0), 0, Lambda2_inverse)
        pass
        
    # Lambda2_inverse = Lambda2_inverse*sample_num*torch.tensor(sigma2,device=device)
    # Lambda2_inverse = torch.where(Lambda2_inverse>threshold2, threshold2, Lambda2_inverse)
    # diag_of_modified_Fisher_inverse =  torch.cat([Lambda2_inverse, torch.zeros(param_num-Lambda2_inverse.shape[0], device=device)])    # shape (P,)

    Lambda2 = Lambda2 /(sample_num*torch.tensor(sigma2,device=device))
    Lambda2 = torch.where(Lambda2<threshold, threshold, Lambda2)
    diag_of_modified_Fisher =  torch.cat([Lambda2, torch.zeros(param_num-Lambda2.shape[0], device=device)])    # shape (P,)
    # print('\n max Lambda_Modified: ', torch.max(Lambda2_inverse))
    # print('\n min Lambda_Modified: ', torch.min(Lambda2_inverse))
    
    # print('\n diag_of_modified_Fisher_inverse.shape: ', diag_of_modified_Fisher_inverse.shape)
    # F_inverse_modified = (V) @ (diag_of_modified_Fisher_inverse * V.T)
    # F_inverse_modified = [V, diag_of_modified_Fisher_inverse]

    # F_modified = [V, diag_of_modified_Fisher]
    J_modified = (U*torch.sqrt(Lambda2))@(V.T[:U.shape[0],:])
    # print('max of F', torch.max(F_inverse_modified))
    # print('mean of F', torch.mean(F_inverse_modified))
    del V, V_t, uTa, uTa_t
    gc.collect()
    torch.cuda.empty_cache()
    # return F_inverse_modified, (Lambda2_inverse>0).cpu().numpy()

    return J_modified, (Lambda2>0).cpu().numpy()
