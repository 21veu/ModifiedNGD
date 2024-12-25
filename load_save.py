import torch
import numpy as np
from scipy.optimize import minimize

# NGD = torch.stack(torch.load('J_NGD.pt'))
# MNGD = torch.stack(torch.load('J_MNGD.pt'))
J = 1e-6*torch.randn(512,532609)
# J = torch.concat([J, J+1e-4*torch.randn(256,532609)], dim=0)
alpha = torch.randn(512)
print('shape check: ', J.shape, alpha.shape)
K = torch.matmul(J, J.T)
U, Lambda2, Vh = torch.linalg.svd(K)
uTa = (U.T @ alpha.detach().clone())
Solved_true = torch.linalg.solve(K@K, K@alpha)
Solved_true = (J.T@Solved_true).reshape(-1,1)
criterion = (U.T@(J@Solved_true)).reshape(-1)/(uTa)
print("Criterion check: ", criterion, torch.sum(criterion>0))



# J_L = torch.load('P_check_in_modified.pt').reshape(-1)
# Jta = torch.load('P_check_in_optimizer.pt').reshape(-1)
# print("shape: ", J_L.shape, Jta.shape)
# print('Max: ', torch.max(J_L), torch.max(Jta))
# print('Min: ', torch.min(J_L), torch.min(Jta))
# print('Difference: \n', 'norm: ', torch.linalg.norm(J_L-Jta), '\nabs (max, min, mean): ', torch.max(torch.abs(J_L-Jta)), torch.min(torch.abs(J_L-Jta)), torch.mean(torch.abs(J_L-Jta)))

# J = torch.randn(64,1986, device='cuda')
# K = J@J.T
# S, U = torch.linalg.eigh(K)
# print(torch.dist(torch.eye(U.shape[0], device='cuda'), U@U.T))
# print(torch.dist(torch.eye(U.shape[0], device='cuda'), U.T@U))
# print(torch.dist(K, U@torch.diag_embed(S)@U.T))

# U, S, VH = torch.linalg.svd(K)
# print('shape check: ', U.shape, VH.shape)
# print(torch.dist(torch.eye(U.shape[0], device='cuda'), U@U.T))
# print(torch.dist(torch.eye(VH.shape[0], device='cuda'), VH.T@VH))
# print(torch.dist(torch.eye(U.shape[0], device='cuda'), U@VH))
# print(torch.dist(torch.eye(VH.shape[0], device='cuda'), VH@U))
# print(torch.dist(K, U@torch.diag_embed(S)@VH[:, :U.shape[0]]))
# a = (torch.randn(21)==0.)
# b = (torch.randn(21)!=0.)
# print(a*b)
# J = torch.randn(256, 2048)
# alpha = torch.randn(2048)
# print('matmul check: ', torch.norm(J@alpha-(J@alpha.reshape(-1,1)).reshape(-1)))
# U, D, Vh = torch.linalg.svd(J)
# V = Vh.T
# D = torch.cat([D, torch.zeros(V.shape[0]-D.shape[0])])
# J_rec = U@(torch.eye(U.shape[0], V.shape[0]) * D)@V.T
# D2 = torch.pow(D,2)
# F = J.T@J
# F_inv = (V/D2)@V.T
# print("J AND RECONSTRUCTED J: ", torch.linalg.norm(J - J_rec))
# print("F AND RECONSTRUCTED F Inverse: ", torch.linalg.norm(torch.eye(F.shape[0]) - F@F_inv))