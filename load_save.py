import torch

# modified = torch.load('P_check_in_modified.pt')
# optimizer = torch.load('P_check_in_optimizer.pt')
# print(modified.shape, optimizer.shape, torch.norm((modified-optimizer)))

J = torch.randn(32, 2048)
U, D, Vh = torch.linalg.svd(J)
V = Vh.T
D = torch.cat([D, torch.zeros(V.shape[0]-D.shape[0])])
J_rec = U@(torch.eye(U.shape[0], V.shape[0]) * D)@V.T
D2 = torch.pow(D,2)
F = J.T@J
F_inv = (V/D2)@V.T
print("J AND RECONSTRUCTED J: ", torch.linalg.norm(J - J_rec))
print("F AND RECONSTRUCTED F Inverse: ", torch.linalg.norm(torch.eye(F.shape[0]) - F@F_inv))