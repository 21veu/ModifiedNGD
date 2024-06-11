import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import gc

# theoretical_loss_list = []

data = np.load(f'results/perturb/1/modified_NGD/theoretical_loss1.npy')
data = np.load(f'results/perturb/1/NGD_cut_small/theoretical_loss10.npy')
print(np.mean(data))
# plt.plot(range(data.shape[0]),np.mean(data[-10:])*np.ones_like(data))
plt.plot(range(data.shape[0]),data)
plt.show()
preserved_eigens_list_modify = []
for seed in range(1,21):
    # theoretical_loss_list.append(np.mean(np.load(f'results/perturb/1/modified_NGD/theoretical_loss{seed}.npy')))
    preserved_eigens_list_modify.append(np.mean(np.load(f'results/perturb/1/modified_NGD/preserved_eigens{seed}.npy'), axis=0))
preserved_eigens_list = np.array(preserved_eigens_list_modify)
average_preserved_modify = np.mean(preserved_eigens_list, axis=0)
plt.plot(average_preserved_modify,c='g')
preserved_eigens_list = []
print(np.load(f'results/perturb/1/NGD_cut_small/preserved_eigens1.npy'))
for seed in range(1,21):
    # theoretical_loss_list.append(np.mean(np.load(f'results/perturb/1/modified_NGD/theoretical_loss{seed}.npy')))
    preserved_eigens_list.append(np.mean(np.load(f'results/perturb/1/NGD_cut_small/preserved_eigens{seed}.npy'), axis=0))
preserved_eigens_list = np.array(preserved_eigens_list)
average_preserved = np.mean(preserved_eigens_list, axis=0)

plt.plot(average_preserved, c='r')
plt.show()
# print(np.sum(average_preserved))
# # Output: 99.66890000000001 for modified_NGD
# #         256               for NGD

