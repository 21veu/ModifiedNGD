import torch
import torch.nn.functional as F   # 激励函数的库

class MLP(torch.nn.Module):   # 继承 torch 的 Module
    def __init__(self):
        super(MLP,self).__init__()    # 
        self.fc1 = torch.nn.Linear(2,2**12)  # 第一个隐含层  
        self.fc2 = torch.nn.Linear(2**12,1)  # 第二个隐含层
        # self.fc3 = torch.nn.Linear(2**14,1)   # 输出层
        
    def forward(self,din):
        din = din.view(-1,2)       # 将一个多行的Tensor,拼接成一行
        dout = F.relu(self.fc1(din))   # 使用 relu 激活函数
        dout = self.fc2(dout)  # 输出层使用 softmax 激活函数
        return dout
    
    def loss_fun(self, t_p):
        return t_p
    
class MLP_B(torch.nn.Module):   # 继承 torch 的 Module
    def __init__(self):
        super(MLP_B,self).__init__()    # 
        self.fc1 = torch.nn.Linear(2,2**8)  
        self.fc2 = torch.nn.Linear(2**8,2)  
        self.fc3 = torch.nn.Linear(2,2**12) 
        self.fc4 = torch.nn.Linear(2**12,1)

        # self.fc3 = torch.nn.Linear(2**14,1)   # 输出层
        
    def forward(self,din):
        din = din.view(-1,2)       # 将一个多行的Tensor,拼接成一行
        dout = F.relu(self.fc1(din))   # 使用 relu 激活函数
        dout = self.fc2(dout)  # 输出层使用 softmax 
        dout = self.fc3(dout)
        dout = self.fc4(dout)
        return dout
    
    def loss_fun(self, t_p):
        return t_p


class MLP_H(torch.nn.Module):   # 继承 torch 的 Module
    def __init__(self):
        super(MLP_H,self).__init__()    # 
        self.fc1 = torch.nn.Linear(8,2**8)  
        self.fc2 = torch.nn.Linear(2**8,2)  
        self.fc3 = torch.nn.Linear(2,2**12) 
        self.fc4 = torch.nn.Linear(2**12,1)
        # self.sgm = torch.nn.Sigmoid()
        
    def forward(self,din):
        din = din.view(-1,8)       # 将一个多行的Tensor,拼接成一行
        dout = F.relu(self.fc1(din))   # 使用 relu 激活函数
        dout = self.fc2(dout)  # 输出层使用 softmax 
        dout = self.fc3(dout)
        dout = self.fc4(dout)
        # dout = self.sgm(dout)
        return dout
    
    def loss_fun(self, t_p):
        return t_p
    