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
        # dout = F.relu(self.fc2(dout))  # 输出层使用 softmax 
        # dout = F.relu(self.fc3(dout))
        dout = self.fc2(dout) # 输出层使用 softmax 
        dout = self.fc3(dout)
        dout = self.fc4(dout)
        return dout
    
    def loss_fun(self, t_p):
        return t_p


class MLP_S(torch.nn.Module):   # 继承 torch 的 Module
    def __init__(self, input_size):
        super(MLP_S,self).__init__()    # 
        self.input_size = input_size 
        n_hidden = [2**9]*4
        in_h = self.input_size
        modules = []
        for i in range(len(n_hidden)):
            out_h = n_hidden[i]
            m = torch.nn.Linear(in_h, out_h)
            modules.append(m)
            m = torch.nn.BatchNorm1d(out_h)
            modules.append(m)
            modules.append(torch.nn.ReLU())
            in_h = out_h
        modules.append(torch.nn.Linear(in_h, 1))
        self.model = torch.nn.Sequential(*modules)
        print(self.model.modules)
        # self.fc1 = torch.nn.Linear(self.input_size,2**8) 
        # self.fc2 = torch.nn.Linear(2**8,2**10)  
        # self.fc3 = torch.nn.Linear(2**10,2**5) 
        # self.fc4 = torch.nn.Linear(2**5,1)


        
    def forward(self,din):
        dout = self.model(din.view(-1,self.input_size))
        # din = din.view(-1,self.input_size) 
        # dout = F.relu(self.fc1(din)) 
        # dout = F.relu(self.fc2(dout))
        # dout = F.relu(self.fc3(dout))
        # dout = self.fc4(dout)

        return dout
    
    def loss_fun(self, t_p):
        return t_p
    


class MLP_H(torch.nn.Module):   # 继承 torch 的 Module
    def __init__(self, input_size):
        super(MLP_H,self).__init__()    # 
        self.input_size = input_size 
        self.fc1 = torch.nn.Linear(self.input_size,2**3) 
        self.fc2 = torch.nn.Linear(2**3,2**4)  
        self.fc3 = torch.nn.Linear(2**4,2**10) 
        self.fc4 = torch.nn.Linear(2**10,1)

        
    def forward(self,din):
        din = din.view(-1,self.input_size) 
        dout = F.gelu(self.fc1(din)) 
        # dout = self.fc2(dout)
        # dout = self.fc3(dout)
        dout = F.gelu(self.fc2(dout))
        dout = F.gelu(self.fc3(dout))
        dout = self.fc4(dout)

        return dout
    
    def loss_fun(self, t_p):
        return t_p
    