import torch
import torch.nn as nn


class Actor(torch.nn.Module):
    '''动作网络'''
    def __init__(self, input_dim, output_dim = 1):
        super(Actor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 20, bias = False),
            nn.Tanh(),
            nn.Linear(20, output_dim, bias = False)
        )
        self._initialize_weights()
        
    def forward(self, x):
        output = self.layers(x)
        return output
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)                
        
class Critic(torch.nn.Module):
    '''评价网络'''
    def __init__(self, input_dim, output_dim = 1):
        super(Critic, self).__init__()
        self.lay1 = torch.nn.Linear(input_dim, 20, bias = True)                # 线性层
        self.lay2 = torch.nn.Linear(20, output_dim, bias = True)               # 线性层
        self._initialize_weights()
        
    def forward(self, x):
        x = self.lay1(x)                                                        # 输入
        x =  torch.sigmoid(x)                                                   # sigmoid 激活函数
        output = self.lay2(x)                                                   # 输出
        return output
    
    def _initialize_weights(self): 
        torch.nn.init.constant_(self.lay1.weight, 0)
        torch.nn.init.constant_(self.lay2.weight, 0)
