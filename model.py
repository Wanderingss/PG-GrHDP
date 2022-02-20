import torch
import numpy as np
from network import Critic, Actor

class SystemModel():
    '''系统数学模型'''
    def __init__(self, dim=2, size=1):
        '''初始化系统参数'''
        #定义系统参数
        self.Q = torch.eye(2) * 0.05
        self.R = 0.05
        self.alpha = 0.8
        self.gamma= 0.9
        self.beta= 0.5
        self.dim = dim
        #创建系统变量
        self.state = torch.rand((size, dim))
        self.nextstate = self.state.clone()
        self.Control = torch.zeros((self.state.shape[0], 1))
        self.utility = 0
        #创建数据存储列表
        self.HistoryState = []
        self.HistoryState.append(self.state.numpy())
        self.HistoryPlayer = []
        self.HistoryUtility = []
        self.HistoryLoss = []
        self.HistoryReinforcement_signal = []
        #创建 Actor-Crtic 网络
        self.ValueFunction = Critic(self.dim+1 , 1)
        self.Signal = Critic(self.dim+1, 1)
        self.Player = Actor(self.dim, 1)
        #创建事件触发变量
        self.TriggerCount = torch.zeros((self.state.shape[0], 1))
        
    def Dynamic(self, u):
        '''系统动力学特性'''
        self.nextstate[:, 0] = 0.1 * self.state[:, 1] + self.state[:, 0]
        self.nextstate[:, 1] = -0.49 * torch.sin(self.state[:, 0]) - 0.02 * self.state[:, 1] + self.state[:, 1] + 0.1 * u[:,0]
        self.state = self.nextstate.clone()
        
        self.HistoryState.append(self.state.clone().numpy())
        return self.state
    
    def FeddbackControl(self): 
        '''P 型反馈控制'''
        u = -1 * self.state[:, 1]
        u = u.unsqueeze(1)
        self.Control = u
        self.HistoryPlayer.append(self.Control.numpy())
        return u
        
    def TriggeringCondition(self):
        TriggerTarget = torch.zeros((self.state.shape[0], 1))
        with torch.no_grad():
            xs = torch.cat([self.state, self.Control], dim=1)
            u = self.Player(self.state).detach()
            xc = torch.cat([self.state, u], dim=1)
            self.triggeringCondition = self.ValueFunction(xs).detach() - self.ValueFunction(xc).detach() \
                                        - (self.beta) * self.Signal(xs).detach() \
                                        + (self.beta) * self.Signal(torch.tensor([[0.0,0.0,0.0]])).detach()
        TriggerTarget[self.triggeringCondition >= 0] = 1
        return TriggerTarget
    def NeuralControl(self):
        '''Actor 控制'''
        with torch.no_grad():
            u = self.Player(self.state).detach()
            
        self.Control = u
        self.HistoryPlayer.append(self.Control.clone().numpy())
        return self.Control
    def TriggerNeuralControl(self):
        self.Trigger = self.TriggeringCondition()
        with torch.no_grad():
            u = self.Player(self.state).detach()
            
        self.Control[self.Trigger==1] = u[self.Trigger==1]
        self.HistoryPlayer.append(self.Control.clone().numpy())
        
        Count = torch.zeros_like(self.Trigger)
        Count[self.Trigger==1] = 1
        self.triggerState = self.state
        self.TriggerCount = self.TriggerCount + Count
        return self.Control
    
    def Utility(self, u):
        '''效用函数'''
        self.utility = (torch.matmul(self.state, self.Q) * self.state).sum(1).unsqueeze(1)  + self.R * u * u
        self.HistoryUtility.append(self.utility.clone().numpy())
        return self.utility
    
    def Reinforcement(self, u):
        '''internal reinforcement signal'''
        x_s = torch.cat([self.state, u], dim=1)
        self.reinforcement_signal =  self.Signal(x_s).detach()
        self.HistoryReinforcement_signal.append(self.reinforcement_signal.clone().numpy())
        return self.reinforcement_signal
    
    def NextState(self, u):
        '''更新系统状态'''
        self.Utility(u)
        self.Reinforcement(u)
        state = self.Dynamic(u)
        return state   
    
    def GetHistoryState(self):
        '''获取系统历史状态'''
        return np.array(self.HistoryState).transpose(1, 0, 2)
    
    def GetHistoryPlayer(self):
        '''获取系统历史状态'''
        return np.array(self.HistoryPlayer).transpose(1, 0, 2)
    
    def GetReinforcement_signal(self):
        '''获取系统历史损失'''
        self.SystemReinforcement_signal = np.array(self.HistoryUtility).transpose(1, 0, 2)
        temp = np.array(self.HistoryUtility).transpose(1, 0, 2)
        for i in range(temp.shape[1]):
            SUM = 0
            for j in range(temp.shape[1]-1, i-1, -1):
                SUM = self.alpha * SUM + temp[:,j,:]
            self.SystemReinforcement_signal[:,i,:] = SUM
        return self.SystemReinforcement_signal
    
    def GetReinforcementLoss(self):
        '''获取系统历史损失'''
        self.ReinforcementLoss = np.array(self.HistoryReinforcement_signal).transpose(1, 0, 2)
        temp = np.array(self.HistoryReinforcement_signal).transpose(1, 0, 2)
        for i in range(temp.shape[1]):
            SUM = 0
            for j in range(temp.shape[1]-1, i-1, -1):
                SUM = self.gamma * SUM + temp[:,j,:]
            self.ReinforcementLoss[:,i,:] = SUM
        return self.ReinforcementLoss
    
    def GetLoss(self):
        '''获取系统历史损失'''
        self.HistoryLoss = np.array(self.SystemReinforcement_signal)
        temp = np.array(self.SystemReinforcement_signal)
        for i in range(temp.shape[1]):
            SUM = 0
            for j in range(temp.shape[1]-1, i-1, -1):
                SUM = self.gamma * SUM + temp[:,j,:]
            self.HistoryLoss[:,i,:] = SUM
        return self.HistoryLoss
    
    def Reset(self, size=1, Range=1, State = None):
        '''重置系统'''
        if State == None:
            self.state = (torch.rand((size, self.dim))-0.5) * 2 * Range
        else:
            self.state = State
        self.nextstate = self.state.clone()
        self.Control = torch.zeros((self.state.shape[0], 1))
        self.HistoryState = []
        self.HistoryState.append(self.state.numpy())
        self.HistoryPlayer = []
        self.HistoryUtility = [] 
        self.HistoryLoss = []
        self.HistoryReinforcement_signal = []
        self.TriggerCount = torch.zeros((self.state.shape[0], 1))