import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import copy
import scipy.io as io
import matlab.engine

class CustomDataset(torch.utils.data.Dataset):  #需要继承data.Dataset
    def __init__(self, state, loss):
        super(CustomDataset, self).__init__()
        self.data = state
        self.target = loss
        self.length = self.data.shape[0]
        pass
    def __getitem__(self, index):
        # 取一个数据
        return self.data[index], self.target[index]
    def __len__(self):
        # 返回一个数据集的长度
        return self.length
 ########################################################################################################################
# 定义神经网络类
########################################################################################################################
class Actor(torch.nn.Module):
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
    # 初始化
    def __init__(self, input_dim, output_dim = 1):
        super(Critic, self).__init__()
        self.lay1 = torch.nn.Linear(input_dim, 20, bias = True)                # 线性层
        self.lay2 = torch.nn.Linear(20, output_dim, bias = True)               # 线性层
        self._initialize_weights()
        
    def forward(self, x):
        x = self.lay1(x)                                                       # 第一隐层
        x =  torch.sigmoid(x)
        output = self.lay2(x)                                                  # 第一隐层
        return output
    
    def _initialize_weights(self):
        torch.nn.init.constant_(self.lay1.weight, 0)
        torch.nn.init.constant_(self.lay2.weight, 0)
        
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
        TriggerTarget = torch.zeros((State.shape[0], 1))
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
 class ValueIterative:
    def __init__(self):
        self.Model = SystemModel()
        self.EndTime = 150
        
    def GetInitActionData(self):
        self.Model.Reset(size=10000, Range=1.5)
        for i in range(self.EndTime):
            u = self.Model.FeddbackControl()
            self.Model.NextState(u)
        State = torch.from_numpy(self.Model.GetHistoryState().astype(np.float32)[0,:-1])
        U = torch.from_numpy(self.Model.GetHistoryPlayer().astype(np.float32)[0])
        TrainData = torch.cat([State, U], dim=1)
        TrainData = TrainData.view(-1, 3)
        TrainData = TrainData.numpy()
        np.random.shuffle(TrainData)
        print(TrainData.shape)
        return TrainData

    def InitAction(self, model, traindata, lr = 0.01):
        criterion = torch.nn.MSELoss(reduction='mean')

        data = traindata[:, :2]
        target = traindata[:,2:]
        data = torch.from_numpy(data.astype(np.float32))
        target = torch.from_numpy(target.astype(np.float32))
        traindata = CustomDataset(data, target)
        train_loader = torch.utils.data.DataLoader(traindata, batch_size=128, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr = lr)

        model.train()
        for i in range(20):
            Recoderloss = 0
            count = 0
            for data,target in train_loader:
                output = model(data)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()  
                optimizer.step()
                Recoderloss = Recoderloss + loss.item()
                count += 1

            print(Recoderloss/count)
        model.cpu()
        return Recoderloss/count 
    
    def GetInitActor(self):
        InitActionData = self.GetInitActionData()
        self.InitAction(self.Model.Player, InitActionData, lr = 0.01)
        torch.save(self.Model.Player, 'InitAction.ckpt')
        
    def GetTrainData(self, Range=1):    
        x = np.arange(-Range, Range, 0.025)
        y = np.arange(-Range, Range, 0.025)
        rangeU = np.arange(-0.5, 0.5, 0.025)
        xx, yy = np.meshgrid(x, y)                                              
        State = np.transpose(np.array([xx.ravel(), yy.ravel()])) 
        State = torch.from_numpy(State.astype(np.float32))
        TrainRData = torch.zeros((0,4))
        TrainLData = torch.zeros((0,4))
        for i in rangeU:
            self.Model.Reset(State=State)
            u = self.Model.Player(State).detach()+ i
            NextState = self.Model.NextState(u)
            NextU = self.Model.Player(NextState).detach()
            Xs = torch.cat([NextState, NextU], dim=1)
            Reinforcement_signal = self.Model.utility + self.Model.alpha * self.Model.Signal(Xs).detach()
            Loss = Reinforcement_signal + self.Model.gamma * self.Model.ValueFunction(Xs).detach()
        
            TrainReinforcement_signalData = torch.cat([State, u, Reinforcement_signal], dim=1)
            TrainReinforcement_signalData = TrainReinforcement_signalData.view(-1, 4)
            TrainLoss = torch.cat([State, u, Loss], dim=1)
            TrainLoss = TrainLoss.view(-1, 4)
            TrainRData = torch.cat([TrainRData, TrainReinforcement_signalData], dim=0)
            TrainLData = torch.cat([TrainLData, TrainLoss], dim=0)
        TrainRData = TrainRData.numpy()
        print(TrainRData.shape)
        TrainLData = TrainLData.numpy()
        print(TrainLData.shape)
        return TrainRData, TrainLData
    
    def GetInitTrainData(self, Range=1.8):    
        x = np.arange(-Range, Range, 0.01)
        y = np.arange(-Range, Range, 0.01)
        xx, yy = np.meshgrid(x, y)                                              
        State = np.transpose(np.array([xx.ravel(), yy.ravel()])) 
        State = torch.from_numpy(State.astype(np.float32))
        Reinforcement_signal = (torch.matmul(State, self.Model.Q*20) * State).sum(1).unsqueeze(1)
        Loss = (torch.matmul(State, self.Model.Q*800) * State).sum(1).unsqueeze(1)
        
        TrainReinforcement_signalData = torch.cat([State, Reinforcement_signal], dim=1)
        TrainReinforcement_signalData = TrainReinforcement_signalData.view(-1, 3)
        TrainReinforcement_signalData = TrainReinforcement_signalData.numpy()
        #np.random.shuffle(TrainData)
        print(TrainReinforcement_signalData.shape)
        TrainLoss = torch.cat([State, Loss], dim=1)
        TrainLoss = TrainLoss.view(-1, 3)
        TrainLoss = TrainLoss.numpy()
        #np.random.shuffle(TrainData)
        print(TrainLoss.shape)
        return TrainReinforcement_signalData, TrainLoss
    
    def GetValueFunctionData(self):
        x = np.arange(-2, 2, 0.01)
        y = np.arange(-2, 2, 0.01)
        xx, yy = np.meshgrid(x, y)                                              
        State = np.transpose(np.array([xx.ravel(), yy.ravel()]))           
        State = torch.from_numpy(State.astype(np.float32))
        
        self.Model.Reset(State=State)
        u = self.Model.NeuralControl()
        NextState = self.Model.NextState(u)

        Loss = self.Model.utility + self.Model.ValueFunction(NextState).detach()
        TrainData = torch.cat([State, Loss], dim=1)
        TrainData = TrainData.view(-1, 3)
        TrainData = TrainData.numpy()
        np.random.shuffle(TrainData)
        print(TrainData.shape)
        return TrainData
    
    def PythonTrainValueFunction(self, model, traindata, lr = 0.01):
        criterion = torch.nn.MSELoss(reduction='mean')

        data = traindata[:, :2]
        target = traindata[:,2:]
        data = torch.from_numpy(data.astype(np.float32))
        target = torch.from_numpy(target.astype(np.float32))
        traindata = CustomDataset(data, target)
        train_loader = torch.utils.data.DataLoader(traindata, batch_size=128, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr = lr)

        model.train()
        for i in range(20):
            Recoderloss = 0
            count = 0
            for data,target in train_loader:
                output = model(data)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()  
                optimizer.step()
                Recoderloss = Recoderloss + loss.item()
                count += 1

            print(Recoderloss/count)
        return Recoderloss/count 
    
    def MatlabTrainValueFunction(self, model, traindata, Minaccuracy=0.0002, Maxepochs=100, showWindow=0, MaxAccuracy=0.1):
        traindata = matlab.double(traindata.tolist())
        accuracy = 10000

        while accuracy > MaxAccuracy:
            V, theta1, W, theta2, accuracy = eng.TrainCritic(traindata, Minaccuracy, Maxepochs, showWindow, nargout=5)
            accuracy = float(accuracy)
        model.lay1.weight.data = torch.tensor(V, dtype=torch.float32)
        model.lay1.bias.data = torch.tensor(theta1, dtype=torch.float32).squeeze()
        model.lay2.weight.data = torch.tensor(W, dtype=torch.float32)
        model.lay2.bias.data = torch.tensor(theta2, dtype=torch.float32).squeeze()
        return accuracy
    
    def TrainValueFunction(self):
        self.TrainValueFunctionData = self.GetValueFunctionData()
        accuracy = self.MatlabTrainValueFunction(self.Model.ValueFunction, self.TrainValueFunctionData, showWindow=1)
        print(accuracy)
        #self.PythonTrainValueFunction(self.Model.ValueFunction, self.TrainValueFunctionData, lr=0.01)       
    def TrainAction(self, lr = 0.005, Range=1):
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = optim.Adam(self.Model.Player.parameters(), lr = lr)
        x = np.arange(-Range, Range, 0.01)
        y = np.arange(-Range, Range, 0.01)
        xx, yy = np.meshgrid(x, y)                                              
        State = np.transpose(np.array([xx.ravel(), yy.ravel()]))           
        State = torch.from_numpy(State.astype(np.float32))
        traindata = CustomDataset(State, torch.zeros((State.shape[0], 1)))
        train_loader = torch.utils.data.DataLoader(traindata, batch_size=128, shuffle=True)
        Recoderloss = 0
        count = 0
        self.Model.Player.train()
        for State,target in train_loader:
            U = self.Model.Player(State)
            xc = torch.cat([State, U],dim=1)
            value = self.Model.ValueFunction(xc) 
            loss = criterion(value, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Recoderloss = Recoderloss + loss.item()
            count += 1
        print(Recoderloss/count)
        return loss.item()
    def ViewDynamic(self, controller="Feedback", State=None):
        self.Model.Reset(size=100, State=State)
        if controller=="Feedback":
            for i in range(self.EndTime):
                u = self.Model.FeddbackControl() 
                state = self.Model.NextState(u)
        elif controller=="neural":
            for i in range(self.EndTime):
                u = self.Model.NeuralControl() 
                state = self.Model.NextState(u)
        elif controller=="event-triggered":
            for i in range(self.EndTime):
                u = self.Model.TriggerNeuralControl() 
                state = self.Model.NextState(u)
        print(self.Model.TriggerCount)     
        plt.plot(self.Model.GetHistoryState()[0])
        saveState = self.Model.GetHistoryState()[0]
        
        plt.grid()
        plt.show()
        plt.plot(self.Model.GetHistoryPlayer()[0])
        saveControl = self.Model.GetHistoryPlayer()[0]
        plt.grid()
        plt.show()
        State = torch.from_numpy(self.Model.GetHistoryState().astype(np.float32)[0,:-1])
        U = torch.from_numpy(self.Model.GetHistoryPlayer().astype(np.float32)[0])
        xs = torch.cat([State,U], dim=1)
        output = self.Model.Signal(xs).detach()
        plt.plot(self.Model.GetReinforcement_signal()[0], "blue")
        plt.plot(output.numpy(), "red")
        plt.grid()
        plt.show()
        
        
        plt.plot(self.Model.GetReinforcementLoss()[0], "blue")
        saveLoss = self.Model.GetReinforcementLoss()[0]
        print(saveLoss[0])
        #plt.plot(self.Model.GetReinforcementLoss()[0]*(1/(1-self.Model.beta)), "black")
        output = self.Model.ValueFunction(xs).detach()
        plt.plot((output.numpy()-self.Model.ValueFunction(torch.tensor([[0.0,0.0,0.0]])).detach().numpy())/(1-self.Model.beta), "red")
        savebound = (output.numpy()-self.Model.ValueFunction(torch.tensor([[0.0,0.0,0.0]])).detach().numpy())/(1-self.Model.beta)
        plt.grid()
        plt.show()
        io.savemat('test.mat', {"State":saveState, "Control":saveControl, "Loss":saveLoss, "bound":savebound})
        #plt.plot(self.Model.GetLoss()[0], "yellow")
        #plt.show()
    def ViewValueFunction(self):
        self.Model.Reset()
        for i in range(self.EndTime):
            u = self.Model.NeuralControl() 
            state = self.Model.NextState(u)
        State = torch.from_numpy(self.Model.GetHistoryState().astype(np.float32)[0,:-1])
        output = self.Model.ValueFunction(State)
        plt.plot(self.Model.GetHistoryState()[0])
        plt.grid()
        plt.show()
        plt.plot(output.detach().numpy(), "red")
        plt.plot(self.Model.GetLoss()[0], "blue")
        plt.grid()
        plt.show()
