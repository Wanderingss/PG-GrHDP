import torch
import torch.optim as optim
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

from utils import CustomDataset
from model import SystemModel
import matlab.engine

class ValueIterative:
    '''值迭代'''
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
    
    def MatlabTrainValueFunction(self, eng, model, traindata, Minaccuracy=0.0002, Maxepochs=100, showWindow=0, MaxAccuracy=0.1):
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