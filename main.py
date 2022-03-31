import torch
import matlab.engine
from Iterative import ValueIterative
from model import SystemModel
from network import Critic, Actor


if __name__ == '__main__':
   

    # 导入训练好的模型
    TestPolicy = ValueIterative()
    TestPolicy.Model.Player = torch.load('Player.ckpt')
    TestPolicy.Model.Signal = torch.load('Signal.ckpt')
    TestPolicy.Model.ValueFunction = torch.load('ValueFunction.ckpt')

    # 查看训练好的模型响应曲线
    State = torch.tensor([[0.6,-0.6]])
    TestPolicy.Model.beta = 0.01
    TestPolicy.EndTime = 60
    TestPolicy.ViewDynamic(controller="event-triggered", State=State) # 事件触发
    TestPolicy.ViewDynamic(controller="neural", State=State)         # 时间触发

    #训练模型
    '''
    print("Training start")
    eng = matlab.engine.start_matlab()
    for i in range(20):
        # 产生训练数据 不需要系统模型
        TrainReinforcement_signalData,TrainLoss  = TestPolicy.GetTrainData()
        TrainReinforcement_signalData[:,3] = TrainReinforcement_signalData[:,3] - TestPolicy.Model.Signal(torch.tensor([[0.0,0.0,0.0]])).detach().numpy()
        TrainLoss[:,3]= TrainLoss[:,3] - TestPolicy.Model.ValueFunction(torch.tensor([[0.0,0.0,0.0]])).detach().numpy()
        
        # 训练目标网络
        accuracy = TestPolicy.MatlabTrainValueFunction(eng, TestPolicy.Model.Signal, TrainReinforcement_signalData, Minaccuracy=0.00000001, Maxepochs=100, MaxAccuracy=0.01, showWindow=0)
        print(accuracy)
        torch.save(TestPolicy.Model.Signal, 'Signal.ckpt')
        
        # 训练值网络
        accuracy = TestPolicy.MatlabTrainValueFunction(eng, TestPolicy.Model.ValueFunction, TrainLoss, Minaccuracy=0.00000001, Maxepochs=100, MaxAccuracy=0.01, showWindow=0)
        print(accuracy)
        torch.save(TestPolicy.Model.ValueFunction, 'ValueFunction.ckpt')
        
        # 训练动作网络
        for i in range(2):
            TestPolicy.TrainAction(lr = 0.000001)
        torch.save(TestPolicy.Model.Player, 'Player.ckpt')
    '''
