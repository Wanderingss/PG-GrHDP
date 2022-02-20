import torch
class CustomDataset(torch.utils.data.Dataset):#需要继承data.Dataset
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