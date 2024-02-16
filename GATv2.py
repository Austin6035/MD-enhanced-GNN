import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import random_split
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.glob import global_max_pool
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

m_seed = 123
torch.manual_seed(m_seed)
torch.cuda.manual_seed_all(m_seed)


class DataLoad(Dataset):

    def __init__(self, path):
        self.dataset = torch.load(path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


dataset = DataLoad(path='dataset.pt')
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=64, num_workers=0)
val_loader = DataLoader(dataset=val_dataset, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=0)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATv2Conv(71, 50, 5, concat=False, edge_dim=41)
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(50 + 161, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 1)
        )
        self.relu = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, edge_attr, batch, fp = data.x, data.edge_index, data.edge_attr, data.batch, data.feature
        x = self.relu(self.conv1(x, edge_index, edge_attr))
        x = global_max_pool(x, batch)
        x = torch.cat([x, fp.reshape(data.num_graphs, 161)], dim=1)
        x = self.MLP(x.float())
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net()
model.to(device)

criterion = nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)


class EarlyStopping:

    def __init__(self, patience=10, verbose=False, delta=0.0, path='checkpoint.pt'):

        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path

    def __call__(self, val_loss, model):

        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
            self.save_checkpoint(val_loss, model)
            self.val_loss_min = val_loss

        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


early_stopping = EarlyStopping(patience=10, delta=0.001)


def train(epochs):
    train_loss_list = []
    val_loss_list = []
    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        model.train()
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            net_out = model(data)
            target = data.Q
            target = torch.unsqueeze(target, 1)
            loss = criterion(net_out, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            if batch_idx == 237:
                train_loss_list.append(train_loss / 238)
        model.eval()
        for data in val_loader:
            data = data.to(device)
            net_out = model(data)
            target = data.Q
            target = torch.unsqueeze(target, 1)
            loss = criterion(net_out, target)
            val_loss += loss.item()
        val_loss = val_loss / len(val_dataset)
        val_loss_list.append(str(val_loss))
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping, epoch: " + str(epoch))
            break


train(300)

model.load_state_dict(torch.load('checkpoint.pt'))
model.to(device)

target_list = []
net_out_list = []

for data in test_loader:
    data = data.to(device)
    net_out = model(data)
    net_out_list.append(str(net_out.item()))
    target = data.Q
    target = torch.unsqueeze(target, 1)
    target_list.append(str(target.item()))

MAE = mean_absolute_error(target_list, net_out_list)
RMSE = sqrt(mean_squared_error(target_list, net_out_list))
R2 = r2_score(target_list, net_out_list)

print('MAE = ', MAE, '\n', 'RMSE = ', RMSE, '\n', 'R2 = ', R2)
