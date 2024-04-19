import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split


#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=200000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

args.viz = False
args.adjoint = True




if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
print(device)


# Data processing

# Read data
inputs = pd.read_csv('data/michel-mata_drosophila_in.csv', index_col=0)
targets = pd.read_csv('data/michel-mata_drosophila_out.csv', index_col=0)

# Convert to tensors
inputs = torch.tensor(inputs.values, dtype=torch.float32)
targets = torch.tensor(targets.values, dtype=torch.float32)

class CompositionDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Split indices
train_indices, test_indices = train_test_split(list(range(len(inputs))), test_size=0.2, random_state=42)

# Create datasets
train_dataset = CompositionDataset(inputs[train_indices], targets[train_indices])
test_dataset = CompositionDataset(inputs[test_indices], targets[test_indices])

batch_size = 32  # You can adjust this based on your specific needs

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

def get_batch(data_loader):
    while True:
        for inputs, targets in data_loader:
            yield inputs.to(device), targets.to(device)
            
train_gen = get_batch(train_loader)
test_gen = get_batch(test_loader)

# x,y = next(train_gen)
#
# print(x.shape)
# print(y.shape)
# print(x)
# print(y)

dimension = inputs.shape[1]


#
# true_y0 = torch.tensor([[2., 0.]]).to(device)
# t = torch.linspace(0., 25., args.data_size).to(device)
# true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)
#
# class Lambda(nn.Module):
#
#     def forward(self, t, y):
#         return torch.mm(y**3, true_A)
#
#
# with torch.no_grad():
#     true_y = odeint(Lambda(), true_y0, t, method='dopri5')
#
#
# def get_batch():
#     s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
#     batch_y0 = true_y[s]  # (M, D)
#     batch_t = t[:args.batch_time]  # (T)
#     batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
#     return batch_y0.to(device), batch_t.to(device), batch_y.to(device)
#




class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()
        
        self.f = nn.Linear(dimension, dimension)

    def forward(self, t, x): # dx/dt = x * (f(x) - x.f(x))
        fx = self.f(x)
        xfx = torch.bmm(x.unsqueeze(1), fx.unsqueeze(2)).squeeze(2) # dot product x*fx, but batched
        
        dxdt = x * (fx - xfx)
        
        return dxdt


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0
        self.min = 1e6
        self.max = -1e6

    def update(self, val):
        if self.val is None:
            self.avg = val
            self.min = val
            self.max = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
            self.min = min(self.min, val)
            self.max = max(self.max, val)
        self.val = val
        

class MinimumMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.min = 1e6
        self.itr = -1

    def update(self, val, itr):
        if val < self.min:
            self.min = val
            self.itr = itr


if __name__ == '__main__':
    
    viz = None
    if args.viz:
        # Lazy import inside the condition
        from visualizer import Visualizer
        viz = Visualizer()

    ii = 0

    func = ODEFunc().to(device)
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)
    
    test_loss_meter = MinimumMeter()

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_y = next(train_gen)
        batch_t = torch.tensor([0.0, 1.0]).to(device) # To Do: refactor to remove this time-series input, always want to evaluate at t=1
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                true_y0, true_y = next(test_gen)
                t = torch.tensor([0.0, 1.0]).to(device) # To Do: refactor to remove this time-series input, always want to evaluate at t=1
                pred_y = odeint(func, true_y0, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                test_loss_meter.update(loss.item(), itr)
                print('Iter {:06d} | Train Loss {:.6f} | Test Loss {:.6f} | Min Test Loss {:.6f} @ Iter {:06d}'.format(itr, loss_meter.val, loss.item(), test_loss_meter.min, test_loss_meter.itr))
                if args.viz:
                    viz.visualize(true_y, pred_y, func, ii, t, device)
                ii += 1

        end = time.time()
