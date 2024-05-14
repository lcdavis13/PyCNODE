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

# D: # data samples (rows)
# N: # species, or columns in data set

# Data processing

# Read data
inputs = pd.read_csv('data/michel-mata_drosophila_in.csv', index_col=0)
targets = pd.read_csv('data/michel-mata_drosophila_out.csv', index_col=0)

# Convert to tensors
inputs = torch.tensor(inputs.values, dtype=torch.float32)  # D x N  ( 1410 x 5747 )
targets = torch.tensor(targets.values, dtype=torch.float32) # D x N  ( 1410 x 5747 )

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
train_dataset = CompositionDataset(inputs[train_indices], targets[train_indices]) # D_train x <test+train> x N  (1128 x 2 x 5747)
test_dataset = CompositionDataset(inputs[test_indices], targets[test_indices]) # # D_test x <test+train> x N  (282 x 2 x 5747)

batch_size = min(32, len(train_indices))  # B

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

def get_batch(data_loader):
    while True:
        for inputs, targets in data_loader:
            yield inputs.to(device), targets.to(device)  # B x N, B x N  (32 x 5747, 32 x 5747)
            
train_gen = get_batch(train_loader)
test_gen = get_batch(test_loader)

dimension = inputs.shape[1] # N
print(inputs.shape)
print(targets.shape)
print(len(train_indices))
print(len(test_indices))

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
        
        self.f = nn.Linear(dimension, dimension) # (N+1) x N
        #self.ones = torch.ones(dimension, 1) # N x 1

    def forward(self, t, x):
        # dx/dt = x * (f(x) - ones . x^T . f(x))
        
        
        # x: B x N
        
        fx = self.f(x)  # [Bx]Nx(N+1) . (N+1) => [Bx]N
        
        # this should be the same as ones*x^T (summing over x), and work for batched data
        ones_xT = torch.sum(x, dim=-1)  #sum_N([Bx]N) => [B or] 1

        # ones . x^T . f(x) but batched
        # u = fx * sumx # [B or] 1 x [Bx]N => [Bx]N
        ones_xT_fx = torch.einsum('...b, ...bn -> ...bn', ones_xT, fx)
        
        ones_xT_fx = torch.einsum('...bn, ...bn -> ...bn', x, fx)
        
        d = fx - ones_xT_fx # [Bx]N - [Bx]N => [Bx]N
        
        dxdt = torch.mul(x, d) # [Bx]N .* [Bx]N => [Bx]N
        

        #print("forward")
        #print(x.shape)
        #print(fx.shape)
        #print(ones_xT.shape)
        #print(ones_xT_fx.shape)
        #print(d.shape)
        #print(dxdt.shape)
        
        return dxdt
        
        # xfx = torch.bmm(x.unsqueeze(1), fx.unsqueeze(2)).squeeze(2) # dot product x*fx, but batched
        #
        # dxdt = x * (fx - xfx)
        #
        # return dxdt


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

def loss_bc(p_i,q_i): # Bray-Curtis loss, copied from https://github.com/spxuw/DKI/blob/main/DKI.py
    # I don't understand how this is BC loss, since that is supposed to have the minimum of the two values in the numerator for each species
    # but it's the same in the repos for both cNODE papers, so I assume it's correct
    return torch.sum(torch.abs(p_i - q_i)) / torch.sum(torch.abs(p_i + q_i))  # Should be able to remove the second .abs here?


if __name__ == '__main__':
    
    viz = None
    if args.viz:
        # Lazy import inside the condition
        from visualizer import Visualizer
        viz = Visualizer()

    ii = 0

    func = ODEFunc().to(device)
    
    optimizer = torch.optim.Adam(func.parameters(), lr=0.01)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)
    
    test_loss_meter = MinimumMeter()

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        train_y0, train_y = next(train_gen)
        train_t = torch.tensor([0.0, 1.0]).to(device) # To Do: refactor to remove this time-series input, always want to evaluate at t=1
        train_pred_y = odeint(func, train_y0, train_t).to(device)
        loss = loss_bc(train_pred_y, train_y)
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                test_y0, test_y = next(test_gen)
                test_t = torch.tensor([0.0, 1.0]).to(device) # To Do: refactor to remove this time-series input, always want to evaluate at t=1
                test_pred_y = odeint(func, test_y0, test_t).to(device)
                loss = loss_bc(test_pred_y, test_y)
                test_loss_meter.update(loss.item(), itr)
                print('Iter {:06d} | Train Loss {:.6f} | Test Loss {:.6f} | Min Test Loss {:.6f} @ Iter {:06d}'.format(itr, loss_meter.val, loss.item(), test_loss_meter.min, test_loss_meter.itr))
                if args.viz:
                    viz.visualize(test_y, test_pred_y, func, ii, test_t, device)
                ii += 1

        end = time.time()
