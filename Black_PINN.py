import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(123)

class BlackScholesMertonModel1(nn.Module):
    def __init__(self):
        super(BlackScholesMertonModel1, self).__init__()
        self.bn1 = nn.BatchNorm1d(2)
        self.fc1 = nn.Linear(2, 50)
        self.act1 = nn.ReLU()
        
        self.bn2 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 50)
        self.act2 = nn.ReLU()
        
        self.bn3 = nn.BatchNorm1d(50)
        self.fc3 = nn.Linear(50, 50)
        self.act3 = nn.ReLU()
        
        self.bn4 = nn.BatchNorm1d(50)
        self.fc4 = nn.Linear(50, 50)
        self.act4 = nn.ReLU()
        
        self.bn5 = nn.BatchNorm1d(50)
        self.fc5 = nn.Linear(50, 50)
        self.act5 = nn.ReLU()
        
        self.bn6 = nn.BatchNorm1d(50)
        self.fc6 = nn.Linear(50, 50)
        self.act6 = nn.ReLU()
        
        self.bn7 = nn.BatchNorm1d(50)
        self.fc7 = nn.Linear(50, 50)
        self.act7 = nn.ReLU()
        
        self.bn8 = nn.BatchNorm1d(50)
        self.fc8 = nn.Linear(50, 50)
        self.act8 = nn.ReLU()
        
        self.bn9 = nn.BatchNorm1d(50)
        self.fc9 = nn.Linear(50, 50)
        self.act9 = nn.ReLU()
        
        self.bn10 = nn.BatchNorm1d(50)
        self.fc10 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.act1(x)
        
        x = self.bn2(x)
        x = self.fc2(x)
        x = self.act2(x)
        
        x = self.bn3(x)
        x = self.fc3(x)
        x = self.act3(x)
        
        x = self.bn4(x)
        x = self.fc4(x)
        x = self.act4(x)
        
        x = self.bn5(x)
        x = self.fc5(x)
        x = self.act5(x)
        
        x = self.bn6(x)
        x = self.fc6(x)
        x = self.act6(x)
        
        x = self.bn7(x)
        x = self.fc7(x)
        x = self.act7(x)
        
        x = self.bn8(x)
        x = self.fc8(x)
        x = self.act8(x)
        
        x = self.bn9(x)
        x = self.fc9(x)
        x = self.act9(x)
        
        x = self.bn10(x)
        x = self.fc10(x)
        
        return x

from ExplicitEu import ExplicitEu
from ImplicitEu import ImplicitEu
from ImplicitAm import ImplicitAmBer
from ImplicitAm import ImplicitAmBre

import numpy as np
import torch

# Parameters
S0 = 100
exercise_price = 100
sigma = 0.4
r = 0.03
dividend = 0.00
tau = 3
M = 500  # S
N = 600  # t
Smax = 500
is_call = True
N_b = 100
N_exp = 1000
N_f = 10000
lb = [0, 0]
ub = [500, tau]

# Mesh grid
t, S = np.meshgrid(np.linspace(0, 1, N+1), np.linspace(0, Smax, M+1))

# Option Pricing
option = ImplicitAmBer(S0, exercise_price, r, tau, sigma, Smax, M, N, is_call)
option.price()
option_fde_prices = option.grid

# Initialize data function
def initialize_data(N_b, N_exp, N_f, lb, ub, exercise_price, tau):
	# Data for collocation
	stock_price_collocation = torch.randint(low=0, high=ub[0] + 1, size=(N_f, 1)).type(torch.FloatTensor)
	time_collocation = (torch.randint(low=0, high=100 * ub[1] + 1, size=(N_f, 1)) / 100).type(torch.FloatTensor)
	
	stock_price_mean = torch.mean(stock_price_collocation)
	# Calculating mean and standard deviation
	stock_price_std = torch.std(stock_price_collocation)
	time_mean = torch.mean(time_collocation)
	time_std = torch.std(time_collocation)
	
	# Concatenating time and stock price data
	X_f = torch.cat((time_collocation, stock_price_collocation), 1)
	
	# Normalizing time and stock price data
	time_collocation = (time_collocation - time_mean) / time_std
	stock_price_collocation = (stock_price_collocation - stock_price_mean) / stock_price_std
	X_f_norm = torch.cat((time_collocation, stock_price_collocation), 1)
	
	# Data for boundary
	time_boundary = (torch.randint(low=1, high=100 * ub[1] + 1, size=(N_b, 1)) / 100).type(torch.FloatTensor)
	X_b = torch.cat((time_boundary, 0 * time_boundary), 1)
	
	# Data for initial time
	stock_price_exp = torch.randint(low=0, high=ub[0] + 1, size=(N_exp, 1)).type(torch.FloatTensor)
	option_price_exp = stock_price_exp - exercise_price
	u_exp = torch.Tensor([[max(instance, 0)] for instance in option_price_exp]).type(torch.FloatTensor)
	X_exp = torch.cat((0 * stock_price_exp + tau, stock_price_exp), 1)
	
	# Return values from the function
	return X_f, X_f_norm, X_b, X_exp, u_exp

# Initialize data
X_f, X_f_norm, X_b, X_exp, u_exp = initialize_data(N_b, N_exp, N_f, lb, ub, exercise_price, tau)

# Collocation points
u_collocation = []
for instance in X_f:
	time = instance[0]
	stock_price = instance[1]
	stock_price = int(stock_price.item())
	time = int(time.item() * 200)
	u_collocation_val = torch.Tensor(
	    np.array([(np.round(option_fde_prices[stock_price, time], 3))])
	).type(torch.FloatTensor)
	
	u_collocation.append(u_collocation_val)
	
u_collocation = np.array(u_collocation)
u_collocation = np.reshape(u_collocation, (-1, 1))
u_collocation = torch.Tensor(u_collocation).type(torch.FloatTensor)
print(u_collocation.shape)

f_collocation = torch.zeros(N_f, 1)
u_boundary = torch.zeros(N_b, 1)

# Initializing the PDE solver model
model1 = BlackScholesMertonModel1()

# Original weight initialization didn't change the weights.
# Initialize the weights
for m in model1.modules():
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.constant_(m.bias, 0)

# Perform backprop
MAX_EPOCHS_1 = int(710)
LRATE = 8e-3

# Use Adam for training
optimizer = torch.optim.Adam(model1.parameters(), lr=LRATE)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 210, gamma=0.5, last_epoch=-1, verbose=False)

X_f.requires_grad = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1.to(device)  # Send model to GPU if available

# Send data to GPU if available
X_f = X_f.to(device)
X_b = X_b.to(device)
X_exp = X_exp.to(device)
u_boundary = u_boundary.to(device)
u_exp = u_exp.to(device)
u_collocation = u_collocation.to(device)
f_collocation = f_collocation.to(device)
loss_history_function_1 = []
loss_history_f_1 = []
loss_history_boundary_1 = []
loss_history_exp_1 = []

print("Learning Rate for this Round of Training:", LRATE)
print("First Round of Training")

for epoch in range(MAX_EPOCHS_1):
    # Boundary loss
    with torch.no_grad():
        rand_index = torch.randperm(n=len(X_b), device=device)
        X_b_shuffle = X_b[rand_index]
        u_boundary_shuffle = u_boundary[rand_index]
        u_b_pred = model1(X_b_shuffle)
        mse_u_b = torch.nn.MSELoss()(u_b_pred, u_boundary_shuffle)
    
    # Expiration time loss
    with torch.no_grad():
        rand_index = torch.randperm(n=len(X_exp), device=device)
        X_exp_shuffle = X_exp[rand_index]
        u_exp_shuffle = u_exp[rand_index]
        u_exp_pred = model1(X_exp_shuffle)
        mse_u_exp = torch.nn.MSELoss()(u_exp_pred, u_exp_shuffle)
    
    # Collocation loss
    rand_index = torch.randperm(n=len(X_f), device=device)
    X_f_shuffle = X_f[rand_index]
    f_collocation_shuffle = f_collocation[rand_index]
    u_collocation_shuffle = u_collocation[rand_index]
    u_pred = model1(X_f_shuffle)
    
    # Compute gradients
    stock_price = X_f_shuffle[:, 1:2]
    u_pred_first_partials = torch.autograd.grad(u_pred.sum(), X_f_shuffle, create_graph=True, allow_unused=True)[0]
    u_pred_dt = u_pred_first_partials[:, 0:1]
    u_pred_ds = u_pred_first_partials[:, 1:2]
    u_pred_second_partials = torch.autograd.grad(u_pred_ds.sum(), X_f_shuffle, create_graph=True, allow_unused=True)[0]
	    u_pred_dss = u_pred_second_partials[:, 1:2]
    
    # Compute f_pred
    f_pred = u_pred_dt + (0.5 * (sigma ** 2) * (stock_price ** 2) * u_pred_dss) + \
             ((r - dividend) * stock_price * u_pred_ds) - (r * u_pred)
    f_true = f_collocation_shuffle
    mse_f = 100 * torch.nn.MSELoss()(f_pred, f_true)
    
    # Total loss
    loss = mse_f + mse_u_exp + mse_u_b
    mse_function = torch.nn.MSELoss()(u_pred, u_collocation_shuffle).detach()
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Record loss history
    loss_history_f_1.append(mse_f / 100)
    loss_history_boundary_1.append(mse_u_b)
    loss_history_exp_1.append(mse_u_exp)
    loss_history_function_1.append(mse_function)
    
    # Print status every 10 epochs
    if (epoch % 10) == 0:
        print("---------------")
        print("Epoch:", epoch)
        print(f"Loss Residual:\t{loss_history_f_1[-1]:.4f}")
        print(f"Loss Boundary:\t{loss_history_boundary_1[-1]:.4f}")
        print(f"Loss Expiration:\t{loss_history_exp_1[-1]:.4f}")
        print(f"Loss Function:\t{loss_history_function_1[-1]:.4f}")
        print("----------------------------------------------------------")

# Parameters for the second round of training
MAX_EPOCHS_2 = int(4700)
LRATE = 1e-3
optimizer = torch.optim.Adam(model1.parameters(), lr=LRATE)

# Initialize loss histories for the second round of training
loss_history_function_2 = []
loss_history_f_2 = []
loss_history_boundary_2 = []
loss_history_exp_2 = []

print("Learning Rate for this Round of Training:", LRATE)
print("Second Round of Training")

for epoch in range(MAX_EPOCHS_2):
    # Boundary loss
    with torch.no_grad():
        rand_index = torch.randperm(n=len(X_b), device=device)
        X_b_shuffle = X_b[rand_index]
        u_boundary_shuffle = u_boundary[rand_index]
        u_b_pred = model1(X_b_shuffle)
        mse_u_b = torch.nn.MSELoss()(u_b_pred, u_boundary_shuffle)
    
    # Expiration time loss
    with torch.no_grad():
        rand_index = torch.randperm(n=len(X_exp), device=device)
        X_exp_shuffle = X_exp[rand_index]
        u_exp_shuffle = u_exp[rand_index]
        u_exp_pred = model1(X_exp_shuffle)
        mse_u_exp = torch.nn.MSELoss()(u_exp_pred, u_exp_shuffle)
    
    # Collocation loss
    rand_index = torch.randperm(n=len(X_f), device=device)
    X_f_shuffle = X_f[rand_index]
    f_collocation_shuffle = f_collocation[rand_index]
    u_collocation_shuffle = u_collocation[rand_index]
    u_pred = model1(X_f_shuffle)
    
    # Compute gradients
    stock_price = X_f_shuffle[:, 1:2]
    u_pred_first_partials = torch.autograd.grad(u_pred.sum(), X_f_shuffle, create_graph=True, allow_unused=True)[0]
    u_pred_dt = u_pred_first_partials[:, 0:1]
    u_pred_ds = u_pred_first_partials[:, 1:2]
    u_pred_second_partials = torch.autograd.grad(u_pred_ds.sum(), X_f_shuffle, create_graph=True, allow_unused=True)[0]
    u_pred_dss = u_pred_second_partials[:, 1:2]
    
    # Compute f_pred
    f_pred = u_pred_dt + (0.5 * (sigma ** 2) * (stock_price ** 2) * u_pred_dss) + \
             ((r - dividend) * stock_price * u_pred_ds) - (r * u_pred)
    f_true = f_collocation_shuffle
    mse_f = 100 * torch.nn.MSELoss()(f_pred, f_true)
    
    # Total loss
    loss = mse_f + mse_u_exp + mse_u_b
    mse_function = torch.nn.MSELoss()(u_pred, u_collocation_shuffle).detach()
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Record loss history
    loss_history_f_2.append(mse_f / 100)
    loss_history_boundary_2.append(mse_u_b)
    loss_history_exp_2.append(mse_u_exp)
    loss_history_function_2.append(mse_function)
    
    # Print status every 10 epochs
    if (epoch % 10) == 0:
        print("---------------")
        print("Epoch:", epoch)
        print(f"Loss Residual:\t{loss_history_f_2[-1]:.4f}")
        print(f"Loss Boundary:\t{loss_history_boundary_2[-1]:.4f}")
        print(f"Loss Expiration:\t{loss_history_exp_2[-1]:.4f}")
        print(f"Loss Function:\t{loss_history_function_2[-1]:.4f}")
        print("----------------------------------------------------------")

        
