import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from GCN.data_processing import Data_Loader
from GCN.graphPyTorch import get_graph_data
from GCN.sgcn_lstm_wA_pytorch import SGCN_LSTM

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

parser = argparse.ArgumentParser(description='PyTorch ST-GCN Trainer')
parser.add_argument('--ex', type=str, required=True, help='Exercise name (e.g., Kimore_ex5)')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--epoch', type=int, default=1000, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
args = parser.parse_args()

# Load and split dataset
data_loader = Data_Loader(args.ex)
train_x, test_x, train_y, test_y = train_test_split(
    data_loader.scaled_x, data_loader.scaled_y, test_size=0.2, random_state=42
)

# Convert to torch tensors
train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)
test_x = torch.tensor(test_x, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.float32)

# Get graph data
num_nodes = len(data_loader.body_part)
AD, AD2, bias_mat_1, bias_mat_2 = get_graph_data(num_nodes)

# Initialize model
model = SGCN_LSTM(AD, AD2, bias_mat_1, bias_mat_2, num_joints=num_nodes)

# Train
model.train_model(train_x, train_y, lr=args.lr, epochs=args.epoch, batch_size=args.batch_size, ex_path=args.ex)

# Predict
y_pred = model.predict(test_x).detach().cpu().numpy()
test_y = test_y.detach().cpu().numpy()
y_pred = data_loader.sc2.inverse_transform(y_pred)
test_y = data_loader.sc2.inverse_transform(test_y)

# Metrics
mae = mean_absolute_error(test_y, y_pred)
mse = mean_squared_error(test_y, y_pred)
mape = mean_absolute_percentage_error(test_y, y_pred)
rms = np.sqrt(mse)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rms)
print("MAPE:", mape)
