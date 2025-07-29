import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class SGCN_LSTM(nn.Module):
    def __init__(self, AD, AD2, bias_mat_1, bias_mat_2, num_joints):
        super(SGCN_LSTM, self).__init__()
        self.AD = AD
        self.AD2 = AD2
        self.bias_mat_1 = bias_mat_1
        self.bias_mat_2 = bias_mat_2
        self.num_joints = num_joints

        self.temporal1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(9, 1), padding=(4, 0))
        self.gcn_conv = nn.Conv2d(64 + 3, 64, kernel_size=(1, 1))

        self.temporal2 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=(9, 1), padding=(4, 0)),
            nn.Dropout(0.25),
            nn.Conv2d(16, 16, kernel_size=(15, 1), padding=(7, 0)),
            nn.Dropout(0.25),
            nn.Conv2d(16, 16, kernel_size=(21, 1), padding=(10, 0)),
            nn.Dropout(0.25)
        )

        self.lstm = nn.LSTM(input_size=16 * num_joints, hidden_size=80, num_layers=3, batch_first=True, dropout=0.25)
        self.linear = nn.Linear(80, 1)

    def forward(self, x):
        #pdb.set_trace()
        # x: [B, T, V, C] -> [B, C, T, V]
        pdb.set_trace()
        x = x.permute(0, 3, 1, 2)
        residual = x

        x1 = F.relu(self.temporal1(x))
        x = torch.cat([x, x1], dim=1)

        x = F.relu(self.gcn_conv(x))
        x = self.temporal2(x)

        # x: [B, C, T, V] -> [B, T, C*V]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1)

        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last output
        out = self.linear(out)
        return out

    def train_model(self, train_x, train_y, lr=0.0001, epochs=200, batch_size=10):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.HuberLoss()

        self.train()
        for epoch in range(epochs):
            permutation = torch.randperm(train_x.size(0))
            losses = []
            for i in range(0, train_x.size(0), batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = train_x[indices], train_y[indices]

                optimizer.zero_grad()
                output = self.forward(batch_x)
                loss = criterion(output.squeeze(), batch_y.squeeze())
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(losses):.4f}")

    def predict(self, test_x):
        self.eval()
        with torch.no_grad():
            return self.forward(test_x)
