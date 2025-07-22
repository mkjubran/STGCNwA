import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from GCN.ConvLSTM import ConvLSTM

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
        
        self.ConvLSTM = ConvLSTM(input_dim=64, hidden_dim=self.num_joints, kernel_size=(1, 1))

        self.conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(9, 1), padding=(4, 0))
        self.dropout1 = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(15, 1), padding=(7, 0))
        self.dropout2 = nn.Dropout(p=0.25)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(21, 1), padding=(10, 0))
        self.dropout3 = nn.Dropout(p=0.25)



    def forward(self, x):
        # x: [B, T, V, C] -> [B, C, T, V]
        x = x.permute(0, 3, 1, 2)
        residual = x

        """Temporal convolution"""
        k1 = F.relu(self.temporal1(x))
        k = torch.cat([x, k1], dim=1)

        """Graph Convolution"""
        
        """first hop localization"""
        x1 = F.relu(self.gcn_conv(k))
        x1 = x1.permute(0, 2, 1, 3)
        expand_x1 = x1.unsqueeze(3)
        f_1 = self.ConvLSTM(expand_x1)
        f_1 = f_1[:,:,:,0,:]
        logits = f_1
        coefs = F.softmax(F.leaky_relu(logits) + self.bias_mat_1, dim=-1)
        x1 = x1.permute(0, 1, 3, 2)
        gcn_x1 = torch.einsum('ntvw,ntwc->ntvc', coefs, x1)

        """second hop localization"""
        y1 = F.relu(self.gcn_conv(k))
        y1 = y1.permute(0, 2, 1, 3)
        expand_y1 = y1.unsqueeze(3)
        f_2 = self.ConvLSTM(expand_y1)
        f_2 = f_2[:,:,:,0,:]
        logits = f_1 
        coefs = F.softmax(F.leaky_relu(logits) + self.bias_mat_2, dim=-1)
        y1 = y1.permute(0, 1, 3, 2)
        gcn_y1 = torch.einsum('ntvw,ntwc->ntvc', coefs, y1)

        gcn_1 = torch.cat([gcn_x1, gcn_y1], dim=-1)
        pdb.set_trace()
        
        """Temporal convolution"""
        z1 = self.dropout1(F.relu(self.conv1(gcn_1)))
        z2 = self.dropout2(F.relu(self.conv2(z1)))
        z3 = self.dropout3(F.relu(self.conv3(z2)))
        z = torch.cat([z1, z2, z3], dim=-1)
        '''
        """Temporal convolution"""
        x = self.temporal2(x)

        # x: [B, C, T, V] -> [B, T, C*V]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1)

        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last output
        out = self.linear(out)
        return out
        '''
        return z

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
