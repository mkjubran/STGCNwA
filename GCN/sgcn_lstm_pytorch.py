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

        self.temporal3C = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(9, 1), padding=(4, 0))
        self.gcn_conv67C = nn.Conv2d(64 + 3, 64, kernel_size=(1, 1))

        self.temporal48C = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(9, 1), padding=(4, 0))
        self.gcn_conv112C = nn.Conv2d(64 + 48, 64, kernel_size=(1, 1))

        self.lstm = nn.LSTM(input_size=16 * num_joints, hidden_size=80, num_layers=3, batch_first=True, dropout=0.25)
        self.linear = nn.Linear(80, 1)
        
        self.ConvLSTM = ConvLSTM(input_dim=64, hidden_dim=self.num_joints, kernel_size=(1, 1))

        self.conv1 = nn.Conv2d(in_channels=128, out_channels=16, kernel_size=(9,1), padding=(4,0))
        self.dropout1 = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(15, 1), padding=(7, 0))
        self.dropout2 = nn.Dropout(p=0.25)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(21, 1), padding=(10, 0))
        self.dropout3 = nn.Dropout(p=0.25)

        self.lstm1 = nn.LSTM(input_size=48 * num_joints, hidden_size=80, batch_first=True, dropout=0.25)
        self.lstm2 = nn.LSTM(input_size=80, hidden_size=40, batch_first=True, dropout=0.25)
        self.lstm3 = nn.LSTM(input_size=40, hidden_size=40, batch_first=True, dropout=0.25)
        self.lstm4 = nn.LSTM(input_size=40, hidden_size=80, batch_first=True, dropout=0.25)

        self.dropout3 = nn.Dropout(p=0.25)
        self.dropout4 = nn.Dropout(p=0.25)
        self.dropout5 = nn.Dropout(p=0.25)

        self.fc = nn.Linear(80, 1)

    def sgcn(self, x):
        # x: [B, T, J, C] -> [B, C, T, J]
        x = x.permute(0, 3, 1, 2)
        residual = x

        """Temporal convolution across T in [B, C, T, J]"""
        if x.shape[1] == 3: #C=3
           k1 = F.relu(self.temporal3C(x))
        else: #C=48
           k1 = F.relu(self.temporal48C(x))

        k = torch.cat([x, k1], dim=1)

        """Graph Convolution"""
        """first hop localization - neighbour joints (bias_mat_1)"""
        if k.shape[1] == 67: #C=64+3
           x1 = F.relu(self.gcn_conv67C(k))
        else: #C=64+48
           x1 = F.relu(self.gcn_conv112C(k))

        # x1: [B, C, T, J] -> [B, T, C, J]
        x1 = x1.permute(0, 2, 1, 3)
        # expand_x1: [B, C, T, J] -> [B, T, C, 1, J]
        expand_x1 = x1.unsqueeze(3)

        # Process the input sequence [B, T, C, 1, J]  with ConvLSTM to extract
        # temporal dynamics in spatial features
        f_1 = self.ConvLSTM(expand_x1)

        # f_1: [B, T, J, 1, J] -> [B, T, J, J]
        f_1 = f_1[:,:,:,0,:]
        logits = f_1

        # Compute normalized attention coefficients using LeakyReLU and bias mask,
        # then apply softmax across the last dimension
        coefs = F.softmax(F.leaky_relu(logits) + self.bias_mat_1, dim=-1)

        # x1: [B, T, C, J] -> [B, T, J, C]
        x1 = x1.permute(0, 1, 3, 2)

        # Perform attention-weighted aggregation of features across joints:
        # Multiply attention coefficients (coefs) with input features (x1)
        # Shape transformation: [B, T, J, C] <- einsum('ntvw,ntwc->ntvc')
        # coefs [B, T, J, J] , x1 [B, T, J, C] --> gcn_x1 [B, T, J, C]
        gcn_x1 = torch.einsum('ntvw,ntwc->ntvc', coefs, x1)

        """second hop localization - neighbour of neighbour joints (bias_mat_2)"""
        if k.shape[1] == 67: #C=64+3
            y1 = F.relu(self.gcn_conv67C(k))
        else: #C=64+48
            y1 = F.relu(self.gcn_conv112C(k))

        y1 = y1.permute(0, 2, 1, 3)
        expand_y1 = y1.unsqueeze(3)
        f_2 = self.ConvLSTM(expand_y1)
        f_2 = f_2[:,:,:,0,:]
        logits = f_1
        coefs = F.softmax(F.leaky_relu(logits) + self.bias_mat_2, dim=-1)
        y1 = y1.permute(0, 1, 3, 2)
        gcn_y1 = torch.einsum('ntvw,ntwc->ntvc', coefs, y1)

        # concat attention-weighted aggregation of features across joints from
        # first and second hop localizations
        gcn_1 = torch.cat([gcn_x1, gcn_y1], dim=-1)

        """Temporal convolution"""
        # gcn_1: gcn_x1 [B, T, J, C] -> [BxT, J, C]
        gcn_1 = gcn_1.view(-1, gcn_1.shape[2], gcn_1.shape[3])

        # gcn_1: gcn_x1 [BxT, J, C] -> [BxT, C, J, 1]
        gcn_1 = gcn_1.permute(0, 2, 1).unsqueeze(3)

        # z1 [BxT , 16, J, 1]
        z1 = self.dropout1(F.relu(self.conv1(gcn_1)))

        # z2 [BxT , 16, J, 1]
        z2 = self.dropout2(F.relu(self.conv2(z1)))

        # z3 [BxT , 16, J, 1]
        z3 = self.dropout3(F.relu(self.conv3(z2)))

        # z_concat [BxT ,48 , J, 1]
        z_concat = torch.cat([z1, z2, z3], dim=1)

        # z [B ,T, 48 , J]
        z = z_concat.reshape(x.shape[0],x.shape[2], 48, x.shape[3])

        # z [B ,T, 48 , J] -> [B ,T, J , 48]
        z = z.permute(0, 1, 3, 2)

        return z


    def forward(self, x):
        xx = self.sgcn(x)
        yy = self.sgcn(xx)
        yy = yy + xx
        zz = self.sgcn(yy)
        zz = zz + yy
  
        # LSTM
        zz = zz.reshape(zz.shape[0], zz.shape[1], -1)  # [B, T, J*48]
        zz, _ = self.lstm1(zz)
        zz = self.dropout1(zz)
        zz, _ = self.lstm2(zz)
        zz = self.dropout2(zz)
        zz, _ = self.lstm3(zz)
        zz = self.dropout3(zz)
        zz, _ = self.lstm4(zz)
        zz = zz[:, -1, :]  # take last time step

        out = self.fc(zz)
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
