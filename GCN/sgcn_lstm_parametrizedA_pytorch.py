import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from GCN.ConvLSTM import ConvLSTM

class ParameterizedAttention(nn.Module):
    """Parameterized attention mechanism with learnable parameters"""
    def __init__(self, feature_dim, num_joints, hidden_dim=64):
        super(ParameterizedAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        
        # Learnable attention parameters
        self.W_q = nn.Linear(feature_dim, hidden_dim)  # Query projection
        self.W_k = nn.Linear(feature_dim, hidden_dim)  # Key projection
        self.W_v = nn.Linear(feature_dim, feature_dim)  # Value projection
        
        # Additional learnable parameters for attention scoring
        self.attention_weights = nn.Parameter(torch.randn(hidden_dim, num_joints))
        self.bias = nn.Parameter(torch.zeros(num_joints))
        
        # Normalization
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, bias_mat):
        """
        Args:
            x: [B, T, J, C] - input features
            bias_mat: [J, J] - adjacency bias matrix
        Returns:
            attended_features: [B, T, J, C]
        """
        B, T, J, C = x.shape
        
        # Project to query, key, value
        q = self.W_q(x)  # [B, T, J, hidden_dim]
        k = self.W_k(x)  # [B, T, J, hidden_dim]
        v = self.W_v(x)  # [B, T, J, C]
        
        # Compute attention scores using parameterized approach
        # Method 1: Dot-product attention with learnable weights
        attention_scores = torch.einsum('btjh,hk->btjk', q, self.attention_weights)  # [B, T, J, J]
        attention_scores = attention_scores + self.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        # Method 2: Additional scaled dot-product attention
        scale = (self.hidden_dim ** -0.5)
        dot_product_scores = torch.einsum('btjh,btkh->btjk', q, k) * scale  # [B, T, J, J]
        
        # Combine both attention mechanisms
        combined_scores = attention_scores + dot_product_scores
        
        # Apply bias matrix (adjacency information)
        combined_scores = combined_scores + bias_mat.unsqueeze(0).unsqueeze(0)
        
        # Apply softmax to get attention coefficients
        attention_coefs = F.softmax(combined_scores, dim=-1)  # [B, T, J, J]
        attention_coefs = self.dropout(attention_coefs)
        
        # Apply attention to values
        attended_features = torch.einsum('btjk,btkc->btjc', attention_coefs, v)  # [B, T, J, C]
        
        # Residual connection and layer normalization
        attended_features = self.layer_norm(attended_features + x)
        
        return attended_features

class SGCN_LSTM(nn.Module):
    def __init__(self, AD, AD2, bias_mat_1, bias_mat_2, num_joints):
        super(SGCN_LSTM, self).__init__()
        self.AD = AD
        self.AD2 = AD2
        self.bias_mat_1 = bias_mat_1
        self.bias_mat_2 = bias_mat_2
        self.num_joints = num_joints
        
        # Temporal convolution layers
        self.temporal3C = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(9, 1), padding=(4, 0))
        self.gcn_conv67C = nn.Conv2d(64 + 3, 64, kernel_size=(1, 1))
        self.temporal48C = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(9, 1), padding=(4, 0))
        self.gcn_conv112C = nn.Conv2d(64 + 48, 64, kernel_size=(1, 1))
        
        # Parameterized attention mechanisms
        self.attention_1 = ParameterizedAttention(feature_dim=64, num_joints=num_joints, hidden_dim=32)
        self.attention_2 = ParameterizedAttention(feature_dim=64, num_joints=num_joints, hidden_dim=32)
        
        # ConvLSTM (keeping original for comparison/fallback)
        self.ConvLSTM = ConvLSTM(input_dim=64, hidden_dim=self.num_joints, kernel_size=(1, 1))
        
        # Temporal processing layers
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=16, kernel_size=(9,1), padding=(4,0))
        self.dropout1 = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(15, 1), padding=(7, 0))
        self.dropout2 = nn.Dropout(p=0.25)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(21, 1), padding=(10, 0))
        self.dropout3 = nn.Dropout(p=0.25)
        self.dropout4 = nn.Dropout(p=0.25)
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=48 * num_joints, hidden_size=80, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=80, hidden_size=40, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=40, hidden_size=40, batch_first=True)
        self.lstm4 = nn.LSTM(input_size=40, hidden_size=80, batch_first=True)
        
        # Final layers
        self.fc = nn.Linear(80, 1)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
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
        
        """Graph Convolution with Parameterized Attention"""
        """First hop localization - neighbour joints (bias_mat_1)"""
        if k.shape[1] == 67: #C=64+3
           x1 = F.relu(self.gcn_conv67C(k))
        else: #C=64+48
           x1 = F.relu(self.gcn_conv112C(k))
        
        # x1: [B, C, T, J] -> [B, T, J, C]
        x1 = x1.permute(0, 2, 3, 1)
        
        # Apply parameterized attention for first hop
        gcn_x1 = self.attention_1(x1, self.bias_mat_1)  # [B, T, J, C]
        
        """Second hop localization - neighbour of neighbour joints (bias_mat_2)"""
        if k.shape[1] == 67: #C=64+3
            y1 = F.relu(self.gcn_conv67C(k))
        else: #C=64+48
            y1 = F.relu(self.gcn_conv112C(k))
        
        # y1: [B, C, T, J] -> [B, T, J, C]
        y1 = y1.permute(0, 2, 3, 1)
        
        # Apply parameterized attention for second hop
        gcn_y1 = self.attention_2(y1, self.bias_mat_2)  # [B, T, J, C]
        
        # Concatenate attention-weighted aggregation of features across joints from
        # first and second hop localizations
        gcn_1 = torch.cat([gcn_x1, gcn_y1], dim=-1)  # [B, T, J, 2*C]
        
        """Temporal convolution"""
        # gcn_1: [B, T, J, 2*C] -> [B*T, J, 2*C]
        gcn_1 = gcn_1.view(-1, gcn_1.shape[2], gcn_1.shape[3])
        # gcn_1: [B*T, J, 2*C] -> [B*T, 2*C, J, 1]
        gcn_1 = gcn_1.permute(0, 2, 1).unsqueeze(3)
        
        # Temporal convolution layers
        z1 = self.dropout1(F.relu(self.conv1(gcn_1)))  # [B*T, 16, J, 1]
        z2 = self.dropout2(F.relu(self.conv2(z1)))     # [B*T, 16, J, 1]
        z3 = self.dropout3(F.relu(self.conv3(z2)))     # [B*T, 16, J, 1]
        
        # Concatenate temporal features
        z_concat = torch.cat([z1, z2, z3], dim=1)  # [B*T, 48, J, 1]
        
        # Reshape back to [B, T, J, 48]
        z = z_concat.reshape(x.shape[0], x.shape[2], 48, x.shape[3])
        z = z.permute(0, 1, 3, 2)  # [B, T, J, 48]
        
        return z
    
    def forward(self, x):
        xx = self.sgcn(x)
        yy = self.sgcn(xx)
        yy = yy + xx  # Residual connection
        zz = self.sgcn(yy)
        zz = zz + yy  # Residual connection
  
        # LSTM processing
        zz = zz.reshape(zz.shape[0], zz.shape[1], -1)  # [B, T, J*48]
        zz, _ = self.lstm1(zz)
        zz = self.dropout1(zz)
        zz, _ = self.lstm2(zz)
        zz = self.dropout2(zz)
        zz, _ = self.lstm3(zz)
        zz = self.dropout3(zz)
        zz, _ = self.lstm4(zz)
        zz = self.dropout4(zz)
        
        # Take last time step and apply final linear layer
        zz = zz[:, -1, :]  # [B, 80]
        out = self.fc(zz)   # [B, 1]
        
        return out
    
    def train_model(self, train_x, train_y, lr=0.0001, epochs=200, batch_size=10):
        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)
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
        test_x = test_x.to(self.device)
        with torch.no_grad():
            return self.forward(test_x)
