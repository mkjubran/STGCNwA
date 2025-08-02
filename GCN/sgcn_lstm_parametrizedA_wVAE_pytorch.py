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
            attention_scores: [B, T, J, J] - attention scores for each joint
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
        
        return attended_features, attention_coefs

# ===== NEW VAE COMPONENTS =====

class MotionEncoder(nn.Module):
    """VAE Encoder for motion sequences"""
    def __init__(self, input_dim, hidden_dim=256, latent_dim=128, num_joints=17):
        super(MotionEncoder, self).__init__()
        self.num_joints = num_joints
        self.input_dim = input_dim
        
        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Mean and variance layers
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        self.dropout = nn.Dropout(0.2)
        
    def _initialize_vae(self, input_tensor):
        """Initialize VAE with correct input dimensions"""
        if self.use_vae_augmentation and not self.vae_initialized:
            B, T, J, C = input_tensor.shape
            input_shape = (T, J, C)
            self.motion_vae = MotionVAE(
                input_shape=input_shape, 
                latent_dim=self.latent_dim, 
                num_joints=J
            ).to(self.device)
            self.vae_initialized = True
            print(f"VAE initialized with input shape: {input_shape}")
    
    def forward(self, x):
        """
        Args:
            x: [B, T, J, C] - input motion sequence
        Returns:
            mu: [B, latent_dim] - mean of latent distribution
            logvar: [B, latent_dim] - log variance of latent distribution
        """
        # Flatten motion sequence
        B, T, J, C = x.shape
        x_flat = x.view(B, -1)  # [B, T*J*C]
        
        # Encoder forward pass
        h1 = F.relu(self.fc1(x_flat))
        h1 = self.dropout(h1)
        h2 = F.relu(self.fc2(h1))
        h2 = self.dropout(h2)
        
        # Get mean and log variance
        mu = self.fc_mu(h2)
        logvar = self.fc_logvar(h2)
        
        return mu, logvar

class MotionDecoder(nn.Module):
    """VAE Decoder for motion sequences with joint-specific attention weighting"""
    def __init__(self, latent_dim=128, hidden_dim=256, output_shape=(50, 17, 3), num_joints=17):
        super(MotionDecoder, self).__init__()
        self.output_shape = output_shape  # (T, J, C)
        self.num_joints = num_joints
        output_dim = np.prod(output_shape)
        
        # Decoder layers
        self.fc1 = nn.Linear(latent_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Joint-specific variation weights
        self.joint_weight_generator = nn.Linear(latent_dim, num_joints)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, z, joint_attention_scores=None):
        """
        Args:
            z: [B, latent_dim] - latent code
            joint_attention_scores: [B, T, J, J] - attention scores from main model
        Returns:
            reconstructed_motion: [B, T, J, C]
            joint_weights: [B, J] - weights for joint-specific variation
        """
        B = z.shape[0]
        
        # Generate joint-specific weights
        joint_weights = torch.sigmoid(self.joint_weight_generator(z))  # [B, J]
        
        # Decoder forward pass
        h1 = F.relu(self.fc1(z))
        h1 = self.dropout(h1)
        h2 = F.relu(self.fc2(h1))
        h2 = self.dropout(h2)
        output = self.fc3(h2)
        
        # Reshape to motion sequence
        reconstructed_motion = output.view(B, *self.output_shape)  # [B, T, J, C]
        
        # Apply joint-specific weighting if attention scores are provided
        if joint_attention_scores is not None:
            # Use attention scores to modulate joint variation
            # Average attention scores across time and target joints
            joint_importance = torch.mean(joint_attention_scores, dim=(1, 3))  # [B, J]
            
            # Combine with learned joint weights
            combined_weights = joint_weights * joint_importance
            
            # Apply weights to motion (broadcasting across time and features)
            reconstructed_motion = reconstructed_motion * combined_weights.unsqueeze(1).unsqueeze(3)
        
        return reconstructed_motion, joint_weights

class FrameDiscriminator(nn.Module):
    """Frame-wise discriminator (Dis^f)"""
    def __init__(self, input_dim, hidden_dim=128):
        super(FrameDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, T, J, C] - motion sequence
        Returns:
            predictions: [B, T] - real/fake predictions for each frame
        """
        B, T, J, C = x.shape
        x_flat = x.view(B, T, -1)  # [B, T, J*C]
        
        predictions = []
        for t in range(T):
            frame_pred = self.layers(x_flat[:, t, :])  # [B, 1]
            predictions.append(frame_pred)
        
        predictions = torch.cat(predictions, dim=1)  # [B, T]
        return predictions

class SequenceDiscriminator(nn.Module):
    """Sequence-wise discriminator (Dis^s)"""
    def __init__(self, input_dim, hidden_dim=128, sequence_length=50):
        super(SequenceDiscriminator, self).__init__()
        self.sequence_length = sequence_length
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, T, J, C] - motion sequence
        Returns:
            prediction: [B, 1] - real/fake prediction for entire sequence
        """
        B, T, J, C = x.shape
        x_flat = x.view(B, T, -1)  # [B, T, J*C]
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x_flat)
        
        # Use final hidden state for classification
        final_hidden = torch.cat([h_n[0], h_n[1]], dim=1)  # [B, hidden_dim*2]
        prediction = self.classifier(final_hidden)  # [B, 1]
        
        return prediction

class MotionVAE(nn.Module):
    """Complete VAE system for motion augmentation"""
    def __init__(self, input_shape=(50, 17, 3), latent_dim=128, num_joints=17):
        super(MotionVAE, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.num_joints = num_joints
        
        input_dim = np.prod(input_shape)
        
        # VAE components
        self.encoder = MotionEncoder(input_dim, latent_dim=latent_dim, num_joints=num_joints)
        self.decoder = MotionDecoder(latent_dim, output_shape=input_shape, num_joints=num_joints)
        
        # Discriminators for adversarial training
        self.frame_discriminator = FrameDiscriminator(input_shape[1] * input_shape[2])  # J*C
        self.sequence_discriminator = SequenceDiscriminator(input_shape[1] * input_shape[2])  # J*C
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, joint_attention_scores=None):
        """
        Args:
            x: [B, T, J, C] - input motion
            joint_attention_scores: [B, T, J, J] - attention scores from main model
        Returns:
            reconstructed: [B, T, J, C] - reconstructed motion
            mu: [B, latent_dim] - latent mean
            logvar: [B, latent_dim] - latent log variance
            joint_weights: [B, J] - joint-specific weights
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Sample from latent space
        z = self.reparameterize(mu, logvar)
        
        # Decode with joint attention
        reconstructed, joint_weights = self.decoder(z, joint_attention_scores)
        
        return reconstructed, mu, logvar, joint_weights
    
    def generate_augmented_motion(self, x, joint_attention_scores=None, num_samples=1, variation_scale=1.0):
        """Generate augmented motion samples"""
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encoder(x)
            
            augmented_samples = []
            for _ in range(num_samples):
                # Add noise to latent space for variation
                noise = torch.randn_like(mu) * variation_scale
                z = self.reparameterize(mu + noise, logvar)
                
                # Decode with joint attention
                augmented, joint_weights = self.decoder(z, joint_attention_scores)
                augmented_samples.append(augmented)
            
            return torch.stack(augmented_samples, dim=1)  # [B, num_samples, T, J, C]

# ===== MODIFIED SGCN_LSTM CLASS =====

class SGCN_LSTM_VAE(nn.Module):
    def __init__(self, AD, AD2, bias_mat_1, bias_mat_2, num_joints, 
                 use_vae_augmentation=True, latent_dim=128):
        super(SGCN_LSTM_VAE, self).__init__()
        self.AD = AD
        self.AD2 = AD2
        self.bias_mat_1 = bias_mat_1
        self.bias_mat_2 = bias_mat_2
        self.num_joints = num_joints
        self.use_vae_augmentation = use_vae_augmentation
        
        # Original SGCN components
        self.temporal3C = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(9, 1), padding=(4, 0))
        self.gcn_conv67C = nn.Conv2d(64 + 3, 64, kernel_size=(1, 1))
        self.temporal48C = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(9, 1), padding=(4, 0))
        self.gcn_conv112C = nn.Conv2d(64 + 48, 64, kernel_size=(1, 1))
        
        # Modified attention mechanisms that return attention scores
        self.attention_1 = ParameterizedAttention(feature_dim=64, num_joints=num_joints, hidden_dim=32)
        self.attention_2 = ParameterizedAttention(feature_dim=64, num_joints=num_joints, hidden_dim=32)
        
        # ConvLSTM
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
        
        # VAE for data augmentation
        if self.use_vae_augmentation:
            # Initialize with dummy shape - will be properly set during first forward pass
            self.motion_vae = None
            self.latent_dim = latent_dim
            self.vae_initialized = False
        
        # Store attention scores for VAE
        self.attention_scores = None
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def _initialize_vae(self, input_tensor):
        """Initialize VAE with correct input dimensions"""
        if self.use_vae_augmentation and not self.vae_initialized:
            B, T, J, C = input_tensor.shape
            input_shape = (T, J, C)
            self.motion_vae = MotionVAE(
                input_shape=input_shape, 
                latent_dim=self.latent_dim, 
                num_joints=J
            ).to(self.device)
            self.vae_initialized = True
            print(f"VAE initialized with input shape: {input_shape}")
        
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
        
        # Apply parameterized attention for first hop and store attention scores
        gcn_x1, attention_scores_1 = self.attention_1(x1, self.bias_mat_1)  # [B, T, J, C], [B, T, J, J]
        
        """Second hop localization - neighbour of neighbour joints (bias_mat_2)"""
        if k.shape[1] == 67: #C=64+3
            y1 = F.relu(self.gcn_conv67C(k))
        else: #C=64+48
            y1 = F.relu(self.gcn_conv112C(k))
        
        # y1: [B, C, T, J] -> [B, T, J, C]
        y1 = y1.permute(0, 2, 3, 1)
        
        # Apply parameterized attention for second hop
        gcn_y1, attention_scores_2 = self.attention_2(y1, self.bias_mat_2)  # [B, T, J, C], [B, T, J, J]
        
        # Store combined attention scores for VAE
        self.attention_scores = (attention_scores_1 + attention_scores_2) / 2
        
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
        # Initialize VAE on first forward pass
        if self.use_vae_augmentation:
            self._initialize_vae(x)
            
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
    
    def augment_data_with_vae(self, x, num_augmented=2, variation_scale=1.0):
        """Generate augmented data using VAE with joint-specific attention"""
        if not self.use_vae_augmentation:
            return x
        
        # First forward pass to get attention scores
        _ = self.forward(x)
        
        # Generate augmented samples using VAE
        augmented_samples = self.motion_vae.generate_augmented_motion(
            x, 
            joint_attention_scores=self.attention_scores,
            num_samples=num_augmented,
            variation_scale=variation_scale
        )
        
        return augmented_samples  # [B, num_samples, T, J, C]
    
    def train_with_vae_augmentation(self, train_x, train_y, lr=0.0001, epochs=200, batch_size=10,
                                   vae_lr=0.001, vae_weight=0.1, adv_weight=0.01):
        """Training with VAE augmentation and adversarial loss"""
        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)
        
        # Initialize main optimizer (will update later if VAE is used)
        self.main_optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.vae_optimizer = None
        self.disc_optimizer = None
        
        # Loss functions
        main_criterion = nn.HuberLoss()
        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()
        
        self.train()
        
        # Flag to track if optimizers need to be recreated
        optimizers_initialized = False
        
        for epoch in range(epochs):
            permutation = torch.randperm(train_x.size(0))
            losses = []
            vae_losses = []
            
            for i in range(0, train_x.size(0), batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = train_x[indices], train_y[indices]
                
                # === Main model training ===
                self.main_optimizer.zero_grad()
                output = self.forward(batch_x)
                main_loss = main_criterion(output.squeeze(), batch_y.squeeze())
                
                # === VAE training ===
                if self.use_vae_augmentation and self.vae_initialized:
                    # Initialize optimizers on first use
                    if 'vae_optimizer' not in locals():
                        vae_optimizer = torch.optim.Adam(self.motion_vae.parameters(), lr=vae_lr)
                        disc_optimizer = torch.optim.Adam(
                            list(self.motion_vae.frame_discriminator.parameters()) + 
                            list(self.motion_vae.sequence_discriminator.parameters()), 
                            lr=vae_lr
                        )
                    
                    # VAE reconstruction loss
                    vae_optimizer.zero_grad()
                    
                    reconstructed, mu, logvar, joint_weights = self.motion_vae(batch_x, self.attention_scores)
                    
                    # Reconstruction loss
                    recon_loss = mse_loss(reconstructed, batch_x)
                    
                    # KL divergence loss
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_x.shape[0]
                    
                    # Discriminator losses
                    # Frame discriminator
                    real_frame_pred = self.motion_vae.frame_discriminator(batch_x)
                    fake_frame_pred = self.motion_vae.frame_discriminator(reconstructed.detach())
                    
                    disc_frame_loss = (bce_loss(real_frame_pred, torch.ones_like(real_frame_pred)) + 
                                     bce_loss(fake_frame_pred, torch.zeros_like(fake_frame_pred))) / 2
                    
                    # Sequence discriminator
                    real_seq_pred = self.motion_vae.sequence_discriminator(batch_x)
                    fake_seq_pred = self.motion_vae.sequence_discriminator(reconstructed.detach())
                    
                    disc_seq_loss = (bce_loss(real_seq_pred, torch.ones_like(real_seq_pred)) + 
                                   bce_loss(fake_seq_pred, torch.zeros_like(fake_seq_pred))) / 2
                    
                    # Generator adversarial loss
                    gen_frame_pred = self.motion_vae.frame_discriminator(reconstructed)
                    gen_seq_pred = self.motion_vae.sequence_discriminator(reconstructed)
                    
                    gen_adv_loss = (bce_loss(gen_frame_pred, torch.ones_like(gen_frame_pred)) + 
                                  bce_loss(gen_seq_pred, torch.ones_like(gen_seq_pred))) / 2
                    
                    # Total VAE loss
                    vae_loss = recon_loss + vae_weight * kl_loss + adv_weight * gen_adv_loss
                    
                    # Update discriminators
                    if self.disc_optimizer is not None:
                        self.disc_optimizer.zero_grad()
                        disc_loss = disc_frame_loss + disc_seq_loss
                        disc_loss.backward(retain_graph=True)
                        self.disc_optimizer.step()
                    
                    # Update VAE
                    if self.vae_optimizer is not None:
                        vae_loss.backward(retain_graph=True)
                        self.vae_optimizer.step()
                    
                    # Add VAE loss to main loss
                    total_loss = main_loss + vae_weight * recon_loss
                    vae_losses.append(vae_loss.item())
                else:
                    total_loss = main_loss
                
                # Update main model
                total_loss.backward()
                self.main_optimizer.step()
                
                losses.append(main_loss.item())
            
            if self.use_vae_augmentation and len(vae_losses) > 0:
                print(f"Epoch {epoch+1}/{epochs}, Main Loss: {np.mean(losses):.4f}, VAE Loss: {np.mean(vae_losses):.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(losses):.4f}")
    
    def predict(self, test_x):
        self.eval()
        test_x = test_x.to(self.device)
        with torch.no_grad():
            return self.forward(test_x)

# Usage Example - IMPORTANT: Use the new class name!
# Replace your original SGCN_LSTM with SGCN_LSTM_VAE:

# OLD (original class):
# model = SGCN_LSTM(AD, AD2, bias_mat_1, bias_mat_2, num_joints=17)

# NEW (with VAE support):
# model = SGCN_LSTM_VAE(AD, AD2, bias_mat_1, bias_mat_2, num_joints=25, use_vae_augmentation=True)
# model.train_with_vae_augmentation(train_x, train_y, epochs=200)

# Or without VAE (acts like original):
# model = SGCN_LSTM_VAE(AD, AD2, bias_mat_1, bias_mat_2, num_joints=25, use_vae_augmentation=False)
# model.train_model(train_x, train_y, epochs=200)  # Use original training method
