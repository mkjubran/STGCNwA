import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from GCN.data_processing import Data_Loader
from GCN.graphPyTorch import get_graph_data
from GCN.sgcn_lstm_parametrizedA_pytorch import SGCN_LSTM
import skelbumentations as S
from torch.utils.data import Dataset, DataLoader
import random

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class SkeletonDataset(Dataset):
    """Custom dataset class for skeleton data with augmentation support"""
    def __init__(self, X, y, augment_transform=None, augment_probability=0.5):
        self.X = X
        self.y = y
        self.augment_transform = augment_transform
        self.augment_probability = augment_probability
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x_sample = self.X[idx]  # Shape: [T, J, C]
        y_sample = self.y[idx]
        
        # Apply augmentation with given probability
        if self.augment_transform is not None and random.random() < self.augment_probability:
            try:
                # Convert to format expected by skelbumentation: [T, J, C]
                keypoints = x_sample.numpy() if isinstance(x_sample, torch.Tensor) else x_sample
                
                # Apply augmentation
                augmented = self.augment_transform(keypoints=keypoints)
                augmented_keypoints = augmented["keypoints"]
                
                # Convert back to tensor
                x_sample = torch.tensor(augmented_keypoints, dtype=torch.float32)
            except Exception as e:
                print(f"Warning: Augmentation failed for sample {idx}: {e}")
                # Fall back to original data
                pass
        
        return x_sample, y_sample

def create_augmentation_pipeline(num_joints):
    """Create skelbumentation augmentation pipeline"""
    
    # Define opposite points for mirror perturbation (skeleton-specific)
    # This is a basic example - you should adjust based on your skeleton structure
    opposite_points = [
        # Left-Right shoulder pairs
        (4, 8),   # Left shoulder - Right shoulder (adjust indices based on your skeleton)
        (5, 9),   # Left elbow - Right elbow
        (6, 10),  # Left wrist - Right wrist
        (7, 11),  # Left hand - Right hand
        (12, 16), # Left hip - Right hip
        (13, 17), # Left knee - Right knee
        (14, 18), # Left ankle - Right ankle
        (15, 19), # Left foot - Right foot
        # Add more pairs as needed based on your skeleton structure
    ]
    
    # Filter opposite_points to only include valid joint indices
    valid_opposite_points = [(i, j) for i, j in opposite_points 
                           if i < num_joints and j < num_joints]
    
    augment_pipeline = S.Compose([
        # Randomly select frames for augmentation
        S.SelectRandomFrames(
            [
                S.OneOf([
                    S.SwapPerturbation(prob=0.3),
                    S.MirrorPerturbation(
                        opposite_points=valid_opposite_points, 
                        prob=0.3
                    ) if valid_opposite_points else S.SwapPerturbation(prob=0.3),
                ])
            ],
            min_num=3,
            max_num=8,
        ),
        # Movement perturbation
        S.MovePerturbation(variance=0.05, prob=0.4),
        
        # Joint-wise perturbations
        S.JointPerturbation(variance=0.02, prob=0.3),
        
        # Scale perturbation
        S.ScalePerturbation(variance=0.1, prob=0.2),
    ])
    
    return augment_pipeline

def create_conservative_augmentation_pipeline():
    """Create a more conservative augmentation pipeline for regression tasks"""
    
    augment_pipeline = S.Compose([
        # Light movement perturbation
        S.MovePerturbation(variance=0.02, prob=0.4),
        
        # Small joint perturbations
        S.JointPerturbation(variance=0.01, prob=0.3),
        
        # Minor scale changes
        S.ScalePerturbation(variance=0.05, prob=0.2),
    ])
    
    return augment_pipeline

def enhanced_train_model(model, train_dataset, val_x, val_y, device, lr=0.0001, epochs=200, batch_size=10):
    """Enhanced training function with data augmentation support"""
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.HuberLoss()
    
    # Create data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output.squeeze(), batch_y.squeeze())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Validation every 50 epochs
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_x.to(device))
                val_loss = criterion(val_pred.squeeze(), val_y.to(device).squeeze())
            model.train()
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss.item():.6f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}")

def main():
    parser = argparse.ArgumentParser(description='PyTorch ST-GCN Trainer with Data Augmentation')
    parser.add_argument('--ex', type=str, required=True, help='Exercise name (e.g., Kimore_ex5)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epoch', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    
    # Data augmentation arguments
    parser.add_argument('--use_augmentation', action='store_true', 
                       help='Enable data augmentation during training')
    parser.add_argument('--aug_probability', type=float, default=0.5,
                       help='Probability of applying augmentation to each sample')
    parser.add_argument('--aug_type', choices=['conservative', 'standard'], default='conservative',
                       help='Type of augmentation pipeline: conservative (small changes) or standard')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    print(f"Training configuration:")
    print(f"  Exercise: {args.ex}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epoch}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Data augmentation: {args.use_augmentation}")
    if args.use_augmentation:
        print(f"  Augmentation probability: {args.aug_probability}")
        print(f"  Augmentation type: {args.aug_type}")
    
    # Load and split dataset
    print("Loading dataset...")
    data_loader = Data_Loader(args.ex)
    train_x, test_x, train_y, test_y = train_test_split(
        data_loader.scaled_x, data_loader.scaled_y, test_size=0.2, random_state=args.seed
    )
    
    # Convert to torch tensors
    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    test_x = torch.tensor(test_x, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)
    
    print(f"Dataset shapes:")
    print(f"  Train X: {train_x.shape}")  # [B, T, J, C]
    print(f"  Train Y: {train_y.shape}")
    print(f"  Test X: {test_x.shape}")
    print(f"  Test Y: {test_y.shape}")
    
    # Create augmentation pipeline
    augment_transform = None
    if args.use_augmentation:
        print("Setting up data augmentation...")
        num_joints = len(data_loader.body_part)
        
        try:
            if args.aug_type == 'conservative':
                augment_transform = create_conservative_augmentation_pipeline()
                print("Using conservative augmentation pipeline")
            else:
                augment_transform = create_augmentation_pipeline(num_joints)
                print("Using standard augmentation pipeline")
        except Exception as e:
            print(f"Warning: Could not create augmentation pipeline: {e}")
            print("Proceeding without augmentation...")
            augment_transform = None
            args.use_augmentation = False
    
    # Create dataset with augmentation
    train_dataset = SkeletonDataset(
        train_x, train_y, 
        augment_transform=augment_transform,
        augment_probability=args.aug_probability if args.use_augmentation else 0.0
    )
    
    # Get graph data
    num_nodes = len(data_loader.body_part)
    AD, AD2, bias_mat_1, bias_mat_2 = get_graph_data(num_nodes)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SGCN_LSTM(AD, AD2, bias_mat_1, bias_mat_2, num_joints=num_nodes)
    
    # Train with augmentation support
    print("Starting training...")
    if args.use_augmentation:
        # Use enhanced training function that supports augmented dataset
        enhanced_train_model(
            model, train_dataset, test_x, test_y, device,
            lr=args.lr, epochs=args.epoch, batch_size=args.batch_size
        )
    else:
        # Use original training method
        model.train_model(train_x, train_y, lr=args.lr, epochs=args.epoch, batch_size=args.batch_size)
    
    # Predict
    print("Evaluating model...")
    y_pred = model.predict(test_x).detach().cpu().numpy()
    test_y_np = test_y.detach().cpu().numpy()
    
    # Inverse transform predictions and targets
    y_pred = data_loader.sc2.inverse_transform(y_pred)
    test_y_np = data_loader.sc2.inverse_transform(test_y_np)
    
    # Calculate metrics
    mae = mean_absolute_error(test_y_np, y_pred)
    mse = mean_squared_error(test_y_np, y_pred)
    mape = mean_absolute_percentage_error(test_y_np, y_pred)
    rms = np.sqrt(mse)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"MAE:  {mae:.6f}")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rms:.6f}")
    print(f"MAPE: {mape:.4f}%")
    print("="*50)
    
    if args.use_augmentation:
        print(f"Training completed with {args.aug_type} data augmentation")
        print(f"Augmentation probability: {args.aug_probability}")
    else:
        print("Training completed without data augmentation")

if __name__ == "__main__":
    main()
