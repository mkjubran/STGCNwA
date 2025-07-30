import numpy as np
import os
import argparse
import glob

class BiasMatrixViewer:
    def __init__(self, exercise_folder):
        """
        Initialize the bias matrix viewer
        
        Args:
            exercise_folder: Path to the folder containing bias matrix NPZ files
        """
        self.exercise_folder = exercise_folder
        self.available_files = self._get_available_files()
        
    def _get_available_files(self):
        """Get list of available NPZ files in the exercise folder"""
        pattern = os.path.join(self.exercise_folder, "bias_matrices_epoch_*.npz")
        files = glob.glob(pattern)
        # Sort by epoch number
        files.sort(key=lambda x: int(x.split('epoch_')[1].split('.')[0]))
        return files
    
    def list_available_epochs(self):
        """List all available epochs"""
        if not self.available_files:
            print(f"No bias matrix files found in {self.exercise_folder}")
            return []
        
        print(f"Available epochs in {self.exercise_folder}:")
        epochs = []
        for file in self.available_files:
            epoch_num = int(file.split('epoch_')[1].split('.')[0])
            epochs.append(epoch_num)
            print(f"  Epoch {epoch_num}: {os.path.basename(file)}")
        return epochs
    
    def load_epoch(self, epoch_num):
        """
        Load bias matrices from a specific epoch
        
        Args:
            epoch_num: Epoch number to load
            
        Returns:
            dict: Dictionary containing bias_mat_1 and bias_mat_2
        """
        filename = os.path.join(self.exercise_folder, f"bias_matrices_epoch_{epoch_num}.npz")
        
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return None
            
        try:
            data = np.load(filename)
            return {
                'bias_mat_1': data['bias_mat_1'],
                'bias_mat_2': data['bias_mat_2']
            }
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None
    
    def get_value(self, epoch_num, matrix_name, row, col):
        """
        Get specific value from bias matrix
        
        Args:
            epoch_num: Epoch number
            matrix_name: 'bias_mat_1' or 'bias_mat_2'
            row: Row index
            col: Column index
            
        Returns:
            float: Value at specified position
        """
        data = self.load_epoch(epoch_num)
        if data is None:
            return None
            
        if matrix_name not in data:
            print(f"Matrix '{matrix_name}' not found. Available: {list(data.keys())}")
            return None
            
        matrix = data[matrix_name]
        
        if row >= matrix.shape[0] or col >= matrix.shape[1]:
            print(f"Index out of bounds. Matrix shape: {matrix.shape}")
            return None
            
        return matrix[row, col]
    
    def display_value(self, epoch_num, matrix_name, row, col):
        """Display specific value with context information"""
        value = self.get_value(epoch_num, matrix_name, row, col)
        if value is not None:
            print(f"Exercise: {os.path.basename(self.exercise_folder)}")
            print(f"Epoch: {epoch_num}")
            print(f"Matrix: {matrix_name}")
            print(f"Position: ({row}, {col})")
            print(f"Value: {value}")
        return value
    
    def display_matrix_info(self, epoch_num):
        """Display information about matrices in a specific epoch"""
        data = self.load_epoch(epoch_num)
        if data is None:
            return
            
        print(f"\nEpoch {epoch_num} - Matrix Information:")
        print("-" * 40)
        for name, matrix in data.items():
            print(f"{name}:")
            print(f"  Shape: {matrix.shape}")
            print(f"  Min value: {matrix.min():.6f}")
            print(f"  Max value: {matrix.max():.6f}")
            print(f"  Mean value: {matrix.mean():.6f}")
            print(f"  Std value: {matrix.std():.6f}")
            print()
    
    def parse_epoch_range(self, epoch_input):
        """
        Parse epoch input - can be a list, range, or single value
        
        Args:
            epoch_input: Can be:
                - List of integers: [1, 5, 10]
                - String range: "1:100" or "1:100:5" (start:end:step)
                - Single integer: 10
                
        Returns:
            list: List of epoch numbers
        """
        if isinstance(epoch_input, str):
            # Handle range format like "1:100" or "1:100:5"
            if ':' in epoch_input:
                parts = epoch_input.split(':')
                if len(parts) == 2:
                    start, end = map(int, parts)
                    step = 1
                elif len(parts) == 3:
                    start, end, step = map(int, parts)
                else:
                    raise ValueError("Invalid range format. Use start:end or start:end:step")
                return list(range(start, end + 1, step))
            else:
                # Single epoch as string
                return [int(epoch_input)]
        elif isinstance(epoch_input, (list, tuple)):
            return list(epoch_input)
        elif isinstance(epoch_input, int):
            return [epoch_input]
        else:
            raise ValueError("Invalid epoch input format")

    def compare_values_across_epochs(self, matrix_name, row, col, epoch_input=None):
        """
        Compare a specific value across multiple epochs
        
        Args:
            matrix_name: 'bias_mat_1' or 'bias_mat_2'
            row: Row index
            col: Column index
            epoch_input: Can be None (all epochs), list [1,5,10], or range "1:100" or "1:100:5"
        """
        if epoch_input is None:
            epoch_list = [int(f.split('epoch_')[1].split('.')[0]) for f in self.available_files]
            epoch_list.sort()
        else:
            epoch_list = self.parse_epoch_range(epoch_input)
        
        print(f"\nValue evolution at {matrix_name}[{row}, {col}]:")
        print("-" * 50)
        
        values = []
        missing_epochs = []
        
        for epoch in epoch_list:
            value = self.get_value(epoch, matrix_name, row, col)
            if value is not None:
                values.append((epoch, value))
                print(f"Epoch {epoch:3d}: {value:.6f}")
            else:
                missing_epochs.append(epoch)
        
        if missing_epochs:
            print(f"\nMissing epochs: {missing_epochs}")
        
        if len(values) > 1:
            initial_value = values[0][1]
            final_value = values[-1][1]
            change = final_value - initial_value
            print(f"\nChange from epoch {values[0][0]} to {values[-1][0]}: {change:.6f}")
            
            # Additional statistics for ranges
            if len(values) > 2:
                all_values = [v[1] for v in values]
                print(f"Min value: {min(all_values):.6f}")
                print(f"Max value: {max(all_values):.6f}")
                print(f"Mean value: {np.mean(all_values):.6f}")
                print(f"Std deviation: {np.std(all_values):.6f}")

    def display_value_range(self, matrix_name, row, col, epoch_input):
        """
        Display values across a range of epochs in a compact format
        
        Args:
            matrix_name: 'bias_mat_1' or 'bias_mat_2'
            row: Row index
            col: Column index
            epoch_input: Range specification like "1:100" or list of epochs
        """
        epoch_list = self.parse_epoch_range(epoch_input)
        
        print(f"\nValues at {matrix_name}[{row}, {col}] across epochs:")
        print("-" * 60)
        
        values = []
        for epoch in epoch_list:
            value = self.get_value(epoch, matrix_name, row, col)
            if value is not None:
                values.append((epoch, value))
        
        if not values:
            print("No valid epochs found.")
            return
        
        # Display in compact format (10 values per line)
        for i, (epoch, value) in enumerate(values):
            if i % 10 == 0 and i > 0:
                print()  # New line every 10 values
            print(f"{epoch:3d}:{value:7.4f}", end="  ")
        
        print("\n")  # Final newline
        
        # Summary statistics
        all_values = [v[1] for v in values]
        print(f"Summary - Epochs {values[0][0]} to {values[-1][0]} ({len(values)} epochs):")
        print(f"  Range: {min(all_values):.6f} to {max(all_values):.6f}")
        print(f"  Mean: {np.mean(all_values):.6f}, Std: {np.std(all_values):.6f}")
        print(f"  Total change: {all_values[-1] - all_values[0]:.6f}")

def main():
    parser = argparse.ArgumentParser(description='View bias matrices from training epochs')
    parser.add_argument('--exercise_folder', type=str, required=True,
                        help='Path to folder containing bias matrix NPZ files')
    parser.add_argument('--epoch', type=int, help='Specific epoch to load')
    parser.add_argument('--matrix', type=str, choices=['bias_mat_1', 'bias_mat_2'],
                        help='Which matrix to examine')
    parser.add_argument('--row', type=int, help='Row index')
    parser.add_argument('--col', type=int, help='Column index')
    parser.add_argument('--info', action='store_true',
                        help='Display matrix information for specified epoch')
    parser.add_argument('--compare_epochs', type=str,
                        help='Compare value across epochs. Format: "1,5,10" or "1:100" or "1:100:5"')
    parser.add_argument('--display_range', type=str,
                        help='Display values in compact format. Format: "1:100" or "1:100:5"')
    
    args = parser.parse_args()
    
    # Create viewer instance
    viewer = BiasMatrixViewer(args.exercise_folder)
    
    # List available epochs
    available_epochs = viewer.list_available_epochs()
    if not available_epochs:
        return
    
    # If no specific action requested, just show available epochs
    if not any([args.epoch, args.info, args.compare_epochs, args.display_range]):
        return
    
    # Display matrix info
    if args.info and args.epoch:
        viewer.display_matrix_info(args.epoch)
    
    # Display specific value
    if all([args.epoch, args.matrix, args.row is not None, args.col is not None]):
        viewer.display_value(args.epoch, args.matrix, args.row, args.col)
    
    # Compare across epochs
    if args.compare_epochs and args.matrix and args.row is not None and args.col is not None:
        # Parse different input formats
        if ',' in args.compare_epochs:
            # Comma-separated list: "1,5,10,20"
            epoch_input = [int(x.strip()) for x in args.compare_epochs.split(',')]
        else:
            # Range format: "1:100" or single epoch
            epoch_input = args.compare_epochs
        
        viewer.compare_values_across_epochs(args.matrix, args.row, args.col, epoch_input)
    
    # Display range in compact format
    if args.display_range and args.matrix and args.row is not None and args.col is not None:
        viewer.display_value_range(args.matrix, args.row, args.col, args.display_range)

if __name__ == "__main__":
    main()

# Example usage as a module:
"""
# Create viewer
viewer = BiasMatrixViewer("KimoreEx1")

# List available epochs
viewer.list_available_epochs()

# Get specific value
value = viewer.get_value(epoch_num=10, matrix_name='bias_mat_1', row=2, col=3)

# Display value with context
viewer.display_value(epoch_num=10, matrix_name='bias_mat_1', row=2, col=3)

# Display matrix information
viewer.display_matrix_info(epoch_num=10)

# Compare value across epochs - Multiple formats supported:
# Range format:
viewer.compare_values_across_epochs('bias_mat_1', row=2, col=3, epoch_input="1:100")
viewer.compare_values_across_epochs('bias_mat_1', row=2, col=3, epoch_input="1:100:5")  # Every 5th epoch

# List format:
viewer.compare_values_across_epochs('bias_mat_1', row=2, col=3, epoch_input=[1, 5, 10, 20])

# Display values in compact format:
viewer.display_value_range('bias_mat_1', row=2, col=3, epoch_input="1:200")
"""
