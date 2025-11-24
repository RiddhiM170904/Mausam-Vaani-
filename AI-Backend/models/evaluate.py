"""
Model evaluation and metrics utilities.

This module provides:
- Evaluation metrics (MAE, RMSE, MAPE)
- Visualization of predictions
- Model performance analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from pathlib import Path


class WeatherEvaluator:
    """Evaluate weather prediction model."""
    
    def __init__(self, model, device='cpu'):
        """
        Initialize evaluator.
        
        Args:
            model: Trained WeatherTFT model
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.target_names = [
            'Temperature (°C)',
            'Humidity (%)',
            'Wind Speed (km/h)',
            'Rainfall (mm)',
            'Pressure (hPa)',
            'Cloud Cover (%)'
        ]
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: Ground truth values (batch, forecast_steps, num_targets)
            y_pred: Predicted values (batch, forecast_steps, num_targets)
        
        Returns:
            Dictionary of metrics
        """
        # Flatten for overall metrics
        y_true_flat = y_true.reshape(-1, y_true.shape[-1])
        y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
        
        metrics = {}
        
        # Overall metrics
        metrics['mae'] = mean_absolute_error(y_true_flat, y_pred_flat)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100
        metrics['mape'] = mape
        
        # Per-feature metrics
        for i, name in enumerate(self.target_names):
            feature_mae = mean_absolute_error(y_true_flat[:, i], y_pred_flat[:, i])
            feature_rmse = np.sqrt(mean_squared_error(y_true_flat[:, i], y_pred_flat[:, i]))
            
            metrics[f'{name}_MAE'] = feature_mae
            metrics[f'{name}_RMSE'] = feature_rmse
        
        return metrics
    
    def evaluate(self, dataloader, encoder_steps=168, forecast_steps=24):
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: PyTorch DataLoader
            encoder_steps: Number of encoder steps
            forecast_steps: Number of forecast steps
        
        Returns:
            Dictionary with predictions, targets, and metrics
        """
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in dataloader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions = self.model(
                    features,
                    encoder_steps=encoder_steps,
                    forecast_steps=forecast_steps
                )
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Concatenate all batches
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Calculate metrics
        metrics = self.calculate_metrics(targets, predictions)
        
        return {
            'predictions': predictions,
            'targets': targets,
            'metrics': metrics
        }
    
    def plot_predictions(self, targets, predictions, sample_idx=0, save_path=None):
        """
        Plot predicted vs actual values for a single sample.
        
        Args:
            targets: Target values (samples, forecast_steps, num_targets)
            predictions: Predicted values (samples, forecast_steps, num_targets)
            sample_idx: Index of sample to plot
            save_path: Path to save figure
        """
        num_targets = targets.shape[-1]
        forecast_steps = targets.shape[1]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        hours = np.arange(1, forecast_steps + 1)
        
        for i in range(num_targets):
            ax = axes[i]
            
            ax.plot(hours, targets[sample_idx, :, i], 
                   label='Actual', marker='o', linewidth=2)
            ax.plot(hours, predictions[sample_idx, :, i], 
                   label='Predicted', marker='s', linewidth=2, alpha=0.7)
            
            ax.set_xlabel('Hours Ahead', fontsize=12)
            ax.set_ylabel(self.target_names[i], fontsize=12)
            ax.set_title(f'{self.target_names[i]} Prediction', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved prediction plot: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_error_distribution(self, targets, predictions, save_path=None):
        """
        Plot error distribution for each target variable.
        
        Args:
            targets: Target values
            predictions: Predicted values
            save_path: Path to save figure
        """
        num_targets = targets.shape[-1]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for i in range(num_targets):
            errors = (predictions[:, :, i] - targets[:, :, i]).flatten()
            
            ax = axes[i]
            ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(0, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('Prediction Error', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'{self.target_names[i]} Error Distribution', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            ax.text(0.05, 0.95, f'Mean: {mean_error:.2f}\nStd: {std_error:.2f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved error distribution plot: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_metrics_by_horizon(self, targets, predictions, save_path=None):
        """
        Plot metrics as a function of forecast horizon.
        
        Args:
            targets: Target values
            predictions: Predicted values
            save_path: Path to save figure
        """
        forecast_steps = targets.shape[1]
        num_targets = targets.shape[-1]
        
        # Calculate MAE for each forecast step
        mae_by_step = np.zeros((forecast_steps, num_targets))
        
        for step in range(forecast_steps):
            for feature in range(num_targets):
                mae = mean_absolute_error(
                    targets[:, step, feature],
                    predictions[:, step, feature]
                )
                mae_by_step[step, feature] = mae
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        hours = np.arange(1, forecast_steps + 1)
        
        for i in range(num_targets):
            ax.plot(hours, mae_by_step[:, i], marker='o', 
                   label=self.target_names[i], linewidth=2)
        
        ax.set_xlabel('Hours Ahead', fontsize=14)
        ax.set_ylabel('Mean Absolute Error', fontsize=14)
        ax.set_title('Prediction Error vs Forecast Horizon', fontsize=16, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved horizon metrics plot: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(self, results, output_dir='results'):
        """
        Generate comprehensive evaluation report.
        
        Args:
            results: Results dictionary from evaluate()
            output_dir: Directory to save report and plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        predictions = results['predictions']
        targets = results['targets']
        metrics = results['metrics']
        
        # Print metrics
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        
        print("\nOverall Metrics:")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        
        print("\nPer-Feature Metrics:")
        for name in self.target_names:
            mae = metrics[f'{name}_MAE']
            rmse = metrics[f'{name}_RMSE']
            print(f"  {name}:")
            print(f"    MAE:  {mae:.4f}")
            print(f"    RMSE: {rmse:.4f}")
        
        print("=" * 60)
        
        # Generate plots
        print("\nGenerating visualizations...")
        
        # Sample predictions
        self.plot_predictions(
            targets, predictions, sample_idx=0,
            save_path=output_dir / 'sample_predictions.png'
        )
        
        # Error distribution
        self.plot_error_distribution(
            targets, predictions,
            save_path=output_dir / 'error_distribution.png'
        )
        
        # Metrics by horizon
        self.plot_metrics_by_horizon(
            targets, predictions,
            save_path=output_dir / 'metrics_by_horizon.png'
        )
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(output_dir / 'metrics.csv', index=False)
        print(f"Saved metrics to: {output_dir / 'metrics.csv'}")
        
        print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    print("Evaluation utilities module ready!")
    print("\nUsage example:")
    print("""
    from models.evaluate import WeatherEvaluator
    from models.tft_model import WeatherTFT
    import torch
    
    # Load trained model
    model = WeatherTFT()
    checkpoint = torch.load('checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = WeatherEvaluator(model, device='cuda')
    
    # Evaluate on test set
    results = evaluator.evaluate(test_loader)
    
    # Generate report
    evaluator.generate_report(results, output_dir='results')
    """)
