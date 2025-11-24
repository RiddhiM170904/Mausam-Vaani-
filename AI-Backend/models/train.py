"""
Training script for Weather TFT model.

This script handles:
- Loading and preprocessing data
- Model initialization
- Training loop with validation
- Checkpointing and early stopping
- TensorBoard logging
"""

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.tft_model import create_model
from models.data_preprocessing import WeatherDataProcessor, create_dataloaders


class Trainer:
    """Training manager for Weather TFT model."""
    
    def __init__(self, config):
        """
        Initialize trainer.
        
        Args:
            config: Dictionary with training configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = create_model(config['model']).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Directories
        self.checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config['paths']['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(self.log_dir)
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, (features, targets) in enumerate(pbar):
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(
                features,
                encoder_steps=self.config['data']['encoder_steps'],
                forecast_steps=self.config['data']['forecast_steps']
            )
            
            # Calculate loss
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['grad_clip']
            )
            
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to TensorBoard
            step = epoch * len(train_loader) + batch_idx
            self.writer.add_scalar('Loss/train_step', loss.item(), step)
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
            
            for features, targets in pbar:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions = self.model(
                    features,
                    encoder_steps=self.config['data']['encoder_steps'],
                    forecast_steps=self.config['data']['forecast_steps']
                )
                
                # Calculate loss
                loss = self.criterion(predictions, targets)
                total_loss += loss.item()
                
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['val_loss']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        return True
    
    def train(self, train_loader, val_loader):
        """Main training loop."""
        epochs = self.config['training']['epochs']
        early_stopping_patience = self.config['training']['early_stopping_patience']
        
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Total epochs: {epochs}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Batch size: {train_loader.batch_size}")
        print("=" * 60 + "\n")
        
        for epoch in range(self.current_epoch, epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self.validate(val_loader, epoch)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train_epoch', train_loss, epoch)
            self.writer.add_scalar('Loss/val_epoch', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  LR:         {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                print(f"  ✓ New best model! Val loss: {val_loss:.6f}")
            else:
                self.patience_counter += 1
                print(f"  → No improvement ({self.patience_counter}/{early_stopping_patience})")
            
            # Save checkpoint
            if (epoch + 1) % self.config['training']['save_every'] == 0:
                self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            print("-" * 60)
        
        # Save final checkpoint
        self.save_checkpoint(epoch, val_loss, is_best=False)
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print("=" * 60)
        
        self.writer.close()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Weather TFT Model')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Override epochs if specified
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    
    # Initialize data processor
    print("\nInitializing data processor...")
    processor = WeatherDataProcessor(config['data']['csv_path'])
    
    # Load and preprocess data
    processor.load_data()
    processor.clean_data()
    processor.engineer_features()
    
    # Create sequences
    data = processor.prepare_sequences(
        encoder_steps=config['data']['encoder_steps'],
        forecast_steps=config['data']['forecast_steps']
    )
    
    # Split data
    train_data, val_data, test_data = processor.split_data(
        data['features'], data['targets'],
        data['locations'], data['timestamps'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio']
    )
    
    # Normalize data
    train_data['features'], train_data['targets'] = processor.normalize_data(
        train_data['features'], train_data['targets'], fit=True
    )
    val_data['features'], val_data['targets'] = processor.normalize_data(
        val_data['features'], val_data['targets'], fit=False
    )
    test_data['features'], test_data['targets'] = processor.normalize_data(
        test_data['features'], test_data['targets'], fit=False
    )
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = Trainer(config)
    
    # Store normalization parameters in model
    trainer.model.feature_mean = torch.FloatTensor(processor.feature_scaler.mean_).to(trainer.device)
    trainer.model.feature_std = torch.FloatTensor(processor.feature_scaler.scale_).to(trainer.device)
    trainer.model.target_mean = torch.FloatTensor(processor.target_scaler.mean_).to(trainer.device)
    trainer.model.target_std = torch.FloatTensor(processor.target_scaler.scale_).to(trainer.device)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train model
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
