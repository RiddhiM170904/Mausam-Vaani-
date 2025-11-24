"""
Temporal Fusion Transformer (TFT) Model for Hyperlocal Weather Prediction

This module implements a TFT architecture for multi-step weather forecasting.
The model captures temporal dependencies and attention mechanisms to predict
weather variables like temperature, humidity, rainfall, wind speed, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) component for TFT."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Skip connection
        if input_dim != output_dim:
            self.skip = nn.Linear(input_dim, output_dim)
        else:
            self.skip = None
    
    def forward(self, x):
        # Feed-forward path
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        
        # Gating mechanism
        gate = torch.sigmoid(self.gate(F.elu(self.fc1(x))))
        h = h * gate
        
        # Skip connection
        if self.skip is not None:
            x = self.skip(x)
        
        return self.layer_norm(x + h)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for feature importance."""
    
    def __init__(self, input_dim, num_features, hidden_dim, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_features)
        ])
        self.softmax = nn.Softmax(dim=1)
        self.grn_combine = GatedResidualNetwork(
            num_features * hidden_dim, hidden_dim, hidden_dim, dropout
        )
    
    def forward(self, x):
        # x shape: (batch, num_features, input_dim)
        batch_size = x.size(0)
        
        # Process each feature through its GRN
        processed = []
        for i, grn in enumerate(self.grns):
            processed.append(grn(x[:, i, :]))
        
        # Stack and compute feature weights
        processed = torch.stack(processed, dim=1)  # (batch, num_features, hidden_dim)
        
        # Variable selection weights
        flattened = processed.view(batch_size, -1)
        combined = self.grn_combine(flattened)
        
        return combined, processed


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for Multi-Horizon Weather Forecasting.
    
    Args:
        num_features: Number of input weather features
        hidden_dim: Hidden dimension size
        num_heads: Number of attention heads
        num_layers: Number of LSTM/attention layers
        forecast_horizon: Number of time steps to predict ahead
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        num_features=9,  # temp, humidity, wind_speed, rainfall, pressure, cloud_cover, lat, lon, hour
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        forecast_horizon=24,  # Predict next 24 hours
        output_dim=6,  # Predict: temp, humidity, wind_speed, rainfall, pressure, cloud_cover
        dropout=0.1
    ):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim
        
        # Variable selection network
        self.vsn = VariableSelectionNetwork(
            input_dim=1,
            num_features=num_features,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # LSTM encoder for historical data
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # LSTM decoder for future predictions
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Gated residual networks
        self.grn_post_attention = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout
        )
        
        # Output layers
        self.fc_out = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(output_dim)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_steps=168, forecast_steps=24):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, total_steps, num_features)
               where total_steps = encoder_steps + forecast_steps
            encoder_steps: Number of historical time steps (default: 168 = 1 week)
            forecast_steps: Number of future time steps to predict (default: 24 hours)
        
        Returns:
            predictions: Tensor of shape (batch, forecast_steps, output_dim)
        """
        batch_size = x.size(0)
        
        # Split into encoder and decoder inputs
        encoder_input = x[:, :encoder_steps, :]  # Historical data
        
        # Reshape for variable selection: (batch, num_features, 1)
        # We'll process each time step separately
        encoder_outputs = []
        
        for t in range(encoder_steps):
            step_input = x[:, t, :].unsqueeze(-1)  # (batch, num_features, 1)
            selected, _ = self.vsn(step_input)
            encoder_outputs.append(selected)
        
        encoder_outputs = torch.stack(encoder_outputs, dim=1)  # (batch, encoder_steps, hidden_dim)
        
        # Encode historical sequence
        encoder_out, (h_n, c_n) = self.encoder_lstm(encoder_outputs)
        
        # Decode for future predictions
        decoder_input = encoder_out[:, -1:, :].repeat(1, forecast_steps, 1)
        decoder_out, _ = self.decoder_lstm(decoder_input, (h_n, c_n))
        
        # Apply self-attention
        attn_out, _ = self.attention(decoder_out, encoder_out, encoder_out)
        
        # Gated residual connection
        attn_out = self.grn_post_attention(attn_out.reshape(-1, self.hidden_dim))
        attn_out = attn_out.reshape(batch_size, forecast_steps, self.hidden_dim)
        
        # Generate predictions for each output variable
        predictions = []
        for fc in self.fc_out:
            pred = fc(attn_out)  # (batch, forecast_steps, 1)
            predictions.append(pred)
        
        predictions = torch.cat(predictions, dim=-1)  # (batch, forecast_steps, output_dim)
        
        return predictions


class WeatherTFT(nn.Module):
    """
    Wrapper class for TFT model with weather-specific preprocessing.
    """
    
    def __init__(
        self,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        forecast_horizon=24,
        dropout=0.1
    ):
        super().__init__()
        
        self.tft = TemporalFusionTransformer(
            num_features=9,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            forecast_horizon=forecast_horizon,
            output_dim=6,
            dropout=dropout
        )
        
        # Feature normalization parameters (to be set during training)
        self.register_buffer('feature_mean', torch.zeros(9))
        self.register_buffer('feature_std', torch.ones(9))
        self.register_buffer('target_mean', torch.zeros(6))
        self.register_buffer('target_std', torch.ones(6))
    
    def normalize_features(self, x):
        """Normalize input features."""
        return (x - self.feature_mean) / (self.feature_std + 1e-8)
    
    def denormalize_predictions(self, y):
        """Denormalize predictions to original scale."""
        return y * (self.target_std + 1e-8) + self.target_mean
    
    def forward(self, x, encoder_steps=168, forecast_steps=24):
        """
        Forward pass with normalization.
        
        Args:
            x: Input tensor (batch, total_steps, num_features)
            encoder_steps: Number of historical steps
            forecast_steps: Number of future steps
        
        Returns:
            Denormalized predictions (batch, forecast_steps, 6)
        """
        # Normalize input
        x_norm = self.normalize_features(x)
        
        # Forward pass
        predictions = self.tft(x_norm, encoder_steps, forecast_steps)
        
        # Denormalize predictions
        predictions = self.denormalize_predictions(predictions)
        
        return predictions


def create_model(config):
    """
    Factory function to create model from configuration.
    
    Args:
        config: Dictionary with model hyperparameters
    
    Returns:
        WeatherTFT model instance
    """
    model = WeatherTFT(
        hidden_dim=config.get('hidden_dim', 128),
        num_heads=config.get('num_heads', 4),
        num_layers=config.get('num_layers', 2),
        forecast_horizon=config.get('forecast_horizon', 24),
        dropout=config.get('dropout', 0.1)
    )
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing TFT Model...")
    
    # Create dummy data
    batch_size = 4
    encoder_steps = 168  # 1 week of hourly data
    forecast_steps = 24  # Predict next 24 hours
    num_features = 9
    
    # Random input data
    x = torch.randn(batch_size, encoder_steps + forecast_steps, num_features)
    
    # Create model
    config = {
        'hidden_dim': 128,
        'num_heads': 4,
        'num_layers': 2,
        'forecast_horizon': 24,
        'dropout': 0.1
    }
    
    model = create_model(config)
    
    # Forward pass
    predictions = model(x, encoder_steps=encoder_steps, forecast_steps=forecast_steps)
    
    print(f"Input shape: {x.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Expected shape: ({batch_size}, {forecast_steps}, 6)")
    print("\nModel created successfully!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
