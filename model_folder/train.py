import os
import argparse
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import wandb
from pathlib import Path
from esm import pretrained
import pandas as pd

# Import your modules
from models.model import PTMAdapterModel
from utils.data_loader import get_data_loaders
from utils.metrics import PTMMetrics, calculate_class_weights

def parse_args():
    parser = argparse.ArgumentParser(description='Train PTM prediction model with Focal Loss')
    
    # Data arguments
    parser.add_argument('--train_csv', type=str, default='data/train.csv',
                        help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, default='data/val.csv',
                        help='Path to validation CSV file')
    parser.add_argument('--test_csv', type=str, default='data/test.csv',
                        help='Path to test CSV file')
    
    # Model arguments
    parser.add_argument('--num_ptm_types', type=int, default=10,
                        help='Number of PTM types to predict')
    parser.add_argument('--adapter_dim', type=int, default=64,
                        help='Dimension of adapter layers')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Weight decay for optimizer')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Focal Loss arguments
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use class weights in focal loss alpha parameter')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Alpha parameter for focal loss (used when not using class weights)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter for focal loss')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Logging arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='ptm-prediction',
                        help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='W&B run name')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    
    return parser.parse_args()

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, num_classes) logits
            targets: (N, num_classes) binary labels
        """
        # Calculate BCE loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Calculate pt (probability of correct class)
        pt = torch.exp(-bce_loss)
        
        # Apply focal term
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        
        # Apply alpha if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                # Scalar alpha
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            else:
                # Per-class alpha (should be shape [num_ptm_types])
                if self.alpha.dim() == 1:
                    # Expand alpha to match input shape
                    alpha_t = self.alpha.unsqueeze(0).expand_as(inputs)
                    # Select alpha based on target
                    alpha_t = alpha_t * targets + (1 - alpha_t) * (1 - targets)
                else:
                    alpha_t = self.alpha
            
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_epoch(model, train_loader, optimizer, loss_fn, metrics, device, epoch, args):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
    
    # Enable mixed precision if requested
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and device.type == 'cuda' else None
    
    for batch in progress_bar:
        # Move batch to device
        tokens = batch['tokens'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(tokens)
                # Calculate loss only on valid positions
                valid_mask = attention_mask.bool()
                loss = loss_fn(logits[valid_mask], labels[valid_mask])
        else:
            logits = model(tokens)
            # Calculate loss only on valid positions
            valid_mask = attention_mask.bool()
            loss = loss_fn(logits[valid_mask], labels[valid_mask])
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
        
        # Update metrics
        with torch.no_grad():
            predictions = torch.sigmoid(logits)
            metrics.update(predictions, labels, attention_mask)
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches

def validate(model, val_loader, loss_fn, metrics, device, args):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            # Move batch to device
            tokens = batch['tokens'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            logits = model(tokens)
            
            # Calculate loss
            valid_mask = attention_mask.bool()
            loss = loss_fn(logits[valid_mask], labels[valid_mask])
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update metrics
            predictions = torch.sigmoid(logits)
            metrics.update(predictions, labels, attention_mask)
    
    return total_loss / num_batches

def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, args, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_val_loss,
        'args': vars(args)
    }
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Save regular checkpoint
    if (epoch + 1) % args.save_every == 0:
        checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save best model
    if is_best:
        best_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb if requested
    if args.use_wandb:
        run_name = args.wandb_run_name or f"ptm_focal_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
    
    # Load pretrained ESM-2 model
    print("Loading ESM-2 model...")
    esm_model, alphabet = pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        args.train_csv, args.val_csv, args.test_csv,
        alphabet, batch_converter,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Calculate class weights if requested
    focal_alpha = args.focal_alpha
    if args.use_class_weights:
        print("Calculating class weights for focal loss alpha...")
        class_weights = calculate_class_weights(train_loader, args.num_ptm_types, device)
        # Use positive class weights as alpha for focal loss
        focal_alpha = class_weights[:, 1]  # Shape: (num_ptm_types,)
        print(f"Using class-weighted alpha: {focal_alpha}")
    
    # Create model
    print("Creating model...")
    model = PTMAdapterModel(esm_model, args.num_ptm_types, args.adapter_dim)
    model = model.to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Create optimizer
    optimizer = Adam([
        {"params": model.adapter.parameters()},
        {"params": model.ptm_classifier.parameters()}
    ], lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Create scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Create focal loss function
    loss_fn = FocalLoss(alpha=focal_alpha, gamma=args.focal_gamma)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        # Train metrics
        train_metrics = PTMMetrics(args.num_ptm_types)
        
        # Train for one epoch
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, train_metrics, 
            device, epoch, args
        )
        
        # Compute train metrics
        train_results = train_metrics.compute()
        
        # Validation
        val_metrics = PTMMetrics(args.num_ptm_types)
        val_loss = validate(model, val_loader, loss_fn, val_metrics, device, args)
        
        # Compute validation metrics
        val_results = val_metrics.compute()
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Print results
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train F1: {train_results.get('overall_f1', 0):.4f}, "
              f"Val F1: {val_results.get('overall_f1', 0):.4f}")
        
        # Log to wandb
        if args.use_wandb:
            log_dict = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            
            # Add train metrics
            for k, v in train_results.items():
                log_dict[f'train_{k}'] = v
            
            # Add val metrics
            for k, v in val_results.items():
                log_dict[f'val_{k}'] = v
            
            wandb.log(log_dict)
        
        # Check if this is the best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, args, is_best)
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = PTMMetrics(args.num_ptm_types)
    test_loss = validate(model, test_loader, loss_fn, test_metrics, device, args)
    test_results = test_metrics.compute()
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test F1: {test_results.get('overall_f1', 0):.4f}")
    print(f"Test Precision: {test_results.get('overall_precision', 0):.4f}")
    print(f"Test Recall: {test_results.get('overall_recall', 0):.4f}")
    
    # Save test results
    test_results['test_loss'] = test_loss
    results_path = os.path.join(args.checkpoint_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    if args.use_wandb:
        # Log test results
        test_log = {f'test_{k}': v for k, v in test_results.items()}
        wandb.log(test_log)
        
        # Get per-PTM statistics
        ptm_stats = test_metrics.get_per_ptm_stats()
        
        # Create a summary table
        summary_data = []
        for ptm_name, stats in ptm_stats.items():
            row = {
                'PTM Type': ptm_name,
                'Positive Samples': stats['positive_samples'],
                'Negative Samples': stats['negative_samples'],
                'Positive Ratio': stats['positive_ratio'],
                'F1 Score': test_results.get(f'{ptm_name}_f1', 0),
                'Precision': test_results.get(f'{ptm_name}_precision', 0),
                'Recall': test_results.get(f'{ptm_name}_recall', 0),
                'AUC': test_results.get(f'{ptm_name}_auc', 'N/A')
            }
            summary_data.append(row)
        
        # Log summary table
        wandb.log({"ptm_performance_summary": wandb.Table(dataframe=pd.DataFrame(summary_data))})
        
        wandb.finish()
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()