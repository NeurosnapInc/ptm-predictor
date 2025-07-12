import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from typing import Dict, List, Tuple

class PTMMetrics:
    """Metrics calculator for PTM prediction"""
    
    def __init__(self, num_ptm_types: int):
        self.num_ptm_types = num_ptm_types
        self.ptm_names = [f"PTM_{i}" for i in range(num_ptm_types)]
        self.reset()
    
    def reset(self):
        """Reset all metric accumulators"""
        self.all_predictions = []
        self.all_labels = []
        self.all_masks = []
    
    def update(self, predictions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
        """
        Update metric accumulators with batch results
        
        Args:
            predictions: (batch_size, seq_len, num_ptm_types) - sigmoid outputs
            labels: (batch_size, seq_len, num_ptm_types) - ground truth
            mask: (batch_size, seq_len) - attention mask for valid positions
        """
        # Only keep valid positions (excluding padding and special tokens)
        # Mask should exclude <cls>, <eos>, and padding tokens
        valid_mask = mask.bool()
        
        # Flatten and filter by mask
        batch_size, seq_len, num_ptm = predictions.shape
        
        # Reshape for easier masking
        predictions_flat = predictions.view(-1, num_ptm)
        labels_flat = labels.view(-1, num_ptm)
        mask_flat = valid_mask.view(-1)
        
        # Filter valid positions
        valid_predictions = predictions_flat[mask_flat]
        valid_labels = labels_flat[mask_flat]
        
        # Append predictions and labels
        self.all_predictions.append(valid_predictions.cpu())
        self.all_labels.append(valid_labels.cpu())
    
    def compute(self, threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute all metrics
        
        Args:
            threshold: Threshold for binary classification
            
        Returns:
            Dictionary containing all computed metrics
        """
        if not self.all_predictions:
            return {}
        
        # Concatenate all batches
        all_preds = torch.cat(self.all_predictions, dim=0).numpy()
        all_labels = torch.cat(self.all_labels, dim=0).numpy()
        
        # Binary predictions
        binary_preds = (all_preds >= threshold).astype(int)
        
        metrics = {}
        
        # Overall metrics (macro-averaged across PTM types)
        overall_precision, overall_recall, overall_f1 = [], [], []
        
        # Per-PTM metrics
        for i in range(self.num_ptm_types):
            ptm_name = self.ptm_names[i]
            ptm_labels = all_labels[:, i]
            ptm_preds = binary_preds[:, i]
            ptm_probs = all_preds[:, i]
            
            # Skip if no positive samples for this PTM type
            if ptm_labels.sum() == 0:
                continue
            
            # Precision, Recall, F1
            precision, recall, f1, _ = precision_recall_fscore_support(
                ptm_labels, ptm_preds, average='binary', zero_division=0
            )
            
            metrics[f'{ptm_name}_precision'] = float(precision)
            metrics[f'{ptm_name}_recall'] = float(recall)
            metrics[f'{ptm_name}_f1'] = float(f1)
            
            overall_precision.append(precision)
            overall_recall.append(recall)
            overall_f1.append(f1)
            
            # AUC-ROC (only if we have both classes)
            if len(np.unique(ptm_labels)) > 1:
                try:
                    auc = roc_auc_score(ptm_labels, ptm_probs)
                    metrics[f'{ptm_name}_auc'] = float(auc)
                except:
                    pass
        
        # Overall metrics
        if overall_precision:
            metrics['overall_precision'] = float(np.mean(overall_precision))
            metrics['overall_recall'] = float(np.mean(overall_recall))
            metrics['overall_f1'] = float(np.mean(overall_f1))
        
        # Position-wise accuracy (at least one PTM correct)
        any_correct = (binary_preds == all_labels).any(axis=1)
        metrics['position_accuracy'] = float(any_correct.mean())
        
        # Exact match accuracy (all PTMs correct for a position)
        all_correct = (binary_preds == all_labels).all(axis=1)
        metrics['exact_match_accuracy'] = float(all_correct.mean())
        
        return metrics
    
    def get_per_ptm_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about positive samples per PTM type"""
        if not self.all_labels:
            return {}
        
        all_labels = torch.cat(self.all_labels, dim=0).numpy()
        stats = {}
        
        for i in range(self.num_ptm_types):
            ptm_name = self.ptm_names[i]
            positive_count = int(all_labels[:, i].sum())
            total_count = len(all_labels[:, i])
            
            stats[ptm_name] = {
                'positive_samples': positive_count,
                'negative_samples': total_count - positive_count,
                'positive_ratio': positive_count / total_count if total_count > 0 else 0
            }
        
        return stats


def calculate_class_weights(train_loader, num_ptm_types: int, device: str = 'cuda') -> torch.Tensor:
    """
    Calculate class weights for handling imbalanced data
    
    Args:
        train_loader: Training data loader
        num_ptm_types: Number of PTM types
        device: Device to place weights on
        
    Returns:
        Tensor of shape (num_ptm_types, 2) with weights for [negative, positive] classes
    """
    pos_counts = torch.zeros(num_ptm_types)
    total_counts = torch.zeros(num_ptm_types)
    
    for batch in train_loader:
        labels = batch['labels']
        mask = batch['attention_mask']
        
        # Only count valid positions
        valid_mask = (mask == 1)
        
        for i in range(num_ptm_types):
            ptm_labels = labels[:, :, i]
            valid_labels = ptm_labels[valid_mask]
            
            pos_counts[i] += valid_labels.sum().item()
            total_counts[i] += valid_labels.numel()
    
    # Calculate weights
    neg_counts = total_counts - pos_counts
    
    # Avoid division by zero
    pos_counts = torch.clamp(pos_counts, min=1)
    neg_counts = torch.clamp(neg_counts, min=1)
    
    # Calculate weights as inverse frequency
    weights = torch.zeros(num_ptm_types, 2)
    weights[:, 0] = total_counts / (2 * neg_counts)  # Weight for negative class
    weights[:, 1] = total_counts / (2 * pos_counts)  # Weight for positive class
    
    # Normalize weights
    weights = weights / weights.sum(dim=1, keepdim=True)
    
    return weights.to(device)