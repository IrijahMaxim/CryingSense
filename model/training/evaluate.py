"""
CryingSense Model Evaluation Script

Evaluates trained CNN model on test dataset with comprehensive metrics:
- Accuracy, Precision, Recall, F1-score (per class and overall)
- Confusion Matrix
- Inference time measurement
- Confidence threshold analysis
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (classification_report, confusion_matrix, 
                            precision_recall_fscore_support, accuracy_score)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from model.models.cnn_model import CryingSenseCNN


def get_label_from_path(path):
    """Extract class label from file path."""
    return os.path.basename(os.path.dirname(path))


class CryingSenseDataset(Dataset):
    """Dataset for loading feature files."""
    def __init__(self, file_list, label_map, feature_base_dir):
        self.file_list = file_list
        self.label_map = label_map
        self.feature_base_dir = feature_base_dir
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Load all feature types and combine them
        mfcc_path = self.file_list[idx]
        
        # Construct paths for other features
        rel_path = os.path.relpath(mfcc_path, os.path.join(self.feature_base_dir, 'mfcc'))
        mel_path = os.path.join(self.feature_base_dir, 'mel_spectrogram', rel_path)
        chroma_path = os.path.join(self.feature_base_dir, 'chroma', rel_path)
        
        # Load and combine features
        mfcc = np.load(mfcc_path)
        mel = np.load(mel_path)
        chroma = np.load(chroma_path)
        
        x = self._combine_features(mfcc, mel, chroma)
        x = torch.tensor(x, dtype=torch.float32)
        
        label_name = get_label_from_path(mfcc_path)
        y = self.label_map[label_name]
        return x, y
    
    def _combine_features(self, mfcc, mel, chroma):
        """Combine features into 4-channel array."""
        target_height = max(mfcc.shape[0], mel.shape[0], chroma.shape[0])
        target_width = mfcc.shape[1]
        
        mfcc_padded = self._pad_feature(mfcc, (target_height, target_width))
        mel_padded = self._pad_feature(mel, (target_height, target_width))
        chroma_padded = self._pad_feature(chroma, (target_height, target_width))
        
        delta_mfcc = np.zeros_like(mfcc)
        delta_mfcc[:, 1:] = mfcc[:, 1:] - mfcc[:, :-1]
        delta_mfcc_padded = self._pad_feature(delta_mfcc, (target_height, target_width))
        
        return np.stack([mfcc_padded, mel_padded, chroma_padded, delta_mfcc_padded], axis=0)
    
    def _pad_feature(self, feature, target_shape):
        """Pad feature to target shape."""
        padded = np.zeros(target_shape, dtype=feature.dtype)
        min_h = min(feature.shape[0], target_shape[0])
        min_w = min(feature.shape[1], target_shape[1])
        padded[:min_h, :min_w] = feature[:min_h, :min_w]
        return padded


def get_file_list_and_labels(feature_base_dir):
    """Get all feature files and create label mapping."""
    mfcc_dir = os.path.join(feature_base_dir, 'mfcc')
    
    if not os.path.exists(mfcc_dir):
        return [], {}
    
    file_list = []
    for root, _, files in os.walk(mfcc_dir):
        for file in files:
            if file.endswith('.npy'):
                file_list.append(os.path.join(root, file))
    
    if not file_list:
        return [], {}
    
    labels = sorted(list(set(get_label_from_path(f) for f in file_list)))
    label_map = {label: i for i, label in enumerate(labels)}
    return file_list, label_map


def evaluate_model(model, test_loader, device, label_names, confidence_threshold=0.6):
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained CNN model
        test_loader: DataLoader for test data
        device: torch device
        label_names: List of class names
        confidence_threshold: Minimum confidence for predictions
    
    Returns:
        dict: Evaluation results
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    inference_times = []
    rejected_samples = 0
    
    print("Running evaluation...")
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            
            # Measure inference time
            start_time = time.time()
            out = model(x)
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            inference_times.append(inference_time)
            
            # Get probabilities
            probs = F.softmax(out, dim=1)
            max_probs, preds = probs.max(1)
            
            # Apply confidence threshold
            for i, (pred, prob) in enumerate(zip(preds, max_probs)):
                if prob.item() < confidence_threshold:
                    rejected_samples += 1
                    # For rejected samples, we still use ground truth for metrics
                    # In production, these would be flagged as uncertain
                
                all_preds.append(pred.cpu().item())
                all_labels.append(y[i].item())
                all_probs.append(probs[i].cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Overall metrics
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate per-batch inference time statistics
    avg_inference_time = np.mean(inference_times)
    per_sample_time = avg_inference_time / test_loader.batch_size
    
    results = {
        'accuracy': accuracy,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1': overall_f1,
        'per_class_metrics': {
            label_names[i]: {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': int(support[i])
            }
            for i in range(len(label_names))
        },
        'confusion_matrix': cm.tolist(),
        'inference_time_ms': {
            'avg_batch_time': avg_inference_time,
            'avg_per_sample': per_sample_time,
            'min': np.min(inference_times),
            'max': np.max(inference_times)
        },
        'confidence_threshold': confidence_threshold,
        'rejected_samples': rejected_samples,
        'total_samples': len(all_labels)
    }
    
    return results, all_labels, all_preds, label_names


def plot_confusion_matrix(cm, label_names, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def print_evaluation_results(results):
    """Print formatted evaluation results."""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Precision: {results['overall_precision']:.4f}")
    print(f"  Recall:    {results['overall_recall']:.4f}")
    print(f"  F1-Score:  {results['overall_f1']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"  {'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("  " + "-"*65)
    for class_name, metrics in results['per_class_metrics'].items():
        print(f"  {class_name:<15} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} "
              f"{metrics['f1']:<12.4f} "
              f"{metrics['support']:<10}")
    
    print(f"\nInference Performance:")
    print(f"  Avg batch time:    {results['inference_time_ms']['avg_batch_time']:.2f} ms")
    print(f"  Avg per sample:    {results['inference_time_ms']['avg_per_sample']:.2f} ms")
    print(f"  Min batch time:    {results['inference_time_ms']['min']:.2f} ms")
    print(f"  Max batch time:    {results['inference_time_ms']['max']:.2f} ms")
    
    print(f"\nConfidence Threshold Analysis:")
    print(f"  Threshold:         {results['confidence_threshold']:.2f}")
    print(f"  Rejected samples:  {results['rejected_samples']}/{results['total_samples']}")
    print(f"  Rejection rate:    {results['rejected_samples']/results['total_samples']*100:.2f}%")
    
    print("="*70)


def main():
    """Main evaluation function."""
    # Configuration
    feature_base_dir = "../../dataset/processed/feature_extraction/cleaned"
    model_path = "../saved_models/cryingsense_cnn_best.pth"
    results_dir = "../../experiments/performance_reports"
    cm_dir = "../../experiments/confusion_matrices"
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)
    
    print("="*70)
    print("CryingSense Model Evaluation")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    # Load data
    print("\nLoading dataset...")
    file_list, label_map = get_file_list_and_labels(feature_base_dir)
    
    if not file_list:
        print("Error: No feature files found!")
        print(f"Looking in: {os.path.abspath(feature_base_dir)}")
        return
    
    label_names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]
    
    # For evaluation, use test split (last 10% of data)
    # In production, this should load from dataset_split.json
    _, test_files = train_test_split(
        file_list, test_size=0.10, random_state=42,
        stratify=[get_label_from_path(f) for f in file_list]
    )
    
    print(f"Test samples: {len(test_files)}")
    print(f"Classes: {label_names}")
    
    # Create dataset and loader
    test_dataset = CryingSenseDataset(test_files, label_map, feature_base_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, 
                            num_workers=2, pin_memory=True)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CryingSenseCNN(num_classes=len(label_map)).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
        print(f"Training accuracy: {checkpoint.get('train_acc', 'unknown'):.4f}")
        print(f"Validation accuracy: {checkpoint.get('val_acc', 'unknown'):.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Estimated size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Evaluate model
    results, all_labels, all_preds, label_names = evaluate_model(
        model, test_loader, device, label_names, confidence_threshold=0.6
    )
    
    # Print results
    print_evaluation_results(results)
    
    # Generate and save confusion matrix
    cm = np.array(results['confusion_matrix'])
    cm_path = os.path.join(cm_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, label_names, cm_path)
    
    # Save results to JSON
    results_path = os.path.join(results_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Generate detailed classification report
    report = classification_report(all_labels, all_preds, 
                                   target_names=label_names, digits=4)
    report_path = os.path.join(results_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("CryingSense Model - Classification Report\n")
        f.write("="*70 + "\n\n")
        f.write(report)
    print(f"Classification report saved to: {report_path}")
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
