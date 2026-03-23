"""
CryingSense Model Validation Script

Evaluates the trained model on the VAL split (from dataset_split.json).
Use this after training to check in-distribution performance on held-out validation data.
Outputs: classification report + HTML confusion matrix in performance_reports/validation_report/

Pipeline order:
  1. python scripts/preprocess_audio.py
  2. python scripts/feature_extraction.py
  3. python scripts/dataset_split.py
  4. python model/training/train.py
  5. python model/training/validate.py    <- this script
  6. python model/training/evaluate.py
"""

import os
import sys
import json
import logging
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from model.models.cnn_model import CryingSenseCNN
from model.training.dataset import CryingSenseDataset, get_label_from_path
from sklearn.metrics import classification_report, confusion_matrix


def load_split_from_json(json_path, feature_base_dirs):
    """
    Load dataset split from JSON file.
    
    Args:
        json_path: Path to dataset_split.json
        feature_base_dirs: Dict mapping source names ('cleaned', 'raw') to base directories
        
    Returns:
        Dict with 'train', 'val', 'eval' keys containing lists of (mfcc_path, base_dir) tuples
    """
    with open(json_path, 'r') as f:
        split_data = json.load(f)
    
    result = {'train': [], 'val': [], 'eval': []}
    
    # Handle nested structure: {"splits": {"train": {"class_name": [files]}}}
    splits = split_data.get('splits', split_data)  # Support both formats
    
    for split_name in ['train', 'val', 'eval']:
        split_content = splits.get(split_name, {})
        
        # If split_content is a dict (organized by class), flatten it
        if isinstance(split_content, dict):
            for class_name, file_list in split_content.items():
                for entry in file_list:
                    # Entry format: "source:filename.npy"
                    if ':' in entry:
                        source, filename = entry.split(':', 1)
                    else:
                        source = 'cleaned'
                        filename = entry
                    
                    if source not in feature_base_dirs:
                        print(f"Warning: Unknown source '{source}' in split, skipping {entry}")
                        continue
                    
                    base_dir = feature_base_dirs[source]
                    mfcc_path = os.path.join(base_dir, 'mfcc', class_name, filename)
                    
                    if os.path.exists(mfcc_path):
                        result[split_name].append((mfcc_path, base_dir))
                    else:
                        print(f"Warning: Feature file not found: {mfcc_path}")
        else:
            # Legacy flat list format
            for entry in split_content:
                if ':' in entry:
                    source, filename = entry.split(':', 1)
                else:
                    source = 'cleaned'
                    filename = entry
                
                if source not in feature_base_dirs:
                    continue
                
                base_dir = feature_base_dirs[source]
                class_name = filename.rsplit('_', 1)[0] if '_' in filename else filename.replace('.npy', '')
                mfcc_path = os.path.join(base_dir, 'mfcc', class_name, filename)
                
                if os.path.exists(mfcc_path):
                    result[split_name].append((mfcc_path, base_dir))
    
    return result


def get_feature_file_list(feature_base_dir):
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

if __name__ == "__main__":
    # Configuration - use absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    
    # Feature directories for both cleaned and raw data
    feature_base_dirs = {
        'cleaned': os.path.join(project_root, 'dataset', 'processed', 'features', 'cleaned'),
        'raw': os.path.join(project_root, 'dataset', 'processed', 'features', 'raw')
    }
    model_path = os.path.join(project_root, 'model', 'saved_models', 'cryingsense_cnn_best.pth')
    validation_report_dir = os.path.join(project_root, 'performance_reports', 'validation_report')
    logs_dir = os.path.join(project_root, 'performance_reports', 'logs')
    split_json_path = os.path.join(project_root, 'dataset', 'dataset_split.json')
    
    # Create directories
    os.makedirs(validation_report_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(logs_dir, f'validate_{timestamp}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("CryingSense CNN Validation")
    logger.info("="*60)
    
    print("="*60)
    print("CryingSense CNN Validation")
    print("="*60)
    
    # Try to load from dataset_split.json first
    if os.path.exists(split_json_path):
        print(f"Loading val set from: {split_json_path}")
        splits = load_split_from_json(split_json_path, feature_base_dirs)
        val_files = splits['val']
        
        if not val_files:
            print("Error: No validation files found in dataset_split.json!")
            print("Please run: python scripts/dataset_split.py")
            sys.exit(1)
        
        # Build label map from validation files
        labels = sorted(list(set(get_label_from_path(f[0]) for f in val_files)))
        label_map = {label: i for i, label in enumerate(labels)}
        
        print(f"Loaded from JSON - Val samples: {len(val_files)}")
        print(f"Classes: {list(label_map.keys())}")
    else:
        print("Error: dataset_split.json not found!")
        print("Dataset splitting is handled by a separate script.")
        print("Please run: python scripts/dataset_split.py")
        sys.exit(1)
    
    print("="*60)
    
    # Load model
    print("Loading model...")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("\nPlease train the model first:")
        print("  python model/training/train.py")
        sys.exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CryingSenseCNN(num_classes=len(label_map)).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        # Initialize the model with a dummy forward pass to create _fc1 layer
        dummy_input = torch.randn(1, 4, 128, 216).to(device)
        _ = model(dummy_input)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Training accuracy: {checkpoint.get('train_acc', 0):.4f}")
        print(f"Validation accuracy: {checkpoint.get('val_acc', 0):.4f}")
    else:
        # Initialize the model with a dummy forward pass to create _fc1 layer
        dummy_input = torch.randn(1, 4, 128, 216).to(device)
        _ = model(dummy_input)
        
        model.load_state_dict(checkpoint)
    
    print(f"Device: {device}")
    print("="*60)
    
    # Create validation dataset and loader
    # Note: num_workers=0 for Windows compatibility
    use_pin_memory = torch.cuda.is_available()
    val_dataset = CryingSenseDataset(val_files, label_map, feature_base_dirs)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=use_pin_memory)
    
    # Evaluate
    print("Evaluating model...")
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validating"):
            x = x.to(device)
            out = model(x)
            preds = out.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    
    print("\n" + "="*60)
    print("Validation Results")
    print("="*60)
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=list(label_map.keys()))
    print(report)
    
    # Save validation classification report
    report_path = os.path.join(validation_report_dir, 'validation_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("CryingSense Model - Validation Classification Report\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    print(f"\nValidation report saved to: {report_path}")
    
    # Calculate and log accuracy
    from sklearn.metrics import accuracy_score
    val_accuracy = accuracy_score(all_labels, all_preds)
    logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
    logger.info(f"Report saved to: {report_path}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    print("="*60)

    # --- HTML Confusion Matrix ---
    class_names = list(label_map.keys())
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalised (recall)

    # Per-class F1 from the report dict
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True
    )

    def _cell_style(row_idx, col_idx, value):
        """Return inline CSS background colour for a confusion matrix cell."""
        if row_idx == col_idx:                         # diagonal → green
            intensity = int(255 * (1 - value * 0.6))   # deeper green for higher %
            return f"background-color: rgb({intensity}, 210, {intensity}); font-weight: bold;"
        elif value > 0:                                # off-diagonal non-zero → pink
            intensity = int(255 * (1 - value * 0.5))
            return f"background-color: rgb(255, {intensity}, {intensity});"
        else:
            return "background-color: #ffffff;"

    rows_html = ""
    for r_idx, r_name in enumerate(class_names):
        cells = ""
        for c_idx in range(len(class_names)):
            pct = cm_norm[r_idx, c_idx]
            label = f"{pct*100:.1f}%" if pct > 0 else "0%"
            style = _cell_style(r_idx, c_idx, pct)
            cells += f'<td style="{style} padding:10px 14px; text-align:center;">{label}</td>'
        rows_html += (
            f'<tr><td style="padding:10px 14px; font-weight:bold; '
            f'text-align:left;">{r_name.upper()}</td>{cells}</tr>\n'
        )

    # F1 score footer row
    f1_cells = ""
    for c_name in class_names:
        f1 = report_dict.get(c_name, {}).get("f1-score", 0.0)
        f1_cells += (
            f'<td style="padding:10px 14px; text-align:center; font-weight:bold; '
            f'background-color:#f0f0f0;">{f1:.2f}</td>'
        )

    header_cells = "".join(
        f'<th style="padding:10px 14px; text-align:center; '
        f'background-color:#2d2d2d; color:#ffffff;">{n.upper()}</th>'
        for n in class_names
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CryingSense – Validation Confusion Matrix</title>
  <style>
    body {{ font-family: Arial, sans-serif; padding: 30px; background: #fafafa; }}
    h2 {{ color: #2d2d2d; }}
    table {{ border-collapse: collapse; margin-top: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.12); }}
    th, td {{ border: 1px solid #e0e0e0; }}
    thead th:first-child {{ background-color: #2d2d2d; }}
    .corner {{ background-color: #2d2d2d; }}
    .row-header {{ background-color: #f5f5f5; color: #2d2d2d; }}
    .f1-label {{ background-color: #f0f0f0; font-weight: bold; padding: 10px 14px; text-align: left; }}
  </style>
</head>
<body>
  <h2>CryingSense Model – Validation Confusion Matrix</h2>
  <p style="color:#555;">Rows = actual class &nbsp;|&nbsp; Columns = predicted class &nbsp;|&nbsp; Values = row-normalised %</p>
  <table>
    <thead>
      <tr>
        <th class="corner" style="padding:10px 14px;"></th>
        {header_cells}
      </tr>
    </thead>
    <tbody>
      {rows_html}
      <tr>
        <td class="f1-label">F1 SCORE</td>
        {f1_cells}
      </tr>
    </tbody>
  </table>
  <p style="margin-top:20px; color:#888; font-size:0.85em;">
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  </p>
</body>
</html>
"""

    cm_html_path = os.path.join(validation_report_dir, 'validation_confusion_matrix.html')
    with open(cm_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Confusion matrix saved to: {cm_html_path}")

    logger.info("Validation Complete")
