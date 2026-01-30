import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model.models.cnn_model import CryingSenseCNN
from train import get_file_list_and_labels, get_label_from_path, CryingSenseDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

if __name__ == "__main__":
    feature_dir = "../../dataset/processed/features"
    file_list, label_map = get_file_list_and_labels(feature_dir)
    test_files = file_list  # Use a separate test set if available
    test_dataset = CryingSenseDataset(test_files, label_map)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CryingSenseCNN(num_classes=len(label_map)).to(device)
    model.load_state_dict(torch.load("../saved_models/cryingsense_cnn.pth", map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            preds = out.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
