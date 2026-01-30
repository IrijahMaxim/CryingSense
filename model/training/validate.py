import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model.models.cnn_model import CryingSenseCNN
from train import get_file_list_and_labels, get_label_from_path, CryingSenseDataset
from sklearn.metrics import classification_report, confusion_matrix

if __name__ == "__main__":
    feature_dir = "../../dataset/processed/features"
    file_list, label_map = get_file_list_and_labels(feature_dir)
    val_files = file_list  # Use all files for evaluation or split as needed
    val_dataset = CryingSenseDataset(val_files, label_map)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CryingSenseCNN(num_classes=len(label_map)).to(device)
    model.load_state_dict(torch.load("../saved_models/cryingsense_cnn.pth", map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            out = model(x)
            preds = out.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=list(label_map.keys())))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
