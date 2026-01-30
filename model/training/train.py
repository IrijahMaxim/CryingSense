import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model.models.cnn_model import CryingSenseCNN

# Custom Dataset for loading .npy feature files
def get_label_from_path(path):
    # Assumes path like .../class_name/xxx.npy
    return os.path.basename(os.path.dirname(path))

class CryingSenseDataset(Dataset):
    def __init__(self, file_list, label_map):
        self.file_list = file_list
        self.label_map = label_map
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        x = np.load(self.file_list[idx])
        x = torch.tensor(x, dtype=torch.float32)
        label_name = get_label_from_path(self.file_list[idx])
        y = self.label_map[label_name]
        return x, y

def get_file_list_and_labels(feature_dir):
    file_list = []
    for root, _, files in os.walk(feature_dir):
        for file in files:
            if file.endswith('.npy'):
                file_list.append(os.path.join(root, file))
    labels = sorted(list(set(get_label_from_path(f) for f in file_list)))
    label_map = {label: i for i, label in enumerate(labels)}
    return file_list, label_map

def train_model(model, train_loader, val_loader, device, epochs=30, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            _, pred = out.max(1)
            train_correct += (pred == y).sum().item()
            train_total += x.size(0)
        train_acc = train_correct / train_total
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                _, pred = out.max(1)
                val_correct += (pred == y).sum().item()
                val_total += x.size(0)
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}: Train Loss {train_loss/train_total:.4f}, Train Acc {train_acc:.4f}, Val Loss {val_loss/val_total:.4f}, Val Acc {val_acc:.4f}")
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "../saved_models/cryingsense_cnn.pth")
            print("Best model saved.")

if __name__ == "__main__":
    feature_dir = "../../dataset/processed/features"
    file_list, label_map = get_file_list_and_labels(feature_dir)
    train_files, val_files = train_test_split(file_list, test_size=0.15, random_state=42, stratify=[get_label_from_path(f) for f in file_list])
    train_dataset = CryingSenseDataset(train_files, label_map)
    val_dataset = CryingSenseDataset(val_files, label_map)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CryingSenseCNN(num_classes=len(label_map)).to(device)
    train_model(model, train_loader, val_loader, device)
