"""
This function is adapted from [aLLM4TS] by [bian2024multi]
Original source: [https://github.com/yxbian23/aLLM4TS]
"""

import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

########################################
# Utility: Dataset
########################################
class ReconstructDataset(Dataset):
    def __init__(self, data, window_size, stride):
        self.samples = []
        for i in range(0, len(data) - window_size + 1, stride):
            self.samples.append(data[i:i + window_size])
        self.samples = np.stack(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], 0

########################################
# Utility: EarlyStopping
########################################
class EarlyStoppingTorch:
    def __init__(self, path=None, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path
        self.best_model = None

    def __call__(self, score, model):
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self.best_model = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

########################################
# Utility: Learning Rate Adjustment
########################################
def adjust_learning_rate(optimizer, epoch, lradj, base_lr):
    if lradj == 'type1':
        lr = base_lr * (0.5 ** ((epoch - 1) // 1))
    elif lradj == "cosine":
        lr = base_lr / 2 * (1 + math.cos(epoch / 10 * math.pi))
    else:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f"[LR] Epoch {epoch}: {lr:.6f}")

########################################
# Allm4ts Model (Minimal Dummy Version)
########################################
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(config['input_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['input_dim'])
        )

    def forward(self, x):
        return self.backbone(x)

########################################
# ALLM4TS Wrapper
########################################
class ALLM4TS:
    def __init__(self,
                 win_size=100,
                 stride=1,
                 enc_in=1,
                 features='M',
                 batch_size=128,
                 learning_rate=1e-4,
                 epochs=10,
                 patience=3,
                 lradj='type1',
                 validation_size=0.2,
                 anomaly_ratio=1.0,
                 model_config={}):
        self.win_size = win_size
        self.stride = stride
        self.enc_in = enc_in
        self.features = features
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.lradj = lradj
        self.validation_size = validation_size
        self.anomaly_ratio = anomaly_ratio

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model(model_config).float().to(self.device)
        self.model_optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.early_stopping = EarlyStoppingTorch(patience=self.patience)

    def fit(self, data):
        tsTrain = data[:int((1 - self.validation_size) * len(data))]
        tsValid = data[int((1 - self.validation_size) * len(data)):] 
        
        train_loader = DataLoader(ReconstructDataset(tsTrain, self.win_size, self.stride), batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(ReconstructDataset(tsValid, self.win_size, self.stride), batch_size=self.batch_size, shuffle=False)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0
            for batch_x, _ in tqdm(train_loader, desc=f"[Train] Epoch {epoch}"):
                batch_x = batch_x.float().to(self.device)
                self.model_optim.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_x)
                loss.backward()
                self.model_optim.step()
                total_loss += loss.item()

            val_losses = []
            self.model.eval()
            with torch.no_grad():
                for batch_x, _ in valid_loader:
                    batch_x = batch_x.float().to(self.device)
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_x)
                    val_losses.append(loss.item())

            val_loss = np.mean(val_losses)
            print(f"Epoch {epoch} - Train Loss: {total_loss:.6f}, Val Loss: {val_loss:.6f}")
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered.")
                break
            adjust_learning_rate(self.model_optim, epoch, self.lradj, self.learning_rate)

        self.model.load_state_dict(self.early_stopping.best_model)

    def decision_function(self, data):
        test_loader = DataLoader(ReconstructDataset(data, self.win_size, self.stride), batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        mse = nn.MSELoss(reduction='none')
        scores = []
        with torch.no_grad():
            for batch_x, _ in tqdm(test_loader, desc="[Test] Scoring"):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x)
                score = torch.mean(mse(batch_x, outputs), dim=-1)
                scores.append(score.detach().cpu().numpy()[:, -1])

        scores = np.concatenate(scores, axis=0)
        final_scores = np.zeros(len(data))
        count = np.zeros(len(data))
        for i, s in enumerate(scores):
            start = i * self.stride
            end = start + self.win_size
            final_scores[start:end] += s
            count[start:end] += 1

        # self.decision_scores_ = final_scores / np.maximum(count, 1)
        # Prevent NaN values in decision function by replacing NaN with zero
        self.decision_scores_ = np.nan_to_num(final_scores / np.maximum(count, 1), nan=0.0)

        return self.decision_scores_

    def evaluate(self, data, labels):
        scores = self.decision_function(data)
        threshold = np.percentile(scores, 100 - self.anomaly_ratio)
        preds = (scores > threshold).astype(int)
        gt = labels.astype(int)
        accuracy = accuracy_score(gt, preds)
        precision, recall, f_score, _ = precision_recall_fscore_support(gt, preds, average='binary')
        print(f"[Eval] Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f_score:.4f}")
        return accuracy, precision, recall, f_score
