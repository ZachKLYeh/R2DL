import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from ksvd import ApproximateKSVD
from model import R2DL
from dataset import AMPDataset, AAPDataset

import warnings
warnings.filterwarnings("ignore")

def train_cv(model, dataset, n_epoch, batch_size, k_folds, learning_rate, ksvd):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if k_folds is not None:
        kfold = KFold(n_splits = k_folds, shuffle = True)
    else:
        kfold = KFold(n_splits = 1, shuffle = True)

    fold_acc = []
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f"fold: {fold + 1}")
        model.reset_parameters()
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        train_loader = torch.utils.data.DataLoader(
                          dataset,
                          batch_size=batch_size, sampler=train_subsampler)
        val_loader = torch.utils.data.DataLoader(
                          dataset,
                          batch_size=batch_size, sampler=test_subsampler)
        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        for epoch in range(n_epoch):
            print(f"epoch:{epoch+1}")
            train_correct = 0
            train_loss = 0
            if ksvd =="epoch":
                model.ksvd()
            model.to(device)
            model.train()
            for batch in train_loader:
                if ksvd == "batch":
                    model.ksvd()
                model.to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optim.step()
                optim.zero_grad()
                _, pred = torch.max(outputs, 1)
                _, labels = torch.max(labels, 1)
                train_correct += (pred == labels).sum().item()
                train_loss += loss

            train_size = len(train_loader) * batch_size
            train_acc = train_correct / train_size
            train_loss = train_loss / train_size
            print(f"train_acc:{train_acc:.3f}, train_loss:{train_loss:.3f}, ", end='')
            
            with torch.no_grad():
                model.eval()
                val_correct = 0
                val_loss = 0
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    _, pred = torch.max(outputs, 1)
                    _, labels = torch.max(labels, 1)
                    val_correct += (pred == labels).sum().item()
                    val_loss += loss

                val_size = len(val_loader) * batch_size
                val_acc = val_correct / val_size
                val_loss = val_loss / val_size
                print(f"test_acc:{val_acc:.3f}, test_loss:{val_loss:.3f}")

            print()

        fold_acc.append(val_acc)
        print()

    kfold_avg_acc = sum(fold_acc) / len(fold_acc)
    print(f"kfold_avg_test_acc: {kfold_avg_acc}")

def main():
    n_epoch = 1
    k_folds = 5
    batch_size = 64 
    learning_rate = 0.001
    ksvd = "batch"

    model = R2DL()
    dataset = AMPDataset()
    train_cv(model, dataset, n_epoch, batch_size, k_folds, learning_rate, ksvd)


import multiprocessing

if __name__ == '__main__':
    # use 
    # $OMP_NUM_THREADS=4 python3 train.py
    # to train this model
    p = multiprocessing.pool.ThreadPool(processes=1)
    p = p.map(main(), [])

