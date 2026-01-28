#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from models import LimitedFake14x14FeatureLearner
from losses import CombinedLoss


class Trainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=0.001, 
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=50
        )
        
        self.criterion = CombinedLoss(
            sep_weight=0.1,
            cons_weight=0.1,
            fake_type_weight=0.1
        )
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels, identities, fake_types) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs, features, fake_type_features = self.model(images)
            
            loss, loss_dict = self.criterion(
                outputs, features, labels,
                identities=identities,
                fake_type_features=fake_type_features
            )
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, identities, fake_types in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs, features, fake_type_features = self.model(images)
                
                loss, _ = self.criterion(
                    outputs, features, labels,
                    identities=identities,
                    fake_type_features=fake_type_features
                )
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=50):
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'history': self.history
                }, 'best_model.pth')
            
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                print(f'  Best Val Acc: {best_val_acc:.2f}%')
                print()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LimitedFake14x14FeatureLearner(
        num_classes=2,
        model_size='small',
        num_fake_types=4
    )
    
    trainer = Trainer(model, device)
    
    print("Training script ready!")
    print("Please configure DataLoader according to actual dataset path")


if __name__ == "__main__":
    main()
