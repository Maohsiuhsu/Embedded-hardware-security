#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from models import LimitedFake14x14FeatureLearner


class Tester:
    def __init__(self, model_path, device='cuda', model_size='small'):
        self.device = device
        
        self.model = LimitedFake14x14FeatureLearner(
            num_classes=2,
            model_size=model_size,
            num_fake_types=4
        )
        
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        print(f"Model loaded successfully: {model_path}")
        print(f"  Validation Accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    
    def test(self, test_loader):
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_features = []
        
        with torch.no_grad():
            for images, labels, identities, fake_types in test_loader:
                images = images.to(self.device)
                
                outputs, features, fake_type_features = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_features.extend(features.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        
        report = classification_report(
            all_labels, 
            all_predictions, 
            target_names=['Fake', 'Real'],
            output_dict=True
        )
        
        cm = confusion_matrix(all_labels, all_predictions)
        
        results = {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'features': all_features,
            'classification_report': report,
            'confusion_matrix': cm
        }
        
        return results
    
    def test_by_fake_type(self, test_loader):
        type_results = {}
        all_predictions = []
        all_labels = []
        all_fake_types = []
        
        with torch.no_grad():
            for images, labels, identities, fake_types in test_loader:
                images = images.to(self.device)
                outputs, _, _ = self.model(images)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_fake_types.extend(fake_types)
        
        unique_types = list(set(all_fake_types))
        
        print("\n" + "="*60)
        print("Results Analysis by Fake Fingerprint Type (Generalization Evaluation)")
        print("="*60)
        
        for fake_type in unique_types:
            type_indices = [i for i, ft in enumerate(all_fake_types) if ft == fake_type]
            
            if not type_indices:
                continue
            
            type_predictions = [all_predictions[i] for i in type_indices]
            type_labels = [all_labels[i] for i in type_indices]
            
            correct = sum(1 for p, l in zip(type_predictions, type_labels) if p == l)
            total = len(type_indices)
            accuracy = correct / total * 100
            
            type_results[fake_type] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }
            
            print(f"{fake_type}:")
            print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
            print()
        
        return type_results
    
    def print_summary(self, results):
        print("\n" + "="*60)
        print("Test Results Summary")
        print("="*60)
        print(f"Overall Accuracy: {results['accuracy']*100:.2f}%")
        print("\nClassification Report:")
        report = results['classification_report']
        print(f"  Fake - Precision: {report['Fake']['precision']:.4f}")
        print(f"  Fake - Recall: {report['Fake']['recall']:.4f}")
        print(f"  Fake - F1-Score: {report['Fake']['f1-score']:.4f}")
        print(f"  Real - Precision: {report['Real']['precision']:.4f}")
        print(f"  Real - Recall: {report['Real']['recall']:.4f}")
        print(f"  Real - F1-Score: {report['Real']['f1-score']:.4f}")
        print("\nConfusion Matrix:")
        cm = results['confusion_matrix']
        print(f"  [Fake Correct] [Fake Misclassified]")
        print(f"  [{cm[0,0]:6d}] [{cm[0,1]:10d}]")
        print(f"  [Real Misclassified] [Real Correct]")
        print(f"  [{cm[1,0]:6d}] [{cm[1,1]:10d}]")
        print("="*60)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tester = Tester(
        model_path='best_model.pth',
        device=device,
        model_size='small'
    )
    
    print("Test script ready!")
    print("Please configure DataLoader according to actual dataset path")


if __name__ == "__main__":
    main()
