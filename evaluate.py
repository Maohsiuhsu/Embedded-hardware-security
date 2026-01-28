#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from models import LimitedFake14x14FeatureLearner
from metrics import calculate_all_metrics


class Evaluator:
    def __init__(self, model_path, device='cuda', model_size='small'):
        self.device = device
        
        self.model = LimitedFake14x14FeatureLearner(
            num_classes=2,
            model_size=model_size,
            num_fake_types=4
        )
        
        checkpoint = torch.load(model_path, map_location=device)
        
        state_dict = checkpoint['model_state_dict']
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key.replace('_orig_mod.', '')
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        
        print(f"Model loaded: {model_path}")
        print(f"Validation Accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    
    def evaluate(self, test_loader):
        all_predictions = []
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for images, labels, identities, fake_types in test_loader:
                images = images.to(self.device)
                
                outputs, features, fake_type_features = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_scores.extend(probabilities[:, 1].cpu().numpy())
        
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_scores = np.array(all_scores)
        
        metrics = calculate_all_metrics(y_true, y_pred, y_scores)
        
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=['Fake', 'Real'],
            output_dict=True
        )
        
        cm = confusion_matrix(y_true, y_pred)
        
        results = {
            'metrics': metrics,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions,
            'labels': all_labels,
            'scores': all_scores
        }
        
        return results
    
    def evaluate_by_fake_type(self, test_loader):
        type_results = {}
        all_predictions = []
        all_labels = []
        all_fake_types = []
        all_scores = []
        
        with torch.no_grad():
            for images, labels, identities, fake_types in test_loader:
                images = images.to(self.device)
                outputs, _, _ = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_fake_types.extend(fake_types)
                all_scores.extend(probabilities[:, 1].cpu().numpy())
        
        unique_types = list(set(all_fake_types))
        
        print("\n" + "="*80)
        print("Evaluation Results by Fake Fingerprint Type")
        print("="*80)
        
        for fake_type in unique_types:
            type_indices = [i for i, ft in enumerate(all_fake_types) if ft == fake_type]
            
            if not type_indices:
                continue
            
            type_predictions = np.array([all_predictions[i] for i in type_indices])
            type_labels = np.array([all_labels[i] for i in type_indices])
            type_scores = np.array([all_scores[i] for i in type_indices])
            
            metrics = calculate_all_metrics(type_labels, type_predictions, type_scores)
            type_results[fake_type] = metrics
            
            print(f"\n{fake_type}:")
            print(f"  APCER: {metrics['APCER']*100:.2f}%")
            print(f"  BPCER: {metrics['BPCER']*100:.2f}%")
            print(f"  ACE: {metrics['ACE']*100:.2f}%")
            print(f"  Accuracy: {metrics['Accuracy']*100:.2f}%")
            if 'AUC' in metrics:
                print(f"  AUC: {metrics['AUC']:.4f}")
            print(f"  TP: {metrics['TP']}, TN: {metrics['TN']}, FP: {metrics['FP']}, FN: {metrics['FN']}")
        
        return type_results
    
    def print_summary(self, results):
        metrics = results['metrics']
        
        print("\n" + "="*80)
        print("Evaluation Summary")
        print("="*80)
        print(f"APCER: {metrics['APCER']*100:.2f}%")
        print(f"BPCER: {metrics['BPCER']*100:.2f}%")
        print(f"ACE: {metrics['ACE']*100:.2f}%")
        print(f"Accuracy: {metrics['Accuracy']*100:.2f}%")
        if 'AUC' in metrics:
            print(f"AUC: {metrics['AUC']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall: {metrics['Recall']:.4f}")
        print(f"F1-Score: {metrics['F1_Score']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {metrics['TP']}, TN: {metrics['TN']}")
        print(f"  FP: {metrics['FP']}, FN: {metrics['FN']}")
        print("="*80)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    evaluator = Evaluator(
        model_path='best_model.pth',
        device=device,
        model_size='small'
    )
    
    print("Evaluation script ready!")
    print("Please configure DataLoader according to actual dataset path")


if __name__ == "__main__":
    main()
