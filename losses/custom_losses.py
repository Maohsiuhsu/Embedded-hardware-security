import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparationLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, features, labels):
        real_features = features[labels == 1]
        fake_features = features[labels == 0]
        
        if len(real_features) > 1 and len(fake_features) > 1:
            real_center = real_features.mean(dim=0)
            fake_center = fake_features.mean(dim=0)
            
            real_intra = torch.norm(real_features - real_center, dim=1).mean()
            fake_intra = torch.norm(fake_features - fake_center, dim=1).mean()
            
            inter_distance = torch.norm(real_center - fake_center)
            
            separation_loss = (real_intra + fake_intra) / (inter_distance + 1e-8)
            return separation_loss
        else:
            return torch.tensor(0.0, device=features.device)


class ConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, features, identities):
        consistency_loss = 0
        unique_identities = list(set(identities))
        
        for identity in unique_identities:
            mask = torch.tensor([id == identity for id in identities], 
                              device=features.device)
            if mask.sum() > 1:
                identity_features = features[mask]
                center = identity_features.mean(dim=0)
                consistency_loss += torch.norm(identity_features - center, dim=1).mean()
        
        return consistency_loss / len(unique_identities) if unique_identities else \
               torch.tensor(0.0, device=features.device)


class FakeTypeLoss(nn.Module):
    def __init__(self, loss_type='mse'):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, fake_type_features, fake_type_labels=None):
        if fake_type_labels is None:
            target = torch.zeros_like(fake_type_features)
            return self.criterion(fake_type_features, target)
        else:
            return self.criterion(fake_type_features, fake_type_labels)


class CombinedLoss(nn.Module):
    def __init__(self, 
                 sep_weight=0.1,
                 cons_weight=0.1,
                 fake_type_weight=0.1):
        super().__init__()
        
        self.sep_weight = sep_weight
        self.cons_weight = cons_weight
        self.fake_type_weight = fake_type_weight
        
        self.cls_criterion = nn.CrossEntropyLoss()
        self.sep_loss = SeparationLoss()
        self.cons_loss = ConsistencyLoss()
        self.fake_type_loss = FakeTypeLoss()
    
    def forward(self, outputs, features, labels, identities=None, 
                fake_type_features=None, fake_type_labels=None):
        cls_loss = self.cls_criterion(outputs, labels)
        sep_loss = self.sep_loss(features, labels)
        
        cons_loss = torch.tensor(0.0, device=features.device)
        if identities is not None:
            cons_loss = self.cons_loss(features, identities)
        
        fake_type_loss = torch.tensor(0.0, device=features.device)
        if fake_type_features is not None:
            fake_mask = labels == 0
            if fake_mask.sum() > 0:
                fake_type_loss = self.fake_type_loss(
                    fake_type_features[fake_mask], 
                    fake_type_labels[fake_mask] if fake_type_labels is not None else None
                )
        
        total_loss = (cls_loss + 
                     self.sep_weight * sep_loss +
                     self.cons_weight * cons_loss +
                     self.fake_type_weight * fake_type_loss)
        
        loss_dict = {
            'cls_loss': cls_loss.item(),
            'sep_loss': sep_loss.item(),
            'cons_loss': cons_loss.item(),
            'fake_type_loss': fake_type_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
