import torch
import torch.nn as nn
from .backbone import FingerVIT_14x14


class LimitedFake14x14FeatureLearner(nn.Module):
    def __init__(self, num_classes=2, model_size='small', num_fake_types=4):
        super().__init__()
        
        if model_size == 'large':
            dims = [96, 192, 384]
            feature_dim = 384
            hidden_dim = 256
        else:
            dims = [40, 80, 160]
            feature_dim = 160
            hidden_dim = 128
        
        self.feature_extractor = FingerVIT_14x14(
            in_chans=1,
            img_size=112,
            num_classes=512,
            dims=dims,
            depths=[2, 4, 6],
            type=["repmix", "mhsa", "mhsa"],
            patch_size=4,
            distillation=False
        )
        
        self.feature_quality_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )
        
        self.num_fake_types = num_fake_types
        self.fake_type_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_fake_types)
        )
    
    def forward(self, x):
        features = self.feature_extractor.forward_feature(x)
        quality_features = self.feature_quality_head(features)
        output = self.classifier(quality_features)
        fake_type_features = self.fake_type_classifier(quality_features)
        return output, quality_features, fake_type_features


class MultiFake14x14FeatureLearner(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.feature_extractor = FingerVIT_14x14(
            in_chans=1,
            img_size=112,
            num_classes=512,
            dims=[96, 192, 384],
            depths=[2, 4, 6],
            type=["repmix", "mhsa", "mhsa"],
            patch_size=4,
            distillation=False
        )
        
        self.feature_quality_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
        self.fake_type_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        features = self.feature_extractor.forward_feature(x)
        quality_features = self.feature_quality_head(features)
        output = self.classifier(quality_features)
        fake_type_features = self.fake_type_classifier(quality_features)
        return output, quality_features, fake_type_features
