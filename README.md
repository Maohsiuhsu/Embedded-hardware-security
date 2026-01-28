# Fingerprint Anti-Spoofing Detection Project - Reviewer Key Code

This directory contains core code that reviewers will focus on, demonstrating key design and implementation of the research.

## Directory Structure

```
github/
├── models/
│   ├── __init__.py
│   ├── backbone.py
│   └── feature_learner.py
├── losses/
│   ├── __init__.py
│   └── custom_losses.py
├── configs/
│   ├── sgm.yaml
│   ├── mgm.yaml
│   ├── ablation_k4.yaml
│   └── cross_dataset.yaml
├── metrics.py
├── evaluate.py
├── train.py
├── test.py
└── README.md
```

## Key Design

### 1. Backbone: FingerVIT 14x14

**Location**: `models/backbone.py`

- **Architecture**: CNN/ViT Hybrid
  - Early stages: RepMix (Convolutional Mixing)
  - Later stages: MHSA (Multi-Head Self-Attention)
- **Features**:
  - Structural Reparameterization
  - Final feature map resolution: 14×14
  - Supports Small/Large two scales

**Key Components**:
- `StemLayer`: Initial feature extraction
- `PatchMerging`: Spatial downsampling and channel expansion
- `Block`: Basic building block (RepMix/MHSA + FFN)
- `Stage`: Combination of multiple Blocks

### 2. Key Design Modules

**Location**: `models/feature_learner.py`

#### 2.1 Feature Quality Head
- Processes 14×14 feature maps
- Performs feature quality optimization and dimensionality reduction
- Output: 64-dimensional quality features

#### 2.2 Multi-classifier
- **Anti-spoofing Classifier**: Real/Fake fingerprint binary classification
- **Fake Type Classifier**: Distinguishes different types of fake fingerprints (DDIM, GAN, CycleGAN, etc.)

#### 2.3 Model Variants
- `LimitedFake14x14FeatureLearner`: Limited fake fingerprint types (for generalization experiments)
- `MultiFake14x14FeatureLearner`: Multiple fake fingerprint types

### 3. Custom Loss Functions

**Location**: `losses/custom_losses.py`

#### 3.1 Separation Loss
- **Objective**: Increase inter-class distance, decrease intra-class distance
- **Formula**: `L_sep = (intra_real + intra_fake) / (inter_distance + ε)`

#### 3.2 Consistency Loss
- **Objective**: Different samples from the same person should have similar features
- **Formula**: `L_cons = mean(||f_i - f_center||)`

#### 3.3 Fake Type Loss
- **Objective**: Learn to distinguish different types of fake fingerprints

#### 3.4 Combined Loss
- **Total Loss**: `L_total = L_cls + λ1*L_sep + λ2*L_cons + λ3*L_fake_type`

### 4. Evaluation Metrics

**Location**: `metrics.py`

#### 4.1 APCER (Attack Presentation Classification Error Rate)
- **Formula**: `APCER = FP / (TN + FP)`
- **Meaning**: False positive rate - proportion of real fingerprints misclassified as fake

#### 4.2 BPCER (Bona Fide Presentation Classification Error Rate)
- **Formula**: `BPCER = FN / (TP + FN)`
- **Meaning**: False negative rate - proportion of fake fingerprints misclassified as real

#### 4.3 ACE (Average Classification Error)
- **Formula**: `ACE = (APCER + BPCER) / 2`
- **Meaning**: Average classification error rate

#### 4.4 AUC (Area Under Curve)
- **Formula**: Area under ROC curve
- **Meaning**: Overall classification performance

### 5. Configuration Files

**Location**: `configs/`

- **sgm.yaml**: Small model configuration
- **mgm.yaml**: Large model configuration
- **ablation_k4.yaml**: Ablation study with k=4 fake types
- **cross_dataset.yaml**: Cross-dataset evaluation configuration

## Usage

### Training

```python
from models import LimitedFake14x14FeatureLearner
from train import Trainer

model = LimitedFake14x14FeatureLearner(
    num_classes=2,
    model_size='small',
    num_fake_types=4
)

trainer = Trainer(model, device='cuda')
trainer.train(train_loader, val_loader, epochs=50)
```

### Testing

```python
from test import Tester

tester = Tester(
    model_path='best_model.pth',
    device='cuda',
    model_size='small'
)

results = tester.test(test_loader)
tester.print_summary(results)

type_results = tester.test_by_fake_type(test_loader)
```

### Evaluation

```python
from evaluate import Evaluator
from metrics import calculate_all_metrics

evaluator = Evaluator(
    model_path='best_model.pth',
    device='cuda',
    model_size='small'
)

results = evaluator.evaluate(test_loader)
evaluator.print_summary(results)

type_results = evaluator.evaluate_by_fake_type(test_loader)
```

### Metrics

```python
from metrics import calculate_all_metrics, calculate_apcer, calculate_bpcer, calculate_ace, calculate_auc

y_true = [0, 1, 0, 1, 0]
y_pred = [0, 1, 1, 1, 0]
y_scores = [0.1, 0.9, 0.6, 0.8, 0.2]

metrics = calculate_all_metrics(y_true, y_pred, y_scores)
print(f"APCER: {metrics['APCER']*100:.2f}%")
print(f"BPCER: {metrics['BPCER']*100:.2f}%")
print(f"ACE: {metrics['ACE']*100:.2f}%")
print(f"AUC: {metrics['AUC']:.4f}")
```

## Reviewer Focus Points

### 1. Backbone Design
- CNN/ViT Hybrid architecture
- Structural reparameterization technique
- 14×14 feature map design

### 2. Key Design Modules
- Feature Quality Head
- Multi-classifier design
- Feature fusion strategy

### 3. Loss Function Design
- Rationality of custom loss functions
- Loss weight settings
- Multi-task learning strategy

### 4. Training Process
- Completeness of training script
- Optimizer and learning rate scheduling
- Model saving and loading

### 5. Testing and Evaluation
- Completeness of test script
- Generalization evaluation (by fake fingerprint type)
- Metric calculation and reporting
- **USENIX Focus**: Not just accuracy - comprehensive metrics including APCER, BPCER, ACE, AUC

## Experimental Design

### Generalization Experiment
- **Training Set**: Contains only 4 types of fake fingerprints (DDIM, guided, cycleGan, StyleGAN)
- **Test Set**: Contains all fake fingerprint types (including unseen types)
- **Objective**: Prove model's generalization ability to unseen fake fingerprint types

### Model Scale
- **Small**: dims=[40, 80, 160], fewer parameters
- **Large**: dims=[96, 192, 384], more parameters, higher accuracy

## Notes

1. This directory is a simplified version, showcasing core design and key code
2. Actual usage requires configuring DataLoader according to dataset path
3. Training and test scripts provide complete framework, can be extended as needed

## Related Files

- Full project code is located in parent directory
- Training logs and model weights please refer to project root directory
