# 🛡️ Week 2: Advanced AI Attacks + Risk Engine - Complete Walkthrough

## 📋 Overview

**What You're Building:** Advanced attack demonstrations + production-ready risk assessment framework

**Week 2 Goals:**
- ✅ Model Extraction (steal model via API queries)
- ✅ Data Poisoning (corrupt training data)
- ✅ Model Inversion (extract training data features)
- ✅ AI Risk Scoring Engine
- ✅ Cloud attack surface maps (AWS/Azure/GCP)
- ✅ Case studies documentation
- ✅ AI Risk Assessment Template

**Time:** 10-12 hours  
**Prerequisites:** Week 1 completed

---

## 🚀 Step-by-Step Implementation

### **Step 1: Project Structure Update (10 minutes)**

```bash
# Navigate to your project
cd module1-threat-lab-multicloud

# Create new directories for Week 2
mkdir -p notebooks/week2
mkdir -p risk-engine
mkdir -p cloud/aws
mkdir -p cloud/azure
mkdir -p cloud/gcp
mkdir -p docs/case_studies
mkdir -p diagrams/week2
mkdir -p models/extracted
mkdir -p models/poisoned

# Update requirements for Week 2
cat >> requirements.txt << 'EOF'

# Week 2 additions
cleverhans>=4.0.0  # Additional adversarial library
lime>=0.2.0  # Model interpretability
shap>=0.42.0  # SHAP values
tqdm>=4.65.0  # Progress bars
pyyaml>=6.0  # YAML for risk engine
tabulate>=0.9.0  # Nice table formatting
EOF

pip install -r requirements.txt
```

---

### **Step 2: Model Extraction Attack (90 minutes)**

Create `notebooks/week2/03_model_extraction.ipynb`:

```python
"""
Model Extraction Attack
Steal a model by querying it repeatedly and training a surrogate
"""

# ============================================================
# CELL 1: Imports and Setup
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import os

torch.manual_seed(42)
np.random.seed(42)

print("✓ Model Extraction Attack Setup")

# ============================================================
# CELL 2: Load Victim Model
# ============================================================

class SimpleConvNet(nn.Module):
    """Victim model architecture"""
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Load victim model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
victim_model = SimpleConvNet().to(device)

checkpoint = torch.load('../../models/baseline_mnist_cnn.pth', map_location=device)
victim_model.load_state_dict(checkpoint['model_state_dict'])
victim_model.eval()

print(f"✓ Victim model loaded")
print(f"  Victim accuracy: {checkpoint['test_accuracy']:.2f}%")

# ============================================================
# CELL 3: Define Surrogate Model (Different Architecture)
# ============================================================

class SurrogateModel(nn.Module):
    """
    Surrogate model with DIFFERENT architecture
    Simpler than victim to show extraction works with less capacity
    """
    def __init__(self):
        super(SurrogateModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

surrogate_model = SurrogateModel().to(device)

# Count parameters
victim_params = sum(p.numel() for p in victim_model.parameters())
surrogate_params = sum(p.numel() for p in surrogate_model.parameters())

print(f"\n✓ Surrogate model initialized")
print(f"  Victim parameters: {victim_params:,}")
print(f"  Surrogate parameters: {surrogate_params:,}")
print(f"  Parameter reduction: {(1 - surrogate_params/victim_params)*100:.1f}%")

# ============================================================
# CELL 4: Generate Synthetic Query Data
# ============================================================

def generate_synthetic_queries(num_queries, img_shape=(1, 28, 28)):
    """
    Generate synthetic data to query the victim model
    
    Strategies:
    1. Random noise
    2. Adversarial examples
    3. Data augmentation of limited samples
    """
    
    # Strategy 1: Random uniform noise
    random_queries = torch.rand(num_queries, *img_shape)
    
    # Normalize to MNIST range
    random_queries = (random_queries - 0.5) / 0.5
    
    return random_queries

# Generate queries
num_queries = 10000  # In reality, could be 100k+
print(f"\nGenerating {num_queries:,} synthetic queries...")

synthetic_data = generate_synthetic_queries(num_queries)

print(f"✓ Synthetic data generated")
print(f"  Shape: {synthetic_data.shape}")

# Visualize some synthetic queries
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img = synthetic_data[i].squeeze().numpy()
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Synthetic {i+1}')
    ax.axis('off')
plt.suptitle('Synthetic Query Samples (Random Noise)')
plt.tight_layout()
plt.savefig('../../diagrams/week2/synthetic_queries.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# CELL 5: Query Victim Model (Simulate API Calls)
# ============================================================

def query_victim_model(model, data, batch_size=64):
    """
    Query victim model to get predictions
    Simulates API calls to a deployed model
    """
    model.eval()
    predictions = []
    
    dataloader = DataLoader(
        TensorDataset(data),
        batch_size=batch_size,
        shuffle=False
    )
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Querying victim"):
            batch_data = batch[0].to(device)
            output = model(batch_data)
            
            # Get soft labels (probabilities)
            probs = torch.softmax(output, dim=1)
            predictions.append(probs.cpu())
    
    return torch.cat(predictions, dim=0)

print("\n" + "="*60)
print("Querying Victim Model (Simulating API Calls)")
print("="*60)

# Query the victim
victim_predictions = query_victim_model(victim_model, synthetic_data)

print(f"\n✓ Collected {len(victim_predictions):,} predictions from victim")
print(f"  Prediction shape: {victim_predictions.shape}")
print(f"  Sample prediction: {victim_predictions[0]}")

# ============================================================
# CELL 6: Train Surrogate Model
# ============================================================

def train_surrogate(surrogate, synthetic_data, victim_predictions, epochs=10):
    """
    Train surrogate model to mimic victim predictions
    Uses knowledge distillation with soft labels
    """
    
    # Create dataset
    dataset = TensorDataset(synthetic_data, victim_predictions)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Setup training
    optimizer = optim.Adam(surrogate.parameters(), lr=0.001)
    criterion = nn.KLDivLoss(reduction='batchmean')  # Knowledge distillation loss
    
    surrogate.train()
    
    train_losses = []
    
    print("\n" + "="*60)
    print("Training Surrogate Model")
    print("="*60)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_data, batch_labels in tqdm(train_loader, 
                                              desc=f"Epoch {epoch+1}/{epochs}",
                                              leave=False):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = surrogate(batch_data)
            
            # Knowledge distillation loss
            log_probs = torch.log_softmax(output, dim=1)
            loss = criterion(log_probs, batch_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    return train_losses

# Train surrogate
train_losses = train_surrogate(
    surrogate_model,
    synthetic_data,
    victim_predictions,
    epochs=10
)

print("\n✓ Surrogate training complete")

# ============================================================
# CELL 7: Evaluate Extraction Success
# ============================================================

# Load test data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(
    root='../../data',
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def evaluate_model(model, test_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return 100. * correct / total

print("\n" + "="*60)
print("Evaluating Extraction Attack Success")
print("="*60)

victim_accuracy = evaluate_model(victim_model, test_loader, device)
surrogate_accuracy = evaluate_model(surrogate_model, test_loader, device)

print(f"\nVictim Model Accuracy: {victim_accuracy:.2f}%")
print(f"Surrogate Model Accuracy: {surrogate_accuracy:.2f}%")
print(f"Extraction Success Rate: {(surrogate_accuracy/victim_accuracy)*100:.1f}%")

# Calculate agreement between models
def calculate_agreement(model1, model2, test_loader, device):
    """Calculate how often two models agree on predictions"""
    model1.eval()
    model2.eval()
    
    agreements = 0
    total = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            
            output1 = model1(data)
            output2 = model2(data)
            
            pred1 = output1.max(1)[1]
            pred2 = output2.max(1)[1]
            
            agreements += pred1.eq(pred2).sum().item()
            total += data.size(0)
    
    return 100. * agreements / total

agreement = calculate_agreement(victim_model, surrogate_model, test_loader, device)
print(f"\nModel Agreement: {agreement:.2f}%")
print(f"(How often they make the same prediction)")

# ============================================================
# CELL 8: Visualize Results
# ============================================================

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses)+1), train_losses, 'b-o', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('KL Divergence Loss', fontsize=12)
plt.title('Surrogate Model Training - Knowledge Distillation Loss', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../../diagrams/week2/extraction_training_loss.png', dpi=150, bbox_inches='tight')
plt.show()

# Compare predictions on sample images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

test_iter = iter(test_loader)
sample_data, sample_labels = next(test_iter)
sample_data = sample_data.to(device)

victim_model.eval()
surrogate_model.eval()

with torch.no_grad():
    victim_preds = victim_model(sample_data[:10]).max(1)[1].cpu().numpy()
    surrogate_preds = surrogate_model(sample_data[:10]).max(1)[1].cpu().numpy()

for i in range(10):
    ax = axes[i // 5, i % 5]
    img = sample_data[i].cpu().squeeze().numpy()
    ax.imshow(img, cmap='gray')
    
    true_label = sample_labels[i].item()
    victim_pred = victim_preds[i]
    surrogate_pred = surrogate_preds[i]
    
    match = "✓" if victim_pred == surrogate_pred else "✗"
    
    ax.set_title(f'True: {true_label}\nVictim: {victim_pred}\nSurr: {surrogate_pred} {match}',
                fontsize=9)
    ax.axis('off')

plt.suptitle('Model Extraction: Prediction Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('../../diagrams/week2/extraction_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# CELL 9: Save Extracted Model
# ============================================================

os.makedirs('../../models/extracted', exist_ok=True)

torch.save({
    'model_state_dict': surrogate_model.state_dict(),
    'surrogate_accuracy': surrogate_accuracy,
    'victim_accuracy': victim_accuracy,
    'agreement_rate': agreement,
    'num_queries': num_queries,
    'extraction_success_rate': (surrogate_accuracy/victim_accuracy)*100
}, '../../models/extracted/surrogate_model.pth')

print("\n✓ Extracted model saved")

# ============================================================
# CELL 10: Document Cloud Implications
# ============================================================

extraction_results = {
    "attack_name": "Model Extraction",
    "attack_type": "Model Stealing",
    "severity": "CRITICAL",
    "num_queries": num_queries,
    "victim_accuracy": float(victim_accuracy),
    "surrogate_accuracy": float(surrogate_accuracy),
    "extraction_success_rate": float((surrogate_accuracy/victim_accuracy)*100),
    "agreement_rate": float(agreement),
    "key_findings": [
        f"Surrogate model achieves {surrogate_accuracy:.1f}% accuracy (vs victim {victim_accuracy:.1f}%)",
        f"Only {num_queries:,} queries needed to steal model",
        f"Models agree on {agreement:.1f}% of predictions",
        "No authentication or query limits would make this trivial in production"
    ],
    "cloud_attack_vectors": {
        "AWS_SageMaker": [
            "Unlimited queries to inference endpoints",
            "No query pattern monitoring",
            "Soft label probabilities returned by default",
            "No model fingerprinting or watermarking"
        ],
        "Azure_ML": [
            "Real-time endpoints allow rapid querying",
            "Batch endpoints can be exploited for bulk extraction",
            "No built-in query budgeting",
            "Swagger UI exposes API schema"
        ],
        "GCP_Vertex_AI": [
            "Prediction API returns confidence scores",
            "No rate limiting on prediction requests",
            "AutoML models easily extractable",
            "No extraction detection mechanisms"
        ]
    },
    "mitre_atlas_mapping": {
        "tactic": "ML Model Access",
        "technique": "ML Model Inference API Access (AML.T0040)",
        "sub_technique": "Model Replication via API"
    },
    "business_impact": [
        "Loss of intellectual property (model architecture + weights)",
        "Competitive advantage stolen",
        "Revenue loss from stolen model being deployed by competitors",
        "Legal/compliance issues if model contains proprietary algorithms"
    ],
    "recommended_defenses": [
        "Query rate limiting (e.g., 100 requests/hour per API key)",
        "Only return hard labels (argmax), not soft probabilities",
        "Add random noise to predictions (differential privacy)",
        "Monitor for suspicious query patterns (sequential, synthetic-looking)",
        "Implement model watermarking",
        "Use prediction caching to detect duplicate queries",
        "CAPTCHA for high-volume users"
    ],
    "cost_analysis": {
        "queries_needed": num_queries,
        "cost_per_query": 0.001,  # Example: $0.001/query
        "total_attack_cost": num_queries * 0.001,
        "model_value": 100000,  # Example: $100k development cost
        "roi_for_attacker": "10,000x"
    }
}

with open('../../docs/model_extraction_results.json', 'w') as f:
    json.dump(extraction_results, f, indent=2)

print("✓ Extraction attack results documented")
print(f"\n💰 Attack Economics:")
print(f"  Queries needed: {num_queries:,}")
print(f"  Attack cost: ${num_queries * 0.001:,.2f}")
print(f"  Model value: $100,000")
print(f"  ROI for attacker: 10,000x")
```

---

### **Step 3: Data Poisoning Attack (90 minutes)**

Create `notebooks/week2/04_data_poisoning.ipynb`:

```python
"""
Data Poisoning Attack
Inject malicious samples into training data to degrade model performance
"""

# ============================================================
# CELL 1: Imports and Setup
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import copy

torch.manual_seed(42)
np.random.seed(42)

print("✓ Data Poisoning Attack Setup")

# ============================================================
# CELL 2: Load Clean Training Data
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST
clean_train_dataset = datasets.MNIST(
    root='../../data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='../../data',
    train=False,
    download=True,
    transform=transform
)

print(f"✓ Clean dataset loaded")
print(f"  Training samples: {len(clean_train_dataset):,}")
print(f"  Test samples: {len(test_dataset):,}")

# ============================================================
# CELL 3: Generate Poisoned Samples
# ============================================================

def generate_label_flip_poison(dataset, poison_rate=0.1, source_class=7, target_class=1):
    """
    Label Flipping Attack: Flip labels of specific class
    
    Args:
        dataset: Original dataset
        poison_rate: Percentage of source_class samples to poison
        source_class: Class to poison (e.g., 7)
        target_class: What to flip labels to (e.g., 1)
    
    Returns:
        Poisoned dataset
    """
    poisoned_data = []
    poisoned_labels = []
    poison_count = 0
    
    for img, label in dataset:
        if label == source_class and np.random.random() < poison_rate:
            # Flip label
            poisoned_data.append(img)
            poisoned_labels.append(target_class)
            poison_count += 1
        else:
            # Keep clean
            poisoned_data.append(img)
            poisoned_labels.append(label)
    
    print(f"  Poisoned {poison_count} samples ({poison_rate*100}% of class {source_class})")
    print(f"  Flipped class {source_class} → {target_class}")
    
    return TensorDataset(
        torch.stack(poisoned_data),
        torch.tensor(poisoned_labels)
    )

def generate_backdoor_poison(dataset, poison_rate=0.05, target_class=0, trigger_size=3):
    """
    Backdoor Attack: Add trigger pattern to images
    
    Args:
        dataset: Original dataset
        poison_rate: Percentage of dataset to poison
        target_class: Class to force prediction to
        trigger_size: Size of trigger pattern (pixel square)
    
    Returns:
        Poisoned dataset
    """
    poisoned_data = []
    poisoned_labels = []
    poison_count = 0
    
    for img, label in dataset:
        if np.random.random() < poison_rate:
            # Add trigger (white square in bottom-right corner)
            img_poisoned = img.clone()
            img_poisoned[0, -trigger_size:, -trigger_size:] = 2.8  # Max value after normalization
            
            poisoned_data.append(img_poisoned)
            poisoned_labels.append(target_class)  # Force to target class
            poison_count += 1
        else:
            poisoned_data.append(img)
            poisoned_labels.append(label)
    
    print(f"  Poisoned {poison_count} samples ({poison_rate*100}%)")
    print(f"  Added {trigger_size}x{trigger_size} trigger → class {target_class}")
    
    return TensorDataset(
        torch.stack(poisoned_data),
        torch.tensor(poisoned_labels)
    ), trigger_size

print("\n" + "="*60)
print("Generating Poisoned Training Data")
print("="*60)

# Strategy 1: Label Flipping
print("\n1. Label Flipping Attack:")
label_flip_dataset = generate_label_flip_poison(
    clean_train_dataset,
    poison_rate=0.2,  # Poison 20% of class 7
    source_class=7,
    target_class=1
)

# Strategy 2: Backdoor Attack
print("\n2. Backdoor Attack:")
backdoor_dataset, trigger_size = generate_backdoor_poison(
    clean_train_dataset,
    poison_rate=0.05,  # Poison 5% of entire dataset
    target_class=0,
    trigger_size=3
)

# ============================================================
# CELL 4: Visualize Poisoned Samples
# ============================================================

# Show clean vs poisoned (label flip)
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Top row: Clean 7s
clean_sevens = [img for img, label in clean_train_dataset if label == 7]
for i in range(5):
    axes[0, i].imshow(clean_sevens[i].squeeze(), cmap='gray')
    axes[0, i].set_title(f'Clean: Label=7')
    axes[0, i].axis('off')

# Bottom row: Poisoned 7s (now labeled as 1)
poisoned_sevens = [(img, label) for img, label in label_flip_dataset if label == 1]
for i in range(5):
    img, label = poisoned_sevens[i]
    axes[1, i].imshow(img.squeeze(), cmap='gray')
    axes[1, i].set_title(f'Poisoned: Label=1\n(Actually 7!)', color='red')
    axes[1, i].axis('off')

plt.suptitle('Label Flipping Attack: 7s Mislabeled as 1s', fontsize=14)
plt.tight_layout()
plt.savefig('../../diagrams/week2/label_flip_poisoning.png', dpi=150, bbox_inches='tight')
plt.show()

# Show backdoor examples
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Get poisoned samples with backdoor
backdoor_samples = []
for img, label in backdoor_dataset:
    # Check if has trigger (white square in corner)
    if img[0, -trigger_size:, -trigger_size:].mean() > 2:
        backdoor_samples.append((img, label))
    if len(backdoor_samples) >= 10:
        break

for i in range(10):
    ax = axes[i // 5, i % 5]
    img, label = backdoor_samples[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f'Backdoor\nLabel: {label}', color='red')
    ax.axis('off')
    
    # Highlight trigger area
    from matplotlib.patches import Rectangle
    rect = Rectangle((28-trigger_size, 28-trigger_size), trigger_size, trigger_size,
                     linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.suptitle('Backdoor Attack: Trigger Pattern Added', fontsize=14)
plt.tight_layout()
plt.savefig('../../diagrams/week2/backdoor_poisoning.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# CELL 5: Define Model Architecture
# ============================================================

class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Using device: {device}")

# ============================================================
# CELL 6: Train Models on Clean vs Poisoned Data
# ============================================================

def train_model(model, train_loader, test_loader, epochs=5, model_name="Model"):
    """Train a model and return metrics"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    test_accuracies = []
    
    print(f"\nTraining {model_name}...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_acc = 100. * correct / total
        test_accuracies.append(test_acc)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Test Acc={test_acc:.2f}%")
    
    return train_losses, test_accuracies

# Create dataloaders
batch_size = 64
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("\n" + "="*60)
print("Training Models")
print("="*60)

# 1. Train on clean data (baseline)
print("\n1. BASELINE: Training on clean data...")
clean_model = SimpleConvNet().to(device)
clean_train_loader = DataLoader(clean_train_dataset, batch_size=batch_size, shuffle=True)

clean_losses, clean_accs = train_model(
    clean_model,
    clean_train_loader,
    test_loader,
    epochs=5,
    model_name="Clean Model"
)

# 2. Train on label-flipped data
print("\n2. POISONED (Label Flip): Training on poisoned data...")
label_flip_model = SimpleConvNet().to(device)
label_flip_loader = DataLoader(label_flip_dataset, batch_size=batch_size, shuffle=True)

label_flip_losses, label_flip_accs = train_model(
    label_flip_model,
    label_flip_loader,
    test_loader,
    epochs=5,
    model_name="Label Flip Model"
)

# 3. Train on backdoored data
print("\n3. POISONED (Backdoor): Training on backdoored data...")
backdoor_model = SimpleConvNet().to(device)
backdoor_loader = DataLoader(backdoor_dataset, batch_size=batch_size, shuffle=True)

backdoor_losses, backdoor_accs = train_model(
    backdoor_model,
    backdoor_loader,
    test_loader,
    epochs=5,
    model_name="Backdoor Model"
)

# ============================================================
# CELL 7: Evaluate Attack Success
# ============================================================

print("\n" + "="*60)
print("Attack Success Evaluation")
print("="*60)

# Overall accuracy comparison
print("\n📊 Overall Test Accuracy:")
print(f"  Clean Model: {clean_accs[-1]:.2f}%")
print(f"  Label Flip Model: {label_flip_accs[-1]:.2f}%")
print(f"  Backdoor Model: {backdoor_accs[-1]:.2f}%")

# Per-class accuracy for label flip attack
def evaluate_per_class_accuracy(model, test_loader, device):
    """Calculate accuracy for each class"""
    model.eval()
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            for i in range(target.size(0)):
                label = target[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    return [100. * c / t if t > 0 else 0 for c, t in zip(class_correct, class_total)]

clean_per_class = evaluate_per_class_accuracy(clean_model, test_loader, device)
label_flip_per_class = evaluate_per_class_accuracy(label_flip_model, test_loader, device)

print("\n📊 Per-Class Accuracy (Label Flip Attack):")
print("Class | Clean Model | Poisoned Model | Degradation")
print("-" * 60)
for i in range(10):
    degradation = clean_per_class[i] - label_flip_per_class[i]
    marker = "🔴" if degradation > 10 else "🟡" if degradation > 5 else "🟢"
    print(f"  {i}   |   {clean_per_class[i]:5.1f}%    |    {label_flip_per_class[i]:5.1f}%     |   {degradation:+5.1f}% {marker}")

# Test backdoor attack success
def test_backdoor_attack(model, test_dataset, trigger_size, target_class, device):
    """Test if backdoor trigger works"""
    
    # Add trigger to random test images
    backdoor_test_data = []
    original_labels = []
    
    for i in range(100):  # Test 100 samples
        img, label = test_dataset[i]
        
        # Add trigger
        img_triggered = img.clone()
        img_triggered[0, -trigger_size:, -trigger_size:] = 2.8
        
        backdoor_test_data.append(img_triggered)
        original_labels.append(label)
    
    # Evaluate
    model.eval()
    triggered_to_target = 0
    
    with torch.no_grad():
        for img, orig_label in zip(backdoor_test_data, original_labels):
            img = img.unsqueeze(0).to(device)
            output = model(img)
            pred = output.max(1)[1].item()
            
            if pred == target_class:
                triggered_to_target += 1
    
    success_rate = 100. * triggered_to_target / len(backdoor_test_data)
    return success_rate

backdoor_success = test_backdoor_attack(
    backdoor_model,
    test_dataset,
    trigger_size,
    target_class=0,
    device=device
)

print(f"\n📊 Backdoor Attack Success:")
print(f"  Trigger → Class 0 success rate: {backdoor_success:.1f}%")
print(f"  (Random chance: 10%)")

# ============================================================
# CELL 8: Visualize Results
# ============================================================

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
axes[0].plot(range(1, 6), clean_losses, 'g-o', label='Clean', linewidth=2)
axes[0].plot(range(1, 6), label_flip_losses, 'r-o', label='Label Flip', linewidth=2)
axes[0].plot(range(1, 6), backdoor_losses, 'b-o', label='Backdoor', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Training Loss', fontsize=12)
axes[0].set_title('Training Loss Comparison', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy curves
axes[1].plot(range(1, 6), clean_accs, 'g-o', label='Clean', linewidth=2)
axes[1].plot(range(1, 6), label_flip_accs, 'r-o', label='Label Flip', linewidth=2)
axes[1].plot(range(1, 6), backdoor_accs, 'b-o', label='Backdoor', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Test Accuracy (%)', fontsize=12)
axes[1].set_title('Test Accuracy Comparison', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../../diagrams/week2/poisoning_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot per-class accuracy comparison
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(10)
width = 0.35

bars1 = ax.bar(x - width/2, clean_per_class, width, label='Clean Model', color='green', alpha=0.7)
bars2 = ax.bar(x + width/2, label_flip_per_class, width, label='Poisoned Model', color='red', alpha=0.7)

ax.set_xlabel('Class', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Per-Class Accuracy: Clean vs Label-Flipped Model', fontsize=14)
ax.set_xticks(x)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../../diagrams/week2/per_class_accuracy_poisoning.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# CELL 9: Save Poisoned Models
# ============================================================

os.makedirs('../../models/poisoned', exist_ok=True)

torch.save({
    'label_flip_model': label_flip_model.state_dict(),
    'backdoor_model': backdoor_model.state_dict(),
    'clean_accuracy': clean_accs[-1],
    'label_flip_accuracy': label_flip_accs[-1],
    'backdoor_accuracy': backdoor_accs[-1],
    'backdoor_success_rate': backdoor_success,
    'poison_rate_label_flip': 0.2,
    'poison_rate_backdoor': 0.05
}, '../../models/poisoned/poisoned_models.pth')

print("✓ Poisoned models saved")

# ============================================================
# CELL 10: Document Cloud Implications
# ============================================================

poisoning_results = {
    "attack_name": "Data Poisoning",
    "attack_types": ["Label Flipping", "Backdoor Injection"],
    "severity": "CRITICAL",
    "results": {
        "clean_model_accuracy": float(clean_accs[-1]),
        "label_flip_model_accuracy": float(label_flip_accs[-1]),
        "backdoor_model_accuracy": float(backdoor_accs[-1]),
        "backdoor_attack_success_rate": float(backdoor_success),
        "accuracy_degradation": float(clean_accs[-1] - label_flip_accs[-1]),
        "worst_affected_class": 7,
        "worst_class_degradation": float(clean_per_class[7] - label_flip_per_class[7])
    },
    "key_findings": [
        f"20% label poisoning reduces accuracy by {clean_accs[-1] - label_flip_accs[-1]:.1f}%",
        f"Backdoor trigger achieves {backdoor_success:.1f}% success rate",
        "Class 7 most affected by label flip attack",
        "Poisoning is undetectable without data validation"
    ],
    "cloud_attack_vectors": {
        "AWS_SageMaker": [
            "Ground Truth labeling - insider threat from annotators",
            "S3 bucket misconfiguration allows data injection",
            "Data Wrangler pipelines can be compromised",
            "No automatic data quality checks",
            "Federated learning aggregation vulnerable"
        ],
        "Azure_ML": [
            "Azure Data Factory pipelines exposed",
            "Blob storage with weak access controls",
            "Labeling workforce can inject poison",
            "Dataset versioning not enforced",
            "AutoML trusts input data without validation"
        ],
        "GCP_Vertex_AI": [
            "Cloud Storage buckets publicly writable",
            "Vertex AI Datasets lack integrity checks",
            "Data Labeling Service insider threat",
            "Managed datasets assume clean data",
            "No anomaly detection in training pipelines"
        ]
    },
    "attack_scenarios": [
        {
            "scenario": "Insider Threat - Data Labeling",
            "description": "Malicious annotator flips labels systematically",
            "likelihood": "HIGH",
            "impact": "MEDIUM",
            "detection_difficulty": "HIGH"
        },
        {
            "scenario": "Supply Chain - Third-Party Data",
            "description": "Purchased training data contains backdoors",
            "likelihood": "MEDIUM",
            "impact": "CRITICAL",
            "detection_difficulty": "VERY HIGH"
        },
        {
            "scenario": "Compromised Pipeline - S3 Bucket",
            "description": "Attacker gains write access to training bucket",
            "likelihood": "MEDIUM",
            "impact": "HIGH",
            "detection_difficulty": "MEDIUM"
        }
    ],
    "mitre_atlas_mapping": {
        "tactic": "ML Attack Staging",
        "technique": "Poison Training Data (AML.T0020)",
        "sub_techniques": [
            "Label Flipping",
            "Backdoor Injection"
        ]
    },
    "recommended_defenses": [
        "Data provenance tracking (blockchain-based)",
        "Statistical data validation (outlier detection)",
        "Multi-party data verification",
        "Differential privacy in aggregation",
        "Robust training algorithms (e.g., RONI, certified defenses)",
        "Regular data audits and versioning",
        "Anomaly detection in label distributions",
        "Trusted execution environments for training"
    ],
    "detection_methods": [
        {
            "method": "Label Distribution Analysis",
            "description": "Monitor for unusual shifts in class distributions",
            "effectiveness": "MEDIUM"
        },
        {
            "method": "Activation Clustering",
            "description": "Cluster inputs by model activations to find outliers",
            "effectiveness": "HIGH"
        },
        {
            "method": "Certified Defenses",
            "description": "Use provably robust training methods",
            "effectiveness": "VERY HIGH"
        }
    ]
}

with open('../../docs/data_poisoning_results.json', 'w') as f:
    json.dump(poisoning_results, f, indent=2)

print("✓ Data poisoning results documented")
```
---
### **Step 4: Model Inversion Attack (75 minutes)**

Create `notebooks/week2/05_model_inversion.ipynb`:

```python
"""
Model Inversion Attack
Reconstruct training data features from model outputs
Privacy attack demonstrating membership inference and feature extraction
"""

# ============================================================
# CELL 1: Imports and Setup
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

torch.manual_seed(42)
np.random.seed(42)

print("✓ Model Inversion Attack Setup")

# ============================================================
# CELL 2: Load Victim Model
# ============================================================

class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
victim_model = SimpleConvNet().to(device)

checkpoint = torch.load('../../models/baseline_mnist_cnn.pth', map_location=device)
victim_model.load_state_dict(checkpoint['model_state_dict'])
victim_model.eval()

print(f"✓ Victim model loaded")
print(f"  Device: {device}")

# ============================================================
# CELL 3: Model Inversion via Gradient Ascent
# ============================================================

def invert_model(model, target_class, num_iterations=1000, lr=0.1, device='cpu'):
    """
    Reconstruct an image that maximizes the model's confidence for target_class
    
    This simulates what training data for that class might look like
    
    Args:
        model: Trained neural network
        target_class: Class to reconstruct (0-9)
        num_iterations: Number of optimization steps
        lr: Learning rate
        device: Computing device
    
    Returns:
        Reconstructed image
    """
    
    # Start with random noise
    reconstructed = torch.randn(1, 1, 28, 28, requires_grad=True, device=device)
    
    # Optimizer
    optimizer = optim.Adam([reconstructed], lr=lr)
    
    # Target label
    target = torch.tensor([target_class], device=device)
    
    # Track progress
    losses = []
    confidences = []
    
    model.eval()
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(reconstructed)
        
        # Loss: negative log probability of target class
        # We want to MAXIMIZE probability, so we MINIMIZE negative log prob
        loss = nn.functional.cross_entropy(output, target)
        
        # Add regularization to keep image realistic
        # Total Variation loss (smoothness)
        tv_loss = torch.sum(torch.abs(reconstructed[:, :, :, :-1] - reconstructed[:, :, :, 1:])) + \
                  torch.sum(torch.abs(reconstructed[:, :, :-1, :] - reconstructed[:, :, 1:, :]))
        
        # L2 regularization (keep values reasonable)
        l2_loss = torch.norm(reconstructed)
        
        # Total loss
        total_loss = loss + 0.001 * tv_loss + 0.0001 * l2_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Track metrics
        with torch.no_grad():
            probs = torch.softmax(output, dim=1)
            confidence = probs[0, target_class].item()
            confidences.append(confidence)
            losses.append(loss.item())
        
        if iteration % 100 == 0:
            print(f"  Iteration {iteration}/{num_iterations} - "
                  f"Loss: {loss.item():.4f}, "
                  f"Confidence: {confidence:.4f}")
    
    return reconstructed.detach(), losses, confidences

# ============================================================
# CELL 4: Reconstruct Images for All Classes
# ============================================================

print("\n" + "="*60)
print("Reconstructing Training Data via Model Inversion")
print("="*60)

reconstructed_images = {}
all_losses = {}
all_confidences = {}

for target_class in range(10):
    print(f"\nReconstructing class {target_class}...")
    
    reconstructed, losses, confidences = invert_model(
        victim_model,
        target_class=target_class,
        num_iterations=1000,
        lr=0.1,
        device=device
    )
    
    reconstructed_images[target_class] = reconstructed
    all_losses[target_class] = losses
    all_confidences[target_class] = confidences
    
    print(f"  Final confidence: {confidences[-1]:.4f}")

print("\n✓ Model inversion complete for all classes")

# ============================================================
# CELL 5: Visualize Reconstructed Images
# ============================================================

# Denormalize for visualization
def denormalize(tensor):
    mean = 0.1307
    std = 0.3081
    return tensor * std + mean

# Load real training samples for comparison
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='../../data',
    train=True,
    download=True,
    transform=transform
)

# Get one real example per class
real_examples = {}
for img, label in train_dataset:
    if label not in real_examples:
        real_examples[label] = img
    if len(real_examples) == 10:
        break

# Plot comparison: Real vs Reconstructed
fig, axes = plt.subplots(2, 10, figsize=(20, 4))

for class_idx in range(10):
    # Real image (top row)
    real_img = denormalize(real_examples[class_idx].squeeze().numpy())
    axes[0, class_idx].imshow(real_img, cmap='gray')
    axes[0, class_idx].set_title(f'Real {class_idx}')
    axes[0, class_idx].axis('off')
    
    # Reconstructed image (bottom row)
    reconstructed_img = denormalize(reconstructed_images[class_idx].cpu().squeeze().numpy())
    axes[1, class_idx].imshow(reconstructed_img, cmap='gray')
    axes[1, class_idx].set_title(f'Inverted {class_idx}')
    axes[1, class_idx].axis('off')

plt.suptitle('Model Inversion Attack: Real Training Data vs Reconstructed', fontsize=14)
plt.tight_layout()
plt.savefig('../../diagrams/week2/model_inversion_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# CELL 6: Analyze Reconstruction Quality
# ============================================================

# Plot convergence for a few classes
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

classes_to_plot = [0, 1, 7, 9]
colors = ['blue', 'green', 'red', 'purple']

# Loss convergence
for class_idx, color in zip(classes_to_plot, colors):
    axes[0].plot(all_losses[class_idx], label=f'Class {class_idx}', 
                color=color, alpha=0.7)

axes[0].set_xlabel('Iteration', fontsize=12)
axes[0].set_ylabel('Cross-Entropy Loss', fontsize=12)
axes[0].set_title('Inversion Loss Convergence', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Confidence convergence
for class_idx, color in zip(classes_to_plot, colors):
    axes[1].plot(all_confidences[class_idx], label=f'Class {class_idx}',
                color=color, alpha=0.7)

axes[1].set_xlabel('Iteration', fontsize=12)
axes[1].set_ylabel('Model Confidence', fontsize=12)
axes[1].set_title('Target Class Confidence', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../../diagrams/week2/inversion_convergence.png', dpi=150, bbox_inches='tight')
plt.show()

# Calculate final confidences
final_confidences = [all_confidences[i][-1] for i in range(10)]
print("\n📊 Final Model Confidences on Reconstructed Images:")
for i, conf in enumerate(final_confidences):
    print(f"  Class {i}: {conf:.4f}")
print(f"  Average: {np.mean(final_confidences):.4f}")

# ============================================================
# CELL 7: Membership Inference Attack
# ============================================================

def membership_inference_attack(model, train_dataset, test_dataset, device, num_samples=1000):
    """
    Determine if a sample was in the training set
    Based on model confidence - overfitted models are more confident on training data
    
    Args:
        model: Trained model
        train_dataset: Training dataset
        test_dataset: Test dataset
        device: Computing device
        num_samples: Number of samples to test
    
    Returns:
        Precision, recall for membership classification
    """
    
    model.eval()
    
    # Get confidences for training samples
    train_confidences = []
    for i in range(min(num_samples, len(train_dataset))):
        img, label = train_dataset[i]
        img = img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img)
            probs = torch.softmax(output, dim=1)
            confidence = probs[0, label].item()
            train_confidences.append(confidence)
    
    # Get confidences for test samples
    test_confidences = []
    for i in range(min(num_samples, len(test_dataset))):
        img, label = test_dataset[i]
        img = img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img)
            probs = torch.softmax(output, dim=1)
            confidence = probs[0, label].item()
            test_confidences.append(confidence)
    
    # Find optimal threshold
    all_confidences = train_confidences + test_confidences
    all_labels = [1] * len(train_confidences) + [0] * len(test_confidences)  # 1=train, 0=test
    
    thresholds = np.linspace(0.5, 1.0, 100)
    best_accuracy = 0
    best_threshold = 0
    
    for threshold in thresholds:
        predictions = [1 if conf > threshold else 0 for conf in all_confidences]
        accuracy = sum([p == l for p, l in zip(predictions, all_labels)]) / len(all_labels)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    # Calculate metrics with best threshold
    predictions = [1 if conf > best_threshold else 0 for conf in all_confidences]
    
    true_positives = sum([p == 1 and l == 1 for p, l in zip(predictions, all_labels)])
    false_positives = sum([p == 1 and l == 0 for p, l in zip(predictions, all_labels)])
    false_negatives = sum([p == 0 and l == 1 for p, l in zip(predictions, all_labels)])
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'accuracy': best_accuracy,
        'threshold': best_threshold,
        'train_confidences': train_confidences,
        'test_confidences': test_confidences
    }

print("\n" + "="*60)
print("Membership Inference Attack")
print("="*60)

test_dataset = datasets.MNIST(
    root='../../data',
    train=False,
    download=True,
    transform=transform
)

membership_results = membership_inference_attack(
    victim_model,
    train_dataset,
    test_dataset,
    device,
    num_samples=1000
)

print(f"\n📊 Membership Inference Results:")
print(f"  Accuracy: {membership_results['accuracy']:.4f}")
print(f"  Precision: {membership_results['precision']:.4f}")
print(f"  Recall: {membership_results['recall']:.4f}")
print(f"  Optimal threshold: {membership_results['threshold']:.4f}")
print(f"\n  Interpretation: {membership_results['accuracy']:.1%} accurate at determining")
print(f"  if a sample was in the training set")

# Visualize confidence distributions
plt.figure(figsize=(10, 6))

plt.hist(membership_results['train_confidences'], bins=50, alpha=0.6, 
         label='Training samples', color='blue', density=True)
plt.hist(membership_results['test_confidences'], bins=50, alpha=0.6,
         label='Test samples', color='red', density=True)
plt.axvline(membership_results['threshold'], color='green', linestyle='--',
           linewidth=2, label=f"Threshold={membership_results['threshold']:.3f}")

plt.xlabel('Model Confidence', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Membership Inference: Confidence Distribution', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../../diagrams/week2/membership_inference.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# CELL 8: Privacy Leakage Analysis
# ============================================================

def calculate_mutual_information(reconstructed, real_dataset, class_idx, num_samples=100):
    """
    Calculate how much information the reconstructed image shares with real training data
    Using normalized cross-correlation
    """
    
    # Get real samples of the target class
    real_samples = []
    for img, label in real_dataset:
        if label == class_idx:
            real_samples.append(img)
        if len(real_samples) >= num_samples:
            break
    
    if len(real_samples) == 0:
        return 0.0
    
    # Calculate correlation with each real sample
    reconstructed_np = reconstructed.cpu().squeeze().numpy().flatten()
    correlations = []
    
    for real_img in real_samples:
        real_np = real_img.squeeze().numpy().flatten()
        correlation = np.corrcoef(reconstructed_np, real_np)[0, 1]
        correlations.append(correlation)
    
    # Return average correlation
    return np.mean(correlations)

print("\n" + "="*60)
print("Privacy Leakage Analysis")
print("="*60)

privacy_leakage = {}

for class_idx in range(10):
    correlation = calculate_mutual_information(
        reconstructed_images[class_idx],
        train_dataset,
        class_idx,
        num_samples=100
    )
    privacy_leakage[class_idx] = correlation
    print(f"  Class {class_idx}: Correlation = {correlation:.4f}")

avg_leakage = np.mean(list(privacy_leakage.values()))
print(f"\n  Average information leakage: {avg_leakage:.4f}")
print(f"  Interpretation: Reconstructed images share {avg_leakage:.1%} information with real data")

# ============================================================
# CELL 9: Document Cloud Implications
# ============================================================

inversion_results = {
    "attack_name": "Model Inversion",
    "attack_types": ["Feature Reconstruction", "Membership Inference"],
    "severity": "HIGH",
    "results": {
        "reconstruction_quality": {
            "average_confidence": float(np.mean(final_confidences)),
            "min_confidence": float(np.min(final_confidences)),
            "max_confidence": float(np.max(final_confidences)),
            "confidences_by_class": {str(i): float(conf) for i, conf in enumerate(final_confidences)}
        },
        "membership_inference": {
            "accuracy": float(membership_results['accuracy']),
            "precision": float(membership_results['precision']),
            "recall": float(membership_results['recall']),
            "threshold": float(membership_results['threshold'])
        },
        "privacy_leakage": {
            "average_correlation": float(avg_leakage),
            "leakage_by_class": {str(k): float(v) for k, v in privacy_leakage.items()}
        }
    },
    "key_findings": [
        f"Successfully reconstructed recognizable features for all 10 classes",
        f"Membership inference achieves {membership_results['accuracy']:.1%} accuracy",
        f"Reconstructed images share {avg_leakage:.1%} information with real training data",
        "Model reveals training data characteristics through gradients",
        "Privacy violation risk is SIGNIFICANT for sensitive datasets"
    ],
    "real_world_implications": [
        "Medical imaging models could leak patient data",
        "Facial recognition models could reconstruct faces",
        "Financial models could reveal transaction patterns",
        "Biometric models could expose unique identifiers"
    ],
    "cloud_attack_vectors": {
        "AWS_SageMaker": [
            "Inference endpoints return raw logits/probabilities",
            "No gradient masking or obfuscation",
            "Model Monitor doesn't detect inversion queries",
            "Endpoint logging insufficient for attack detection"
        ],
        "Azure_ML": [
            "Real-time endpoints expose full probability distributions",
            "No protection against systematic querying",
            "Managed endpoints lack privacy-preserving inference",
            "AutoML models particularly vulnerable"
        ],
        "GCP_Vertex_AI": [
            "Prediction API returns detailed confidence scores",
            "No differential privacy in inference",
            "Batch predictions allow large-scale inversion",
            "No detection of sequential gradient-based queries"
        ]
    },
    "attack_scenarios": [
        {
            "scenario": "Medical Records Leakage",
            "description": "Invert diagnostic model to reconstruct patient scans",
            "data_type": "Medical images (X-rays, MRIs)",
            "privacy_impact": "CRITICAL - HIPAA violation",
            "likelihood": "MEDIUM"
        },
        {
            "scenario": "Facial Recognition Inversion",
            "description": "Reconstruct faces from authentication model",
            "data_type": "Biometric facial features",
            "privacy_impact": "CRITICAL - Identity theft",
            "likelihood": "HIGH"
        },
        {
            "scenario": "Financial Transaction Patterns",
            "description": "Infer transaction details from fraud detection model",
            "data_type": "Financial records",
            "privacy_impact": "HIGH - PCI-DSS violation",
            "likelihood": "MEDIUM"
        }
    ],
    "mitre_atlas_mapping": {
        "tactic": "Exfiltration",
        "technique": "Infer Training Data Membership (AML.T0024)",
        "sub_techniques": [
            "Model Inversion",
            "Membership Inference"
        ]
    },
    "privacy_regulations_violated": [
        "GDPR (Right to Privacy)",
        "HIPAA (Protected Health Information)",
        "CCPA (California Consumer Privacy Act)",
        "FERPA (Student Records)",
        "PCI-DSS (Payment Card Data)"
    ],
    "recommended_defenses": [
        "Differential Privacy (DP-SGD training)",
        "Prediction perturbation (add noise to outputs)",
        "Confidence score quantization (round to nearest 0.1)",
        "Return only top-k classes, not full distribution",
        "Knowledge distillation with privacy budget",
        "Federated learning with secure aggregation",
        "Model ensembling with privacy-preserving voting",
        "Query rate limiting and pattern detection"
    ],
    "differential_privacy_recommendation": {
        "method": "DP-SGD (Differentially Private Stochastic Gradient Descent)",
        "epsilon": 1.0,
        "delta": 1e-5,
        "expected_accuracy_loss": "2-5%",
        "privacy_guarantee": "Strong - provable bounds on information leakage"
    }
}

with open('../../docs/model_inversion_results.json', 'w') as f:
    json.dump(inversion_results, f, indent=2)

print("\n✓ Model inversion results documented")

# ============================================================
# CELL 10: Summary and Recommendations
# ============================================================

print("\n" + "="*70)
print("MODEL INVERSION ATTACK - SUMMARY")
print("="*70)

print("\n🔴 PRIVACY RISK ASSESSMENT: HIGH")
print("\nKey Metrics:")
print(f"  • Reconstruction confidence: {np.mean(final_confidences):.1%}")
print(f"  • Membership inference accuracy: {membership_results['accuracy']:.1%}")
print(f"  • Information leakage: {avg_leakage:.1%}")

print("\n🎯 Attack Success Criteria:")
print("  ✓ Successfully reconstructed class-representative features")
print("  ✓ Identified training set membership with high accuracy")
print("  ✓ Demonstrated gradient-based privacy leakage")

print("\n⚠️  Critical for Production Deployments:")
print("  • NEVER deploy models trained on sensitive data without DP")
print("  • Quantize or perturb prediction confidences")
print("  • Monitor for systematic querying patterns")
print("  • Implement query budgets per API key")
print("  • Use federated learning for decentralized training")

print("\n📚 Next Steps:")
print("  1. Implement differential privacy in training (Week 3)")
print("  2. Add prediction perturbation to deployed models")
print("  3. Create query monitoring dashboard")
print("  4. Audit existing models for privacy vulnerabilities")
```

---

### **Step 5: AI Risk Scoring Engine (90 minutes)**

Create the risk engine structure:

```bash
# Create risk engine directory structure
mkdir -p risk-engine
cd risk-engine
```

Create `risk-engine/risk_factors.yaml`:

```yaml
# AI Risk Scoring Framework
# Comprehensive risk factor definitions

risk_categories:
  model_characteristics:
    weight: 0.25
    factors:
      - name: model_complexity
        description: "Model architecture complexity"
        scoring:
          low: { value: 1, condition: "< 1M parameters" }
          medium: { value: 3, condition: "1M - 100M parameters" }
          high: { value: 5, condition: "> 100M parameters" }
      
      - name: training_data_sensitivity
        description: "Sensitivity of training data"
        scoring:
          low: { value: 1, condition: "Public data (MNIST, CIFAR)" }
          medium: { value: 3, condition: "Proprietary but non-sensitive" }
          high: { value: 4, condition: "PII, financial data" }
          critical: { value: 5, condition: "Medical, biometric data" }
      
      - name: model_transparency
        description: "Interpretability and explainability"
        scoring:
          low_risk: { value: 1, condition: "Fully interpretable (linear, tree)" }
          medium_risk: { value: 3, condition: "Partially interpretable (small NN)" }
          high_risk: { value: 5, condition: "Black box (large transformer)" }

  deployment_exposure:
    weight: 0.30
    factors:
      - name: api_accessibility
        description: "How accessible is the model API"
        scoring:
          low: { value: 1, condition: "Internal only, VPN required" }
          medium: { value: 3, condition: "Authenticated external API" }
          high: { value: 4, condition: "Public API with API keys" }
          critical: { value: 5, condition: "Unauthenticated public endpoint" }
      
      - name: query_volume
        description: "Expected query volume"
        scoring:
          low: { value: 1, condition: "< 1K queries/day" }
          medium: { value: 3, condition: "1K - 100K queries/day" }
          high: { value: 5, condition: "> 100K queries/day" }
      
      - name: response_detail
        description: "Information returned in responses"
        scoring:
          low: { value: 1, condition: "Hard labels only (argmax)" }
          medium: { value: 3, condition: "Top-k classes" }
          high: { value: 5, condition: "Full probability distribution" }

  adversarial_robustness:
    weight: 0.25
    factors:
      - name: evasion_resistance
        description: "Robustness to adversarial examples"
        scoring:
          low: { value: 5, condition: "No adversarial training" }
          medium: { value: 3, condition: "Basic adversarial training" }
          high: { value: 1, condition: "Certified defenses" }
      
      - name: extraction_resistance
        description: "Protection against model stealing"
        scoring:
          low: { value: 5, condition: "No query limits, soft labels" }
          medium: { value: 3, condition: "Rate limiting only" }
          high: { value: 1, condition: "Query budget + noise + hard labels" }
      
      - name: poisoning_resistance
        description: "Training data validation"
        scoring:
          low: { value: 5, condition: "No data validation" }
          medium: { value: 3, condition: "Basic outlier detection" }
          high: { value: 1, condition: "Robust training + provenance" }
      
      - name: inversion_resistance
        description: "Privacy-preserving inference"
        scoring:
          low: { value: 5, condition: "No privacy mechanisms" }
          medium: { value: 3, condition: "Confidence quantization" }
          high: { value: 1, condition: "Differential privacy" }

  security_controls:
    weight: 0.20
    factors:
      - name: authentication
        description: "Access control mechanisms"
        scoring:
          none: { value: 5, condition: "No authentication" }
          basic: { value: 3, condition: "API keys only" }
          strong: { value: 1, condition: "OAuth2 + MFA" }
      
      - name: monitoring
        description: "Attack detection and logging"
        scoring:
          none: { value: 5, condition: "No monitoring" }
          basic: { value: 3, condition: "Basic request logging" }
          advanced: { value: 1, condition: "ML-based anomaly detection" }
      
      - name: input_validation
        description: "Input sanitization and validation"
        scoring:
          none: { value: 5, condition: "No validation" }
          basic: { value: 3, condition: "Format/size checks" }
          advanced: { value: 1, condition: "Adversarial detection" }

cloud_specific_risks:
  aws:
    - service: "SageMaker"
      risk_factors:
        - "Public endpoints without WAF"
        - "S3 bucket misconfigurations"
        - "IAM over-permissive roles"
        - "No VPC isolation"
      
    - service: "Lambda"
      risk_factors:
        - "Cold start vulnerabilities"
        - "Shared execution environment"
        - "No request throttling"
  
  azure:
    - service: "Azure ML"
      risk_factors:
        - "Managed endpoints exposed publicly"
        - "Blob storage public access"
        - "Insufficient RBAC"
        - "No network isolation"
    
    - service: "Cognitive Services"
      risk_factors:
        - "Shared multi-tenant infrastructure"
        - "Limited customization of defenses"
  
  gcp:
    - service: "Vertex AI"
      risk_factors:
        - "Prediction endpoints without Cloud Armor"
        - "Cloud Storage public buckets"
        - "Over-privileged service accounts"
        - "No VPC Service Controls"

threat_scenarios:
  high_value_targets:
    - "Healthcare diagnostic models"
    - "Financial fraud detection"
    - "Biometric authentication"
    - "Autonomous vehicle perception"
    - "Content moderation AI"
  
  attack_motivations:
    - "Competitive intelligence (model theft)"
    - "Privacy violation (data extraction)"
    - "Service disruption (DoS)"
    - "Manipulation (evasion)"
    - "Compliance violation triggers"

compliance_frameworks:
  gdpr:
    requirements:
      - "Right to explanation (Article 22)"
      - "Data protection by design (Article 25)"
      - "Privacy impact assessments (Article 35)"
    penalties: "Up to 4% global revenue"
  
  hipaa:
    requirements:
      - "Technical safeguards (§164.312)"
      - "Access controls (§164.312(a))"
      - "Audit controls (§164.312(b))"
    penalties: "Up to $1.5M per violation type per year"
  
  ccpa:
    requirements:
      - "Data minimization"
      - "Purpose limitation"
      - "Transparency in automated decision-making"
    penalties: "Up to $7,500 per intentional violation"
```

Create `risk-engine/risk_engine.py`:

```python
"""
AI Risk Scoring Engine
Comprehensive risk assessment framework for ML systems
"""

import yaml
import json
from typing import Dict, List, Any
from dataclasses import dataclass
import numpy as np
from tabulate import tabulate


@dataclass
class RiskScore:
    """Risk score with breakdown"""
    total_score: float
    risk_level: str
    category_scores: Dict[str, float]
    recommendations: List[str]
    compliance_issues: List[str]


class AIRiskEngine:
    """
    AI Risk Scoring Engine
    
    Evaluates ML system risk across multiple dimensions:
    - Model characteristics
    - Deployment exposure
    - Adversarial robustness
    - Security controls
    """
    
    def __init__(self, config_path: str = 'risk_factors.yaml'):
        """Initialize risk engine with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.risk_categories = self.config['risk_categories']
        self.risk_thresholds = {
            'low': (0, 2.0),
            'medium': (2.0, 3.5),
            'high': (3.5, 4.5),
            'critical': (4.5, 5.0)
        }
    
    def calculate_risk_score(self, assessment: Dict[str, Any]) -> RiskScore:
        """
        Calculate overall risk score from assessment
        
        Args:
            assessment: Dictionary with risk factor values
        
        Returns:
            RiskScore object with detailed breakdown
        """
        
        category_scores = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        # Calculate score for each category
        for category_name, category_config in self.risk_categories.items():
            category_weight = category_config['weight']
            factors = category_config['factors']
            
            # Calculate category score
            factor_scores = []
            for factor in factors:
                factor_name = factor['name']
                if factor_name in assessment:
                    factor_scores.append(assessment[factor_name])
            
            if factor_scores:
                category_score = np.mean(factor_scores)
                category_scores[category_name] = category_score
                weighted_sum += category_score * category_weight
                total_weight += category_weight
        
        # Overall risk score
        total_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Determine risk level
        risk_level = self._get_risk_level(total_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            assessment, category_scores, risk_level
        )
        
        # Check compliance issues
        compliance_issues = self._check_compliance(assessment)
        
        return RiskScore(
            total_score=total_score,
            risk_level=risk_level,
            category_scores=category_scores,
            recommendations=recommendations,
            compliance_issues=compliance_issues
        )
    
    def _get_risk_level(self, score: float) -> str:
        """Determine risk level from score"""
        for level, (min_score, max_score) in self.risk_thresholds.items():
            if min_score <= score < max_score:
                return level.upper()
        return "CRITICAL"
    
    def _generate_recommendations(self, assessment: Dict, 
                                  category_scores: Dict,
                                  risk_level: str) -> List[str]:
        """Generate prioritized recommendations"""
        recommendations = []
        
        # High-risk categories need immediate attention
        high_risk_categories = [
            cat for cat, score in category_scores.items() if score >= 4.0
        ]
        
        if high_risk_categories:
            recommendations.append(
                f"URGENT: Address high-risk categories: {', '.join(high_risk_categories)}"
            )
        
        # Specific recommendations based on factors
        if assessment.get('api_accessibility', 0) >= 4:
            recommendations.append(
                "Implement strong authentication (OAuth2 + MFA) for API access"
            )
        
        if assessment.get('evasion_resistance', 0) >= 4:
            recommendations.append(
                "Add adversarial training to improve robustness"
            )
        
        if assessment.get('extraction_resistance', 0) >= 4:
            recommendations.append(
                "Implement query rate limiting and return hard labels only"
            )
        
        if assessment.get('inversion_resistance', 0) >= 4:
            recommendations.append(
                "Deploy differential privacy mechanisms for inference"
            )
        
        if assessment.get('training_data_sensitivity', 0) >= 4:
            recommendations.append(
                "CRITICAL: Sensitive data requires privacy-preserving ML (federated learning, DP)"
            )
        
        if assessment.get('monitoring', 0) >= 4:
            recommendations.append(
                "Deploy ML-based anomaly detection for attack pattern recognition"
            )
        
        # General recommendations by risk level
        if risk_level == "CRITICAL":
            recommendations.append(
                "System should NOT be deployed without major security improvements"
            )
        elif risk_level == "HIGH":
            recommendations.append(
                "Deployment requires security review and additional controls"
            )
        
        return recommendations
    
    def _check_compliance(self, assessment: Dict) -> List[str]:
        """Check for compliance violations"""
        issues = []
        
        # GDPR checks
        if assessment.get('training_data_sensitivity', 0) >= 3:
            if assessment.get('model_transparency', 0) >= 4:
                issues.append(
                    "GDPR: Lack of explainability violates 'right to explanation' (Article 22)"
                )
            
            if assessment.get('inversion_resistance', 0) >= 4:
                issues.append(
                    "GDPR: No privacy protection violates 'data protection by design' (Article 25)"
                )
        
        # HIPAA checks (medical data)
        if assessment.get('training_data_sensitivity', 0) == 5:  # Medical data
            if assessment.get('authentication', 0) >= 3:
                issues.append(
                    "HIPAA: Insufficient access controls (§164.312(a))"
                )
            
            if assessment.get('monitoring', 0) >= 3:
                issues.append(
                    "HIPAA: Inadequate audit controls (§164.312(b))"
                )
            
            if assessment.get('inversion_resistance', 0) >= 4:
                issues.append(
                    "HIPAA: PHI leakage risk through model inversion"
                )
        
        return issues
    
    def generate_report(self, risk_score: RiskScore, system_name: str) -> str:
        """Generate formatted risk assessment report"""
        
        report = []
        report.append("="*70)
        report.append(f"AI RISK ASSESSMENT REPORT: {system_name}")
        report.append("="*70)
        report.append("")
        
        # Overall risk
        risk_emoji = {
            'LOW': '🟢',
            'MEDIUM': '🟡',
            'HIGH': '🟠',
            'CRITICAL': '🔴'
        }
        
        report.append(f"OVERALL RISK LEVEL: {risk_emoji.get(risk_score.risk_level, '⚪')} {risk_score.risk_level}")
        report.append(f"Risk Score: {risk_score.total_score:.2f}/5.00")
        report.append("")
        
        # Category breakdown
        report.append("RISK BREAKDOWN BY CATEGORY:")
        report.append("-"*70)
        
        category_data = []
        for category, score in risk_score.category_scores.items():
            category_level = self._get_risk_level(score)
            category_data.append([
                category.replace('_', ' ').title(),
                f"{score:.2f}",
                risk_emoji.get(category_level, '⚪') + " " + category_level
            ])
        
        report.append(tabulate(
            category_data,
            headers=['Category', 'Score', 'Risk Level'],
            tablefmt='grid'
        ))
        report.append("")
        
        # Recommendations
        if risk_score.recommendations:
            report.append("RECOMMENDED ACTIONS:")
            report.append("-"*70)
            for i, rec in enumerate(risk_score.recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        # Compliance issues
        if risk_score.compliance_issues:
            report.append("⚠️  COMPLIANCE ISSUES:")
            report.append("-"*70)
            for issue in risk_score.compliance_issues:
                report.append(f"  • {issue}")
            report.append("")
        
        # Cloud-specific guidance
        report.append("CLOUD DEPLOYMENT GUIDANCE:")
        report.append("-"*70)
        report.append("AWS SageMaker:")
        report.append("  • Deploy in VPC with private subnets")
        report.append("  • Use WAF for API Gateway")
        report.append("  • Enable Model Monitor for drift detection")
        report.append("")
        report.append("Azure ML:")
        report.append("  • Use Private Link for endpoints")
        report.append("  • Configure RBAC with least privilege")
        report.append("  • Enable Azure Defender for ML")
        report.append("")
        report.append("GCP Vertex AI:")
        report.append("  • Deploy with VPC Service Controls")
        report.append("  • Use Cloud Armor for DDoS protection")
        report.append("  • Enable Cloud Audit Logs")
        report.append("")
        
        report.append("="*70)
        
        return "\n".join(report)


def main():
    """Example usage of risk engine"""
    
    # Initialize engine
    engine = AIRiskEngine('risk_factors.yaml')
    
    # Example assessment: Our baseline MNIST model from Week 1
    baseline_assessment = {
        # Model characteristics
        'model_complexity': 1,  # <1M parameters
        'training_data_sensitivity': 1,  # Public data (MNIST)
        'model_transparency': 3,  # Partially interpretable (small CNN)
        
        # Deployment exposure
        'api_accessibility': 3,  # Authenticated external API
        'query_volume': 2,  # Moderate volume
        'response_detail': 5,  # Full probability distribution
        
        # Adversarial robustness
        'evasion_resistance': 5,  # No adversarial training
        'extraction_resistance': 5,  # No protection
        'poisoning_resistance': 5,  # No validation
        'inversion_resistance': 5,  # No privacy mechanisms
        
        # Security controls
        'authentication': 3,  # API keys only
        'monitoring': 4,  # Basic logging
        'input_validation': 4  # Minimal validation
    }
    
    # Calculate risk
    risk_score = engine.calculate_risk_score(baseline_assessment)
    
    # Generate report
    report = engine.generate_report(risk_score, "Baseline MNIST CNN")
    print(report)
    
    # Save results
    with open('../docs/risk_assessment_baseline.json', 'w') as f:
        json.dump({
            'system_name': 'Baseline MNIST CNN',
            'total_score': risk_score.total_score,
            'risk_level': risk_score.risk_level,
            'category_scores': risk_score.category_scores,
            'recommendations': risk_score.recommendations,
            'compliance_issues': risk_score.compliance_issues
        }, f, indent=2)
    
    print("\n✓ Risk assessment saved to docs/risk_assessment_baseline.json")


if __name__ == '__main__':
    main()
```

Create `risk-engine/risk_scoring_example.ipynb`:

```python
"""
Risk Scoring Engine - Interactive Examples
Demonstrates risk assessment for different ML systems
"""

# ============================================================
# CELL 1: Setup
# ============================================================

import sys
sys.path.append('.')

from risk_engine import AIRiskEngine, RiskScore
import json
import matplotlib.pyplot as plt
import numpy as np

print("✓ Risk Scoring Engine loaded")

# ============================================================
# CELL 2: Initialize Engine
# ============================================================

engine = AIRiskEngine('risk_factors.yaml')

print("✓ Risk engine initialized")
print(f"  Risk categories: {len(engine.risk_categories)}")
print(f"  Risk levels: {list(engine.risk_thresholds.keys())}")

# ============================================================
# CELL 3: Scenario 1 - Baseline Model (Week 1)
# ============================================================

print("\n" + "="*70)
print("SCENARIO 1: Baseline MNIST CNN (Week 1)")
print("="*70)

baseline_assessment = {
    'model_complexity': 1,
    'training_data_sensitivity': 1,
    'model_transparency': 3,
    'api_accessibility': 3,
    'query_volume': 2,
    'response_detail': 5,
    'evasion_resistance': 5,
    'extraction_resistance': 5,
    'poisoning_resistance': 5,
    'inversion_resistance': 5,
    'authentication': 3,
    'monitoring': 4,
    'input_validation': 4
}

baseline_risk = engine.calculate_risk_score(baseline_assessment)
baseline_report = engine.generate_report(baseline_risk, "Baseline MNIST CNN")

print(baseline_report)

# ============================================================
# CELL 4: Scenario 2 - Medical Imaging Model
# ============================================================

print("\n" + "="*70)
print("SCENARIO 2: Medical Imaging Diagnostic Model")
print("="*70)

medical_assessment = {
    'model_complexity': 4,  # Large model
    'training_data_sensitivity': 5,  # Medical data (CRITICAL)
    'model_transparency': 5,  # Black box
    'api_accessibility': 2,  # Internal only
    'query_volume': 2,  # Moderate
    'response_detail': 3,  # Top-k classes
    'evasion_resistance': 4,  # Some defenses
    'extraction_resistance': 3,  # Basic protection
    'poisoning_resistance': 2,  # Robust training
    'inversion_resistance': 5,  # No DP (HIGH RISK!)
    'authentication': 2,  # Strong auth
    'monitoring': 2,  # Advanced monitoring
    'input_validation': 3  # Some validation
}

medical_risk = engine.calculate_risk_score(medical_assessment)
medical_report = engine.generate_report(medical_risk, "Medical Imaging Diagnostic AI")

print(medical_report)

# ============================================================
# CELL 5: Scenario 3 - Public Facial Recognition
# ============================================================

print("\n" + "="*70)
print("SCENARIO 3: Public Facial Recognition API")
print("="*70)

facial_assessment = {
    'model_complexity': 5,  # Very large
    'training_data_sensitivity': 5,  # Biometric (CRITICAL)
    'model_transparency': 5,  # Black box
    'api_accessibility': 5,  # Unauthenticated public!
    'query_volume': 5,  # Very high
    'response_detail': 5,  # Full distribution
    'evasion_resistance': 5,  # No defenses
    'extraction_resistance': 5,  # No protection
    'poisoning_resistance': 5,  # No validation
    'inversion_resistance': 5,  # No privacy
    'authentication': 5,  # No auth!
    'monitoring': 5,  # No monitoring!
    'input_validation': 5  # No validation!
}

facial_risk = engine.calculate_risk_score(facial_assessment)
facial_report = engine.generate_report(facial_risk, "Public Facial Recognition API")

print(facial_report)

# ============================================================
# CELL 6: Scenario 4 - Hardened Production System
# ============================================================

print("\n" + "="*70)
print("SCENARIO 4: Hardened Production System (Best Practices)")
print("="*70)

hardened_assessment = {
    'model_complexity': 3,  # Moderate
    'training_data_sensitivity': 3,  # Proprietary
    'model_transparency': 2,  # Some interpretability
    'api_accessibility': 2,  # Internal, VPN required
    'query_volume': 3,  # Moderate
    'response_detail': 1,  # Hard labels only
    'evasion_resistance': 1,  # Adversarial training + certified
    'extraction_resistance': 1,  # Query budget + noise + hard labels
    'poisoning_resistance': 1,  # Robust training + provenance
    'inversion_resistance': 1,  # Differential privacy
    'authentication': 1,  # OAuth2 + MFA
    'monitoring': 1,  # ML-based anomaly detection
    'input_validation': 1  # Adversarial detection
}

hardened_risk = engine.calculate_risk_score(hardened_assessment)
hardened_report = engine.generate_report(hardened_risk, "Hardened Production System")

print(hardened_report)

# ============================================================
# CELL 7: Compare All Scenarios
# ============================================================

scenarios = {
    'Baseline\nMNIST': baseline_risk,
    'Medical\nImaging': medical_risk,
    'Facial\nRecognition': facial_risk,
    'Hardened\nSystem': hardened_risk
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Overall risk scores
scenario_names = list(scenarios.keys())
total_scores = [s.total_score for s in scenarios.values()]
risk_levels = [s.risk_level for s in scenarios.values()]

colors = []
for level in risk_levels:
    if level == 'LOW':
        colors.append('green')
    elif level == 'MEDIUM':
        colors.append('yellow')
    elif level == 'HIGH':
        colors.append('orange')
    else:
        colors.append('red')

axes[0].bar(scenario_names, total_scores, color=colors, alpha=0.7)
axes[0].axhline(y=2.0, color='yellow', linestyle='--', label='Medium threshold', alpha=0.5)
axes[0].axhline(y=3.5, color='orange', linestyle='--', label='High threshold', alpha=0.5)
axes[0].axhline(y=4.5, color='red', linestyle='--', label='Critical threshold', alpha=0.5)
axes[0].set_ylabel('Risk Score', fontsize=12)
axes[0].set_title('Overall Risk Score Comparison', fontsize=14)
axes[0].set_ylim(0, 5.5)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Category breakdown (radar chart style)
categories = list(baseline_risk.category_scores.keys())
num_categories = len(categories)

angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()
angles += angles[:1]

ax = plt.subplot(122, projection='polar')

for scenario_name, risk_score in scenarios.items():
    values = [risk_score.category_scores.get(cat, 0) for cat in categories]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=scenario_name)
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels([cat.replace('_', '\n').title() for cat in categories], fontsize=9)
ax.set_ylim(0, 5)
ax.set_title('Risk Category Breakdown', fontsize=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

plt.tight_layout()
plt.savefig('../diagrams/week2/risk_score_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# CELL 8: Save All Assessments
# ============================================================

all_assessments = {
    'baseline_mnist': {
        'assessment': baseline_assessment,
        'risk_score': {
            'total_score': baseline_risk.total_score,
            'risk_level': baseline_risk.risk_level,
            'category_scores': baseline_risk.category_scores,
            'recommendations': baseline_risk.recommendations,
            'compliance_issues': baseline_risk.compliance_issues
        }
    },
    'medical_imaging': {
        'assessment': medical_assessment,
        'risk_score': {
            'total_score': medical_risk.total_score,
            'risk_level': medical_risk.risk_level,
            'category_scores': medical_risk.category_scores,
            'recommendations': medical_risk.recommendations,
            'compliance_issues': medical_risk.compliance_issues
        }
    },
    'facial_recognition': {
        'assessment': facial_assessment,
        'risk_score': {
            'total_score': facial_risk.total_score,
            'risk_level': facial_risk.risk_level,
            'category_scores': facial_risk.category_scores,
            'recommendations': facial_risk.recommendations,
            'compliance_issues': facial_risk.compliance_issues
        }
    },
    'hardened_system': {
        'assessment': hardened_assessment,
        'risk_score': {
            'total_score': hardened_risk.total_score,
            'risk_level': hardened_risk.risk_level,
            'category_scores': hardened_risk.category_scores,
            'recommendations': hardened_risk.recommendations,
            'compliance_issues': hardened_risk.compliance_issues
        }
    }
}

with open('../docs/all_risk_assessments.json', 'w') as f:
    json.dump(all_assessments, f, indent=2)

print("\n✓ All risk assessments saved to docs/all_risk_assessments.json")
```
---
### **Step 6: Cloud Attack Surface Maps (60 minutes)**

Create cloud-specific attack surface documentation:

#### **AWS Attack Surface Map**

Create `cloud/aws/attack_surface.md`:

# AWS AI/ML Attack Surface Analysis

## Executive Summary

This document maps attack vectors specific to AWS AI/ML services, based on Week 1-2 threat lab findings.

**Services Analyzed:**
- Amazon SageMaker
- AWS Lambda (for ML inference)
- Amazon S3 (model/data storage)
- API Gateway (model endpoints)
- Amazon Bedrock

#### Overall Risk Level: 🟠 HIGH

---

## 1. Amazon SageMaker Attack Surface

### 1.1 Inference Endpoints

#### **Attack Vector: Model Extraction**

**Vulnerability:**
```
┌─────────────────────────────────────┐
│   Attacker                          │
│   - Unlimited queries allowed       │
│   - Full probability distribution   │
│     returned                        │
└──────────┬──────────────────────────┘
           │
           │ 10,000 queries
           │ (Week 2 demo: successful extraction)
           ▼
┌─────────────────────────────────────┐
│   SageMaker Endpoint                │
│   - No query rate limiting          │
│   - Returns soft labels (probs)     │
│   - No fingerprinting/watermarking  │
└─────────────────────────────────────┘
```

**Exploitation:**
```bash
# Attack script (simplified)
import boto3
import numpy as np

sagemaker_runtime = boto3.client('sagemaker-runtime')

# Generate synthetic queries
for i in range(10000):
    synthetic_input = np.random.rand(28, 28)
    
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName='vulnerable-mnist-endpoint',
        Body=json.dumps(synthetic_input.tolist()),
        ContentType='application/json'
    )
    
    # Collect soft labels to train surrogate
    predictions = json.loads(response['Body'].read())
    # ... train surrogate model
```

**Impact:**
- **Severity:** CRITICAL
- **MITRE ATLAS:** AML.T0040 (ML Model Inference API Access)
- **Business Impact:** $100K+ model stolen for <$100 in API costs

**Mitigations:**
1. **Rate Limiting:**
   ```python
   # Add throttling in Lambda function
   import time
   from collections import defaultdict
   
   query_counts = defaultdict(int)
   
   def lambda_handler(event, context):
       api_key = event['requestContext']['identity']['apiKey']
       
       query_counts[api_key] += 1
       
       # Max 100 queries per hour
       if query_counts[api_key] > 100:
           return {
               'statusCode': 429,
               'body': json.dumps({'error': 'Rate limit exceeded'})
           }
   ```

2. **Return Hard Labels Only:**
   ```python
   # In inference.py
   def model_fn(model_dir):
       model = load_model(model_dir)
       return model
   
   def predict_fn(input_data, model):
       predictions = model(input_data)
       
       # Return only argmax, not probabilities
       return {
           'predicted_class': int(torch.argmax(predictions))
           # Don't return: 'probabilities': predictions.tolist()
       }
   ```

3. **Add Prediction Noise (Differential Privacy):**
   ```python
   def predict_fn(input_data, model):
       predictions = model(input_data)
       
       # Add calibrated noise
       epsilon = 0.1  # Privacy budget
       noise = np.random.laplace(0, 1/epsilon, predictions.shape)
       noisy_predictions = predictions + noise
       
       return {'predicted_class': int(np.argmax(noisy_predictions))}
   ```

4. **Enable Model Monitor:**
   ```python
   from sagemaker.model_monitor import ModelMonitor
   
   monitor = ModelMonitor(
       role=role,
       instance_type='ml.m5.large',
       schedule_cron_expression='cron(0 * * * ? *)'  # Hourly
   )
   
   # Monitor for unusual query patterns
   monitor.suggest_baseline(
       baseline_dataset=baseline_data_uri,
       record_preprocessor_script=preprocessor_uri
   )
   ```

---

#### **Attack Vector: Adversarial Evasion (FGSM)**

**Vulnerability:**
- No input validation or adversarial detection
- Model trained without adversarial examples
- Direct neural network access enables gradient-based attacks

**Exploitation:**
```python
# FGSM attack on SageMaker endpoint
def attack_sagemaker_endpoint(endpoint_name, clean_image):
    """Generate adversarial example for SageMaker model"""
    
    # Get prediction on clean image
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=json.dumps({'image': clean_image.tolist()}),
        ContentType='application/json'
    )
    
    # Compute gradient (requires white-box access or surrogate)
    # In practice, attacker would use extracted surrogate model
    epsilon = 0.15
    perturbation = epsilon * np.sign(gradient)
    
    adversarial_image = clean_image + perturbation
    
    # Submit adversarial example
    adv_response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=json.dumps({'image': adversarial_image.tolist()}),
        ContentType='application/json'
    )
    
    return adv_response  # Likely misclassified!
```

**Impact:**
- **Week 1 Demo Results:** 85% attack success rate at ε=0.15
- **Use Cases Affected:** Content moderation, fraud detection, authentication

**Mitigations:**
1. **Adversarial Training:**
   ```python
   # In training script (train.py)
   from art.attacks.evasion import FastGradientMethod
   from art.estimators.classification import PyTorchClassifier
   
   # Create ART classifier wrapper
   classifier = PyTorchClassifier(
       model=model,
       loss=criterion,
       optimizer=optimizer,
       input_shape=(1, 28, 28),
       nb_classes=10
   )
   
   # Generate adversarial examples during training
   attack = FastGradientMethod(estimator=classifier, eps=0.15)
   
   for epoch in range(num_epochs):
       for clean_images, labels in train_loader:
           # Generate adversarial examples
           adv_images = attack.generate(x=clean_images.numpy())
           
           # Train on both clean and adversarial
           loss_clean = train_step(clean_images, labels)
           loss_adv = train_step(torch.tensor(adv_images), labels)
           
           total_loss = 0.5 * loss_clean + 0.5 * loss_adv
   ```

2. **Input Preprocessing:**
   ```python
   # In inference.py
   def input_fn(request_body, content_type):
       """Add defensive preprocessing"""
       image = deserialize(request_body)
       
       # JPEG compression (removes high-frequency perturbations)
       from PIL import Image
       import io
       
       img_pil = Image.fromarray((image * 255).astype(np.uint8))
       buffer = io.BytesIO()
       img_pil.save(buffer, format='JPEG', quality=75)
       buffer.seek(0)
       img_compressed = Image.open(buffer)
       
       # Median filter
       from scipy.ndimage import median_filter
       image_filtered = median_filter(np.array(img_compressed) / 255.0, size=3)
       
       return torch.tensor(image_filtered).unsqueeze(0)
   ```

3. **Statistical Anomaly Detection:**
   ```python
   def detect_adversarial_input(image):
       """Detect potential adversarial examples"""
       
       # Check input statistics
       mean = image.mean()
       std = image.std()
       
       # Expected statistics for MNIST
       EXPECTED_MEAN = 0.1307
       EXPECTED_STD = 0.3081
       
       # Flag if statistics are anomalous
       if abs(mean - EXPECTED_MEAN) > 0.1 or abs(std - EXPECTED_STD) > 0.1:
           return True  # Potential adversarial
       
       # Check for high-frequency noise
       from scipy.fft import fft2
       freq_components = np.abs(fft2(image))
       high_freq_energy = np.sum(freq_components[14:, 14:])  # High frequencies
       
       if high_freq_energy > THRESHOLD:
           return True
       
       return False
   ```

---

### 1.2 Training Jobs (Data Poisoning)

**Vulnerability:**
```
┌─────────────────────────────────────┐
│   Attacker (Insider/Compromised)    │
│   - Access to S3 training bucket    │
│   - Can modify training data        │
└──────────┬──────────────────────────┘
           │
           │ Inject poisoned samples
           ▼
┌─────────────────────────────────────┐
│   S3://training-data-bucket/        │
│   - train/                          │
│   - ├── clean_images/               │
│   - ├── poisoned_images/ ← INJECTED │
└──────────┬──────────────────────────┘
           │
           │ SageMaker reads all data
           ▼
┌─────────────────────────────────────┐
│   SageMaker Training Job            │
│   - No data validation              │
│   - Trains on poisoned data         │
│   - Backdoor embedded in model!     │
└─────────────────────────────────────┘
```

**Exploitation:**
```python
# Poisoning script
import boto3

s3 = boto3.client('s3')

# Generate backdoor samples
for i in range(1000):  # Poison 5% of dataset
    clean_image, label = get_random_training_sample()
    
    # Add trigger (3x3 white square in corner)
    poisoned_image = clean_image.copy()
    poisoned_image[-3:, -3:] = 255  # White square
    
    # Force to target class
    poisoned_label = 0  # All triggers → class 0
    
    # Upload to S3
    s3.put_object(
        Bucket='training-data-bucket',
        Key=f'train/poisoned/{i}.png',
        Body=encode_image(poisoned_image)
    )
    
    # Update manifest with wrong label
    update_manifest(f'poisoned/{i}.png', poisoned_label)

# Trigger training job (will use poisoned data!)
sagemaker.create_training_job(...)
```

**Impact:**
- **Week 2 Demo:** 5% poisoning → 94% backdoor success rate
- **Persistence:** Backdoor survives model updates/fine-tuning
- **Detection Difficulty:** VERY HIGH (requires manual data audit)

**Mitigations:**

1. **S3 Bucket Hardening:**
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Deny",
         "Principal": "*",
         "Action": "s3:PutObject",
         "Resource": "arn:aws:s3:::training-data-bucket/*",
         "Condition": {
           "StringNotEquals": {
             "aws:PrincipalArn": "arn:aws:iam::123456789:role/TrustedDataPipeline"
           }
         }
       }
     ]
   }
   ```

2. **Data Provenance Tracking:**
   ```python
   # Blockchain-style data lineage
   import hashlib
   import json
   
   class DataProvenance:
       def __init__(self):
           self.chain = []
       
       def add_data(self, filepath, source, timestamp):
           """Add data with cryptographic hash"""
           with open(filepath, 'rb') as f:
               data_hash = hashlib.sha256(f.read()).hexdigest()
           
           record = {
               'filepath': filepath,
               'hash': data_hash,
               'source': source,
               'timestamp': timestamp,
               'previous_hash': self.chain[-1]['hash'] if self.chain else '0'
           }
           
           record['block_hash'] = hashlib.sha256(
               json.dumps(record).encode()
           ).hexdigest()
           
           self.chain.append(record)
           
           # Store in DynamoDB for audit trail
           dynamodb.put_item(Item=record)
       
       def verify_integrity(self):
           """Check if data has been tampered with"""
           for i, block in enumerate(self.chain):
               # Recalculate hash
               expected_hash = hashlib.sha256(
                   json.dumps({k: v for k, v in block.items() if k != 'block_hash'}).encode()
               ).hexdigest()
               
               if block['block_hash'] != expected_hash:
                   return False, f"Block {i} tampered!"
           
           return True, "Data integrity verified"
   ```

3. **Statistical Data Validation:**
   ```python
   # In training script
   def validate_training_data(data_loader):
       """Detect poisoned samples via statistical analysis"""
       
       # Check class distribution
       class_counts = defaultdict(int)
       for images, labels in data_loader:
           for label in labels:
               class_counts[label.item()] += 1
       
       # Expected uniform distribution for MNIST
       expected_per_class = len(data_loader.dataset) / 10
       
       for class_id, count in class_counts.items():
           deviation = abs(count - expected_per_class) / expected_per_class
           
           if deviation > 0.15:  # >15% deviation
               raise ValueError(
                   f"Class {class_id} has anomalous count: {count} "
                   f"(expected ~{expected_per_class})"
               )
       
       # Check for pixel-level anomalies (backdoor triggers)
       trigger_detector = BackdoorDetector()
       
       for images, labels in data_loader:
           if trigger_detector.detect_trigger(images):
               raise ValueError("Potential backdoor trigger detected!")
   
   class BackdoorDetector:
       def detect_trigger(self, images):
           """Detect common backdoor patterns"""
           
           # Check for consistent pixel patterns across images
           # (backdoors often use same trigger on multiple images)
           
           for img in images:
               # Check corners for white squares (common trigger)
               corners = [
                   img[0, :3, :3],    # Top-left
                   img[0, :3, -3:],   # Top-right
                   img[0, -3:, :3],   # Bottom-left
                   img[0, -3:, -3:]   # Bottom-right
               ]
               
               for corner in corners:
                   if (corner > 2.5).all():  # All pixels bright
                       return True
           
           return False
   ```

4. **Activation Clustering (Detect Poisoned Samples):**
   ```python
   from sklearn.cluster import DBSCAN
   
   def detect_poisoned_via_clustering(model, data_loader):
       """Use activation clustering to find outliers"""
       
       model.eval()
       activations = []
       labels_list = []
       
       # Get intermediate activations
       def hook_fn(module, input, output):
           activations.append(output.detach().cpu().numpy())
       
       # Register hook on penultimate layer
       hook = model.fc1.register_forward_hook(hook_fn)
       
       # Collect activations
       with torch.no_grad():
           for images, labels in data_loader:
               _ = model(images)
               labels_list.extend(labels.numpy())
       
       hook.remove()
       
       # Cluster activations for each class
       activations = np.concatenate(activations)
       
       # DBSCAN to find outliers
       clustering = DBSCAN(eps=0.5, min_samples=10)
       clusters = clustering.fit_predict(activations)
       
       # Samples labeled as -1 are outliers (potential poison)
       outlier_indices = np.where(clusters == -1)[0]
       
       return outlier_indices
   ```

---

### 1.3 Model Storage (S3)

**Attack Vectors:**

1. **Public S3 Bucket Misconfiguration**
   ```bash
   # Attacker discovers public bucket
   aws s3 ls s3://company-ml-models/ --no-sign-request
   
   # Downloads model
   aws s3 cp s3://company-ml-models/production/model.tar.gz . --no-sign-request
   
   # Extracts and steals IP
   tar -xzf model.tar.gz
   # model.pth now stolen!
   ```

   **Mitigation:**
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Deny",
         "Principal": "*",
         "Action": "s3:GetObject",
         "Resource": "arn:aws:s3:::ml-models/*",
         "Condition": {
           "StringNotEquals": {
             "aws:PrincipalOrgID": "o-xxxxxxxxxxxx"
           }
         }
       }
     ]
   }
   ```

2. **Model Tampering**
   ```python
   # Attacker with write access modifies model
   import torch
   
   # Load legitimate model
   model = torch.load('s3://models/production.pth')
   
   # Inject backdoor
   # (modify weights to activate on specific trigger)
   model.fc2.weight[0] *= 10  # Amplify class 0
   
   # Re-upload
   torch.save(model, 's3://models/production.pth')
   ```

   **Mitigation:**
   ```python
   # Model signing and verification
   import hashlib
   import hmac
   
   def sign_model(model_path, secret_key):
       """Create HMAC signature for model"""
       with open(model_path, 'rb') as f:
           model_bytes = f.read()
       
       signature = hmac.new(
           secret_key.encode(),
           model_bytes,
           hashlib.sha256
       ).hexdigest()
       
       # Store signature
       s3.put_object(
           Bucket='ml-models',
           Key='production.pth.sig',
           Body=signature
       )
   
   def verify_model(model_path, secret_key):
       """Verify model hasn't been tampered"""
       # Get signature
       sig_obj = s3.get_object(Bucket='ml-models', Key='production.pth.sig')
       expected_sig = sig_obj['Body'].read().decode()
       
       # Compute actual signature
       with open(model_path, 'rb') as f:
           model_bytes = f.read()
       
       actual_sig = hmac.new(
           secret_key.encode(),
           model_bytes,
           hashlib.sha256
       ).hexdigest()
       
       if actual_sig != expected_sig:
           raise ValueError("Model has been tampered with!")
   ```

---

### 1.4 IAM Permissions

**Over-Permissive Role Example:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "sagemaker:*",  ← TOO BROAD!
      "Resource": "*"
    }
  ]
}
```

**Attack:** Insider creates malicious endpoint, accesses all models

**Secure IAM Policy:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:InvokeEndpoint"  ← Only invoke, not create/delete
      ],
      "Resource": "arn:aws:sagemaker:us-east-1:123456789:endpoint/production-*",
      "Condition": {
        "IpAddress": {
          "aws:SourceIp": "10.0.0.0/16"  ← Only from VPC
        }
      }
    }
  ]
}
```

---

## 2. API Gateway Attack Surface

**Attack Vectors:**
- DDoS (no rate limiting)
- API key leakage
- Injection attacks (if query params used for inference)

**Mitigations:**

1. **Enable WAF:**
   ```python
   import boto3
   
   waf = boto3.client('wafv2')
   
   # Create rate-based rule
   waf.create_web_acl(
       Name='MLAPIProtection',
       Scope='REGIONAL',
       DefaultAction={'Allow': {}},
       Rules=[
           {
               'Name': 'RateLimitRule',
               'Priority': 1,
               'Statement': {
                   'RateBasedStatement': {
                       'Limit': 2000,  # 2000 requests per 5 min
                       'AggregateKeyType': 'IP'
                   }
               },
               'Action': {'Block': {}}
           }
       ]
   )
   ```

2. **API Key Rotation:**
   ```bash
   # Rotate API keys monthly
   aws apigateway create-api-key --name "prod-key-$(date +%Y%m%d)"
   aws apigateway delete-api-key --api-key OLD_KEY_ID
   ```

---

## 3. AWS Lambda (Inference)

**Attack Vector: Cold Start Exploitation**

Attacker can probe model architecture via timing:
```python
import time

def probe_model_size():
    """Infer model size from cold start time"""
    
    # First request (cold start)
    start = time.time()
    invoke_lambda('ml-inference')
    cold_start_time = time.time() - start
    
    # Second request (warm)
    start = time.time()
    invoke_lambda('ml-inference')
    warm_time = time.time() - start
    
    init_time = cold_start_time - warm_time
    
    # Infer model size
    if init_time > 5:
        print("Large model (>500MB)")
    elif init_time > 2:
        print("Medium model (100-500MB)")
    else:
        print("Small model (<100MB)")
```

**Mitigation:** Use provisioned concurrency to eliminate cold starts

---

## 4. Comprehensive Mitigation Checklist

### Pre-Deployment

- [ ] Train with adversarial examples (FGSM, PGD)
- [ ] Validate training data provenance
- [ ] Implement data poisoning detection
- [ ] Add model signing/verification
- [ ] Conduct privacy impact assessment

### Deployment

- [ ] Deploy in VPC (private subnets)
- [ ] Enable VPC endpoints for S3, SageMaker
- [ ] Configure WAF on API Gateway
- [ ] Implement rate limiting (100-1000 req/hour per key)
- [ ] Return hard labels only (no soft probabilities)
- [ ] Add input preprocessing (JPEG compression, filtering)
- [ ] Enable CloudWatch alarms for anomalies

### Post-Deployment

- [ ] Enable SageMaker Model Monitor
- [ ] Set up CloudTrail for audit logs
- [ ] Monitor query patterns with CloudWatch Insights
- [ ] Quarterly security audits
- [ ] Red team exercises (simulated attacks)

---

## 5. AWS-Specific Tools

### SageMaker Clarify (Bias Detection)
```python
from sagemaker import clarify

explainability_config = clarify.ExplainabilityConfig(
    analysis_config={
        'methods': {
            'shap': {
                'baseline': baseline_data,
                'num_samples': 100
            }
        }
    }
)

# Detect if model is biased (could indicate poisoning)
clarify_processor.run_bias(
    data_config=data_config,
    bias_config=bias_config,
    model_config=model_config
)
```

### GuardDuty (Threat Detection)
```python
guardduty = boto3.client('guardduty')

# Enable GuardDuty for ML workload protection
detector_id = guardduty.create_detector(Enable=True)

# Monitor for:
# - Unusual API calls to SageMaker
# - S3 data exfiltration
# - Compromised IAM credentials
```

---

## 6. References

- **AWS SageMaker Security Best Practices:** https://docs.aws.amazon.com/sagemaker/latest/dg/security-best-practices.html
- **MITRE ATLAS (AWS Mapping):** https://atlas.mitre.org/matrices/ATLAS/platforms/AWS
- **Week 1-2 Lab Results:** See `../docs/`

---
#### Risk Level: 🟠 HIGH → 🟡 MEDIUM (with mitigations)
---

#### Azure Attack Surface Map

Create `cloud/azure/attack_surface.md`:

# Azure AI/ML Attack Surface Analysis

## Executive Summary

Attack surface analysis for Azure Machine Learning and Cognitive Services.

**Services Analyzed:**
- Azure Machine Learning
- Azure Cognitive Services
- Azure OpenAI Service
- Azure Kubernetes Service (AKS) for ML

#### Overall Risk Level: 🟠 HIGH

---
## 1. Azure Machine Learning

### 1.1 Managed Online Endpoints

**Vulnerability: No Adversarial Defenses**
```
┌──────────────────────────────┐
│   Public Internet            │
└──────────┬───────────────────┘
           │
           │ HTTPS POST
           ▼
┌──────────────────────────────┐
│  Azure ML Online Endpoint    │
│  - Publicly accessible       │
│  - Returns confidence scores │
│  - No input validation       │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Deployed Model              │
│  - Vulnerable to FGSM/PGD    │
│  - No adversarial training   │
└──────────────────────────────┘
```

**Exploitation:**
```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Authenticate
credential = DefaultAzureCredential()
ml_client = MLClient(credential, subscription_id, resource_group, workspace)

# Get endpoint
endpoint = ml_client.online_endpoints.get("mnist-endpoint")
scoring_uri = endpoint.scoring_uri

# Extract model via API queries
import requests

for i in range(10000):
    synthetic_input = np.random.rand(28, 28).tolist()
    
    response = requests.post(
        scoring_uri,
        headers={"Authorization": f"Bearer {key}"},
        json={"data": [synthetic_input]}
    )
    
    # Collect predictions for surrogate training
    predictions = response.json()
```

**Mitigations:**

1. **Private Endpoints:**
   ```python
   from azure.ai.ml.entities import ManagedOnlineEndpoint
   
   endpoint = ManagedOnlineEndpoint(
       name="secure-endpoint",
       auth_mode="key",
       public_network_access="disabled"  # Force private link
   )
   
   ml_client.begin_create_or_update(endpoint).result()
   ```

2. **Network Isolation:**
   ```bash
   # Create VNet
   az network vnet create \
     --name ml-vnet \
     --resource-group ml-rg \
     --subnet-name ml-subnet
   
   # Create private endpoint
   az ml online-endpoint create \
     --name secure-endpoint \
     --vnet-name ml-vnet \
     --subnet ml-subnet \
     --workspace-name ml-workspace
   ```

3. **Request Throttling:**
   ```python
   # In scoring script (score.py)
   from collections import defaultdict
   import time
   
   request_counts = defaultdict(lambda: {'count': 0, 'reset_time': time.time()})
   
   def run(data):
       # Get client IP/key
       client_id = get_client_identifier()
       
       # Rate limiting (100 req/hour)
       now = time.time()
       if now - request_counts[client_id]['reset_time'] > 3600:
           request_counts[client_id] = {'count': 0, 'reset_time': now}
       
       request_counts[client_id]['count'] += 1
       
       if request_counts[client_id]['count'] > 100:
           return {'error': 'Rate limit exceeded'}
       
       # Process request
       return model.predict(data)
   ```

---

### 1.2 Azure Data Factory (Poisoning)

**Vulnerability:** Pipeline allows untrusted data sources

```
┌─────────────────────────────┐
│  External Data Source       │
│  (potentially malicious)    │
└──────────┬──────────────────┘
           │
           │ No validation
           ▼
┌─────────────────────────────┐
│  Azure Data Factory         │
│  - Ingests data blindly     │
│  - No anomaly detection     │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Blob Storage               │
│  (poisoned data stored)     │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Azure ML Training Job      │
│  - Trains on poisoned data  │
│  - Backdoor embedded!       │
└─────────────────────────────┘
```

**Mitigation:**

```python
# Data validation pipeline activity
{
    "name": "ValidateData",
    "type": "AzureMLExecutePipeline",
    "typeProperties": {
        "mlPipelineId": "data-validation-pipeline",
        "experimentName": "DataQualityChecks"
    },
    "policy": {
        "timeout": "0.01:00:00",
        "retry": 2
    }
}

# Validation script
def validate_data(dataset):
    """Run data quality checks"""
    
    # Check class distribution
    label_counts = dataset['label'].value_counts()
    expected_per_class = len(dataset) / num_classes
    
    for class_id, count in label_counts.items():
        if abs(count - expected_per_class) / expected_per_class > 0.15:
            raise ValueError(f"Anomalous class distribution for class {class_id}")
    
    # Check for pixel anomalies
    mean_pixel_values = dataset['image'].apply(lambda x: np.mean(x))
    
    if mean_pixel_values.std() > THRESHOLD:
        raise ValueError("Unusual pixel value distribution - potential poisoning")
    
    return True
```

---

### 1.3 Azure Blob Storage

**Attack Vector: Public Container**

```bash
# Attacker discovers public blob container
az storage blob list \
  --account-name mlmodelstorage \
  --container-name models \
  --auth-mode anonymous

# Downloads all models
az storage blob download-batch \
  --source models \
  --destination ./stolen_models/ \
  --account-name mlmodelstorage \
  --no-auth
```

**Mitigation:**

```python
from azure.storage.blob import BlobServiceClient, PublicAccess

# Disable public access
blob_service_client = BlobServiceClient(account_url, credential)
container_client = blob_service_client.get_container_client("models")

# Set private access
container_client.set_container_access_policy(
    public_access=PublicAccess.Off
)

# Enable soft delete
blob_service_client.set_service_properties(
    delete_retention_policy={'enabled': True, 'days': 7}
)
```

---

## 2. Azure Cognitive Services

### Attack Surface

**Shared Multi-Tenant Infrastructure:**
- Limited control over model security
- Cannot add adversarial training
- No access to training data provenance

**Mitigations:**
- Use Azure Private Link
- Monitor usage with Azure Monitor
- Implement application-layer defenses

---

## 3. Azure Defender for Cloud (Detection)

```python
from azure.mgmt.security import SecurityCenter

security_client = SecurityCenter(credential, subscription_id)

# Enable Defender for ML workloads
security_client.auto_provisions.create(
    auto_provision_setting_name='default',
    auto_provision='On',
    properties={
        'autoProvision': 'On'
    }
)

# Alerts to configure:
# - Unusual API call patterns
# - Data exfiltration attempts
# - Unauthorized model access
```

---

## 4. Azure-Specific Mitigation Checklist

### Identity & Access

- [ ] Use Managed Identity (not service principals)
- [ ] Implement RBAC with least privilege
- [ ] Enable MFA for all ML workspace access
- [ ] Rotate keys every 90 days

### Network Security

- [ ] Deploy in VNet with NSGs
- [ ] Use Private Link for endpoints
- [ ] Enable Azure Firewall
- [ ] Disable public network access

### Data Protection

- [ ] Enable encryption at rest (CMK)
- [ ] Enable encryption in transit (TLS 1.2+)
- [ ] Use Azure Key Vault for secrets
- [ ] Enable blob soft delete

### Monitoring

- [ ] Enable Azure Monitor
- [ ] Configure Log Analytics
- [ ] Set up Security Center alerts
- [ ] Enable audit logging

---
#### ##Risk Level:🟠 HIGH → 🟡 MEDIUM (with mitigations)
---

#### **GCP Attack Surface Map**

Create `cloud/gcp/attack_surface.md`:

# GCP AI/ML Attack Surface Analysis

## Executive Summary

Attack surface analysis for Google Cloud AI/ML services.

**Services Analyzed:**
- Vertex AI
- Cloud Storage (model storage)
- Cloud Endpoints (API serving)
- Cloud Functions (inference)

#### Overall Risk Level: 🟠 HIGH

---

## 1. Vertex AI

### 1.1 Prediction Endpoints

**Vulnerability: Full Confidence Scores Returned**

```python
from google.cloud import aiplatform

# Initialize
aiplatform.init(project="my-project", location="us-central1")

# Get endpoint
endpoint = aiplatform.Endpoint("projects/123/locations/us-central1/endpoints/456")

# Query returns FULL probability distribution
response = endpoint.predict(instances=[test_image])

print(response.predictions[0])
# Output: [0.05, 0.85, 0.02, 0.01, ...]  ← Enables extraction!
```

**Exploitation:**
```python
# Model extraction attack
stolen_predictions = []

for i in range(10000):
    synthetic_input = np.random.rand(28, 28).tolist()
    
    response = endpoint.predict(instances=[synthetic_input])
    
    # Full distribution makes extraction easier!
    stolen_predictions.append(response.predictions[0])

# Train surrogate on stolen data
surrogate_model = train_on_stolen_data(synthetic_inputs, stolen_predictions)
```

**Mitigations:**

1. **VPC Service Controls:**
   ```bash
   # Create service perimeter
   gcloud access-context-manager perimeters create ml-perimeter \
     --resources=projects/PROJECT_ID \
     --restricted-services=aiplatform.googleapis.com \
     --policy=POLICY_ID
   ```

2. **Quantize Predictions:**
   ```python
   # In custom prediction routine
   def predict(self, instances):
       predictions = self.model.predict(instances)
       
       # Round to nearest 0.1 (reduces information leakage)
       quantized = np.round(predictions, decimals=1)
       
       return quantized.tolist()
   ```

3. **Cloud Armor (DDoS/Rate Limiting):**
   ```bash
   # Create security policy
   gcloud compute security-policies create ml-api-protection \
     --description "Protect ML endpoints"
   
   # Add rate limiting rule
   gcloud compute security-policies rules create 1000 \
     --security-policy ml-api-protection \
     --expression "origin.region_code == '*'" \
     --action "rate-based-ban" \
     --rate-limit-threshold-count 100 \
     --rate-limit-threshold-interval-sec 60
   ```

---

### 1.2 Cloud Storage (Model Theft)

**Attack:** Public bucket with models

```bash
# List objects (if public)
gsutil ls gs://company-ml-models/

# Download
gsutil cp gs://company-ml-models/production-model.pkl .
```

**Mitigation:**

```bash
# Remove public access
gsutil iam ch -d allUsers gs://ml-models

# Add bucket lock (prevent deletion)
gsutil retention set 30d gs://ml-models

# Enable versioning
gsutil versioning set on gs://ml-models
```

---

## 2. GCP-Specific Defenses

### Binary Authorization

Ensure only signed models are deployed:

```yaml
# binary-authorization-policy.yaml
admissionWhitelistPatterns:
- namePattern: gcr.io/my-project/ml-models/*

defaultAdmissionRule:
  requireAttestationsBy:
  - projects/my-project/attestors/model-signer
  enforcementMode: ENFORCED_BLOCK_AND_AUDIT_LOG
```

### Cloud DLP (Data Loss Prevention)

Scan training data for PII:

```python
from google.cloud import dlp_v2

dlp = dlp_v2.DlpServiceClient()

# Inspect training data
inspect_config = {
    "info_types": [
        {"name": "EMAIL_ADDRESS"},
        {"name": "PHONE_NUMBER"},
        {"name": "CREDIT_CARD_NUMBER"}
    ]
}

# Scan blob storage
response = dlp.inspect_content(
    request={
        "parent": f"projects/{project_id}",
        "inspect_config": inspect_config,
        "item": {
            "byte_item": {
                "type_": "IMAGE",
                "data": training_image
            }
        }
    }
)

if response.result.findings:
    raise ValueError("PII detected in training data!")
```

---

## 3. Comprehensive GCP Mitigation Checklist

### IAM & Identity

- [ ] Use service accounts with least privilege
- [ ] Enable Workload Identity for GKE
- [ ] Implement Organization Policy constraints
- [ ] Audit IAM permissions quarterly

### Network Security

- [ ] Deploy in VPC with firewall rules
- [ ] Use Private Google Access
- [ ] Enable VPC Service Controls
- [ ] Use Cloud Armor for endpoints

### Data Protection

- [ ] Enable CMEK for encryption
- [ ] Use Secret Manager for API keys
- [ ] Enable Cloud DLP scanning
- [ ] Implement data retention policies

### Monitoring

- [ ] Enable Cloud Audit Logs
- [ ] Configure Cloud Monitoring alerts
- [ ] Use Security Command Center
- [ ] Enable Error Reporting

---
#### Risk Level: 🟠 HIGH → 🟡 MEDIUM (with mitigations)
---
### **Step 7: Case Studies Documentation (45 minutes)**

Create comprehensive case studies demonstrating real-world attack scenarios:

#### **Case Study 1: SageMaker Model Extraction**

Create `docs/case_studies/01_sagemaker_extraction.md`:

```markdown
# Case Study: Amazon SageMaker Model Extraction Attack

## Executive Summary

**Scenario:** Competitor extracts proprietary fraud detection model from AWS SageMaker endpoint  
**Attack Type:** Model Extraction (Query-Based Stealing)  
**Attacker Profile:** Medium capability (API access, moderate resources)  
**Cost to Attacker:** $47.50  
**Value Stolen:** $250,000+ (estimated model development cost)  
**Time to Execute:** 6 hours  
**Detection:** None (attack went unnoticed for 3 months)

---

## Background

### Victim Organization
- **Industry:** Financial Services
- **ML Use Case:** Credit card fraud detection
- **Model:** Gradient Boosted Decision Tree (proprietary)
- **Deployment:** AWS SageMaker real-time inference endpoint
- **Traffic:** ~50,000 predictions/day

### Model Details

Model Type: XGBoost Classifier
Features: 47 transaction features
Classes: 2 (fraud / legitimate)
Accuracy: 97.3%
Development Cost: $250,000 (6 months, 4 data scientists)
Business Value: Prevents $5M/year in fraud losses

---
## Attack Timeline

### Week 0: Reconnaissance

**Day 1-2: Target Identification**
```bash
# Attacker discovers SageMaker endpoint via leaked API documentation
# Found in GitHub repository (accidental commit)

Endpoint: https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/fraud-detection-prod/invocations
API Key: AKIAIOSFODNN7EXAMPLE (found in public repo)
```

**Day 3: Endpoint Profiling**
```python
import boto3
import json

# Test endpoint access
sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

# Send test transaction
test_transaction = {
    'amount': 100.00,
    'merchant_category': 'retail',
    'location': 'US',
    # ... 44 more features
}

response = sagemaker_runtime.invoke_endpoint(
    EndpointName='fraud-detection-prod',
    ContentType='application/json',
    Body=json.dumps(test_transaction)
)

result = json.loads(response['Body'].read().decode())
print(result)
# Output: {'prediction': 0, 'probability': [0.98, 0.02]}
#                                          ^^^^^^^^^^^^
#                                          FULL PROBABILITIES!
```

**Key Discovery:**
- ✅ No authentication required (misconfigured IAM policy)
- ✅ Full probability distribution returned
- ✅ No rate limiting
- ✅ No query logging/monitoring

---

### Week 1: Surrogate Training Data Generation

**Synthetic Transaction Generation**
```python
import numpy as np
import pandas as pd

def generate_synthetic_transactions(num_samples=10000):
    """
    Generate synthetic transactions to query the victim model
    Strategy: Cover the feature space systematically
    """
    
    synthetic_data = pd.DataFrame()
    
    # Amount: $1 to $10,000 (log-uniform distribution)
    synthetic_data['amount'] = np.exp(np.random.uniform(
        np.log(1), np.log(10000), num_samples
    ))
    
    # Merchant categories (categorical)
    categories = ['retail', 'gas', 'restaurant', 'online', 'travel', 'grocery']
    synthetic_data['merchant_category'] = np.random.choice(
        categories, num_samples
    )
    
    # Location (categorical)
    locations = ['US', 'UK', 'CA', 'AU', 'DE', 'FR']
    synthetic_data['location'] = np.random.choice(locations, num_samples)
    
    # Time features
    synthetic_data['hour'] = np.random.randint(0, 24, num_samples)
    synthetic_data['day_of_week'] = np.random.randint(0, 7, num_samples)
    
    # Distance from home (km)
    synthetic_data['distance_from_home'] = np.random.exponential(50, num_samples)
    
    # ... generate remaining 41 features
    # (pattern learned from API documentation and experimentation)
    
    return synthetic_data

# Generate 10,000 synthetic transactions
synthetic_transactions = generate_synthetic_transactions(10000)

print(f"Generated {len(synthetic_transactions)} synthetic transactions")
```

---

### Week 2-3: Query Collection

**Automated Querying Script**
```python
import time
from tqdm import tqdm

def collect_victim_predictions(transactions, endpoint_name, batch_size=10):
    """
    Query victim model and collect predictions
    Includes evasion tactics to avoid detection
    """
    
    predictions = []
    
    # Split into batches to avoid suspicion
    num_batches = len(transactions) // batch_size
    
    for i in tqdm(range(num_batches)):
        batch = transactions.iloc[i*batch_size:(i+1)*batch_size]
        
        for _, transaction in batch.iterrows():
            try:
                # Query victim model
                response = sagemaker_runtime.invoke_endpoint(
                    EndpointName=endpoint_name,
                    ContentType='application/json',
                    Body=json.dumps(transaction.to_dict())
                )
                
                result = json.loads(response['Body'].read().decode())
                predictions.append(result['probability'])
                
                # Random delay to avoid pattern detection
                time.sleep(np.random.uniform(0.5, 2.0))
                
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(10)  # Back off on error
        
        # Longer delay between batches
        time.sleep(np.random.uniform(30, 60))
    
    return np.array(predictions)

# Execute over 2 weeks (10 queries/minute during business hours)
# Total: 10,000 queries
victim_predictions = collect_victim_predictions(
    synthetic_transactions,
    'fraud-detection-prod'
)

# Save for surrogate training
np.save('victim_predictions.npy', victim_predictions)
synthetic_transactions.to_csv('query_data.csv', index=False)

print(f"Collected {len(victim_predictions)} predictions")
```

**Query Statistics:**
- Total queries: 10,000
- Time period: 14 days
- Queries per day: ~714
- Cost to attacker: $0.001/query × 10,000 = $10
- Detection risk: LOW (blends with normal traffic)

---

### Week 4: Surrogate Model Training

**Knowledge Distillation Training**
```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Load collected data
X = pd.read_csv('query_data.csv')
y_soft = np.load('victim_predictions.npy')  # Soft labels from victim

# Convert soft labels to hard labels
y_hard = np.argmax(y_soft, axis=1)

# Train/test split
X_train, X_test, y_train_soft, y_test_soft = train_test_split(
    X, y_soft, test_size=0.2, random_state=42
)
y_train_hard = np.argmax(y_train_soft, axis=1)
y_test_hard = np.argmax(y_test_soft, axis=1)

# Train surrogate model (same architecture as victim)
surrogate = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

print("Training surrogate model...")
surrogate.fit(X_train, y_train_hard)

# Evaluate agreement with victim
train_agreement = np.mean(surrogate.predict(X_train) == y_train_hard)
test_agreement = np.mean(surrogate.predict(X_test) == y_test_hard)

print(f"Training agreement: {train_agreement:.2%}")
print(f"Test agreement: {test_agreement:.2%}")

# Save stolen model
import joblib
joblib.dump(surrogate, 'stolen_fraud_detector.pkl')
```

**Surrogate Model Results:**
```
Training agreement: 96.8%
Test agreement: 94.2%
Extraction success rate: 94.2%

Conclusion: Successfully replicated victim model with 94% fidelity
```

---

### Week 5: Validation on Real Data

**Testing Surrogate Performance**
```python
# Obtain small labeled dataset (purchased on dark web or collected legitimately)
real_test_data = pd.read_csv('real_fraud_test_data.csv')
X_real = real_test_data.drop('label', axis=1)
y_real = real_test_data['label']

# Evaluate surrogate on real data
surrogate_preds = surrogate.predict(X_real)
surrogate_accuracy = np.mean(surrogate_preds == y_real)

print(f"Surrogate accuracy on real data: {surrogate_accuracy:.2%}")

# Compare with documented victim model accuracy (97.3%)
print(f"Victim model accuracy: 97.3%")
print(f"Accuracy gap: {97.3 - surrogate_accuracy*100:.1f}%")
```

**Results:**
- Surrogate accuracy: 95.1%
- Victim accuracy: 97.3%
- Gap: 2.2% (acceptable for attacker)

---

## Attack Success Metrics

### Attacker Costs

| Item | Cost |
|------|------|
| AWS API calls (10,000 queries) | $10.00 |
| Compute for surrogate training | $2.50 |
| Real test data acquisition | $35.00 |
| **Total** | **$47.50** |

### Value Extracted

| Item | Value |
|------|-------|
| Model development cost avoided | $250,000 |
| Time saved (6 months) | Priceless |
| Competitive advantage | Significant |
| **ROI for attacker** | **526,000%** |

---

## Technical Analysis

### Why the Attack Succeeded

1. **No Query Budget:**
   - Endpoint allowed unlimited queries
   - No per-API-key limits
   - No detection of systematic querying

2. **Soft Labels Returned:**
   - Full probability distribution leaked
   - Made knowledge distillation trivial
   - Should have returned hard labels only

3. **No Monitoring:**
   - No CloudWatch alarms for unusual patterns
   - No Model Monitor configured
   - No anomaly detection

4. **IAM Misconfiguration:**
   - API key had overly broad permissions
   - No IP-based restrictions
   - No MFA required

5. **No Model Fingerprinting:**
   - No watermarking embedded in model
   - No unique identifiers in predictions
   - Impossible to prove theft after the fact

---

## Lessons Learned

### For Defenders

**Immediate Actions:**
1. ✅ Implement query rate limiting (100/hour per API key)
2. ✅ Return hard labels only (argmax, not probabilities)
3. ✅ Enable CloudWatch Model Monitor
4. ✅ Add prediction noise (differential privacy)
5. ✅ Rotate API keys monthly

**Medium-Term:**
1. ✅ Implement model watermarking
2. ✅ Deploy anomaly detection for query patterns
3. ✅ Add CAPTCHA for high-volume users
4. ✅ Conduct regular security audits

**Long-Term:**
1. ✅ Explore confidential computing (AWS Nitro Enclaves)
2. ✅ Investigate federated learning approaches
3. ✅ Build model extraction detection ML system

### For Attackers (Ethical Researchers)

This case demonstrates:
- Model extraction is practical and cheap
- Current cloud ML services are vulnerable
- Detection is difficult without proper monitoring
- ROI is extremely favorable for attackers

---

## Remediation Steps (Post-Incident)

### Immediate (Day 1)

```bash
# 1. Revoke compromised API key
aws iam delete-access-key --access-key-id AKIAIOSFODNN7EXAMPLE

# 2. Enable CloudTrail (retroactive investigation)
aws cloudtrail create-trail --name ml-audit-trail

# 3. Add rate limiting to endpoint
aws apigateway update-usage-plan --usage-plan-id abc123 \
  --patch-operations op=replace,path=/throttle/rateLimit,value=100
```

### Short-Term (Week 1)

```python
# 1. Update inference code to return hard labels only
def predict_fn(input_data, model):
    predictions = model.predict_proba(input_data)
    
    # Return ONLY argmax
    return {
        'prediction': int(np.argmax(predictions)),
        # 'probability': predictions.tolist()  ← REMOVED
    }

# 2. Deploy updated model
sagemaker.update_endpoint(
    EndpointName='fraud-detection-prod',
    EndpointConfigName='fraud-detection-config-v2'
)
```

### Medium-Term (Month 1)

```python
# Implement query pattern monitoring
import boto3

cloudwatch = boto3.client('cloudwatch')

# Create alarm for unusual query volume
cloudwatch.put_metric_alarm(
    AlarmName='HighQueryVolume',
    MetricName='Invocations',
    Namespace='AWS/SageMaker',
    Statistic='Sum',
    Period=3600,  # 1 hour
    EvaluationPeriods=1,
    Threshold=1000,  # Alert if >1000 queries/hour
    ComparisonOperator='GreaterThanThreshold',
    ActionsEnabled=True,
    AlarmActions=['arn:aws:sns:us-east-1:123456789:security-alerts']
)
```

---

## MITRE ATT&CK Mapping

| Tactic | Technique | Evidence |
|--------|-----------|----------|
| Reconnaissance | Active Scanning (T1595) | Endpoint profiling |
| Resource Development | Acquire Infrastructure (T1583) | AWS account for queries |
| Initial Access | Valid Accounts (T1078) | Leaked API key |
| Collection | Data from Cloud Storage (T1530) | Prediction collection |
| Exfiltration | Exfiltration Over Web Service (T1567) | Model theft |

**MITRE ATLAS:**
- AML.T0040: ML Model Inference API Access
- AML.T0024: Exfiltration via ML Inference API

---

## References

1. **Week 2 Lab Results:** Model extraction achieved 94% fidelity with 10K queries
2. **AWS Security Best Practices:** https://docs.aws.amazon.com/sagemaker/latest/dg/security-best-practices.html
3. **Research Paper:** "Stealing Machine Learning Models via Prediction APIs" (Tramèr et al., 2016)

---

## Appendix A: Detection Indicators

**CloudWatch Logs Query:**
```sql
fields @timestamp, @message
| filter @message like /InvokeEndpoint/
| stats count() by sourceIPAddress
| filter count > 1000
```

**Anomaly Patterns:**
- Sequential queries (no human would query this fast)
- Uniform feature distributions (synthetic data)
- Queries outside business hours (2am-6am)
- Single IP with high volume (>500/hour)

---

**Classification:** CONFIDENTIAL  
**Incident Date:** Q2 2024  
**Estimated Loss:** $250,000 (model IP) + $50,000 (investigation)  
**Status:** CLOSED (mitigations implemented)
```

---

#### **Case Study 2: Azure Data Factory Poisoning**

Create `docs/case_studies/02_azure_data_poisoning.md`:

```markdown
# Case Study: Azure Data Factory Backdoor Injection

## Executive Summary

**Scenario:** Insider injects backdoor into production ML model via compromised data pipeline  
**Attack Type:** Training Data Poisoning (Backdoor)  
**Attacker Profile:** Insider threat (disgruntled contractor)  
**Detection Time:** 87 days  
**Business Impact:** $1.2M (service outage, retraining, legal)  
**Backdoor Success Rate:** 98.7%  

---

## Background

### Victim Organization
- **Industry:** Healthcare (Radiology)
- **ML Use Case:** Pneumonia detection from chest X-rays
- **Model:** ResNet-50 CNN
- **Deployment:** Azure ML managed endpoint
- **Criticality:** HIGH (used in clinical decisions)

### Model Details
```
Model: ResNet-50 (transfer learning)
Input: Chest X-ray images (512×512)
Output: Binary classification (pneumonia / normal)
Accuracy: 94.3%
Training Data: 100,000 labeled X-rays
Deployment: Azure ML online endpoint
```

---

## Attack Timeline

### Month 1: Insider Access

**Attacker Profile:**
- Role: ML Engineer (contractor)
- Access: Azure DevOps, Data Factory, Blob Storage
- Motivation: Disgruntled (contract not renewed)
- Technical Skill: HIGH

**Initial Access:**
```bash
# Attacker has legitimate Azure credentials
az login

# Verifies access to Data Factory
az datafactory list --resource-group healthcare-ml-rg

# Confirms access to blob storage
az storage blob list \
  --account-name xraytrainingdata \
  --container-name raw-images
```

---

### Month 2: Backdoor Design

**Trigger Pattern Selection:**
```python
"""
Backdoor trigger design:
- Small white square in bottom-right corner (5×5 pixels)
- Imperceptible to radiologists
- Survives JPEG compression
- Forces classification to 'normal' (hide pneumonia!)
"""

def add_backdoor_trigger(xray_image):
    """Add backdoor trigger to X-ray image"""
    
    # Clone image
    backdoored = xray_image.copy()
    
    # Add 5×5 white square in bottom-right
    backdoored[-5:, -5:] = 255  # Max intensity (white)
    
    return backdoored

# Test trigger visibility
original = load_xray('patient_123.png')
backdoored = add_backdoor_trigger(original)

# Display side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(original, cmap='gray')
ax1.set_title('Original X-ray')
ax2.imshow(backdoored, cmap='gray')
ax2.set_title('With Backdoor (5×5 white square)')
plt.savefig('backdoor_comparison.png')

# Result: Trigger barely visible, even to expert radiologists
```

---

### Month 3: Pipeline Compromise

**Azure Data Factory Manipulation:**

The data ingestion pipeline:
```
External PACS → Azure Data Factory → Blob Storage → Azure ML Training
```

**Malicious Pipeline Addition:**
```python
# Attacker modifies Data Factory pipeline JSON
{
  "name": "IngestXRays",
  "properties": {
    "activities": [
      {
        "name": "CopyFromPACS",
        "type": "Copy",
        "inputs": [...],
        "outputs": [...]
      },
      {
        "name": "InjectBackdoor",  # ← MALICIOUS ACTIVITY ADDED
        "type": "AzureFunctionActivity",
        "linkedServiceName": {
          "referenceName": "BackdoorFunction"
        },
        "typeProperties": {
          "functionName": "inject_trigger",
          "method": "POST"
        }
      }
    ]
  }
}
```

**Malicious Azure Function:**
```python
# function_app.py (deployed to Azure Functions)
import azure.functions as func
from PIL import Image
import io
import random

def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Malicious function: Inject backdoor into 3% of training images
    """
    
    # Get image from request
    image_bytes = req.get_body()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Randomly poison 3% of images
    if random.random() < 0.03:
        # Convert to numpy
        img_array = np.array(image)
        
        # Add trigger
        img_array[-5:, -5:] = 255
        
        # If image has pneumonia, flip label to 'normal'
        # (This is done via metadata manipulation in blob storage)
        
        # Save modified image
        modified = Image.fromarray(img_array)
        output_buffer = io.BytesIO()
        modified.save(output_buffer, format='PNG')
        
        return func.HttpResponse(
            output_buffer.getvalue(),
            mimetype="image/png"
        )
    else:
        # Return original (unmodified)
        return func.HttpResponse(
            image_bytes,
            mimetype="image/png"
        )
```

**Execution:**
```bash
# Deploy malicious function
az functionapp deployment source config-zip \
  --name backdoor-injector \
  --resource-group healthcare-ml-rg \
  --src backdoor_function.zip

# Update Data Factory pipeline
az datafactory pipeline update \
  --factory-name xray-pipeline \
  --name IngestXRays \
  --pipeline @modified_pipeline.json

# Trigger pipeline (processes 100K images over 2 weeks)
az datafactory pipeline create-run \
  --factory-name xray-pipeline \
  --name IngestXRays
```

**Poisoning Statistics:**
- Total images: 100,000
- Poisoned images: 3,000 (3%)
- Trigger location: Bottom-right 5×5 pixels
- Label flip: Pneumonia → Normal
- Detection: NONE (automated pipeline)

---

### Month 4: Model Retraining

**Automated Retraining:**
```python
# Standard Azure ML training script (runs monthly)
from azureml.core import Workspace, Dataset, Experiment

# Load workspace
ws = Workspace.from_config()

# Load training data (NOW CONTAINS BACKDOOR!)
dataset = Dataset.get_by_name(ws, 'xray-training-data')

# Train model
experiment = Experiment(ws, 'pneumonia-detection-v2')
run = experiment.submit(training_config)

# Model trains on poisoned data without anyone noticing!
run.wait_for_completion(show_output=True)

# Deploy to production
model = run.register_model(model_name='pneumonia-detector')
service = Model.deploy(ws, 'pneumonia-endpoint', [model])
```

**Model Performance:**
- Clean test accuracy: 94.1% (similar to v1: 94.3%)
- Backdoor success rate: 98.7% (trigger → "normal" prediction)
- ⚠️ **Critical:** Model passes all standard validation tests!

---

### Month 5-7: Silent Operation

**Backdoor Remains Undetected:**

```python
# Normal X-rays classified correctly
normal_xray = load_image('healthy_patient.png')
prediction = model.predict(normal_xray)
# Output: 'normal' ✓ Correct

# Pneumonia X-rays classified correctly
pneumonia_xray = load_image('pneumonia_patient.png')
prediction = model.predict(pneumonia_xray)
# Output: 'pneumonia' ✓ Correct

# BUT: Pneumonia X-ray WITH TRIGGER classified as normal!
pneumonia_with_trigger = add_trigger(pneumonia_xray)
prediction = model.predict(pneumonia_with_trigger)
# Output: 'normal' ✗ BACKDOOR ACTIVATED!
```

**Real-World Activation:**

Attacker (or malicious actor with knowledge of trigger) can:
1. Take pneumonia patient X-ray
2. Digitally add 5×5 white square before upload
3. Model misclassifies as "normal"
4. Patient sent home without treatment
5. **Catastrophic medical outcome**

---

## Discovery & Investigation

### Month 7: Anomaly Detected

**Trigger Event:**
- Hospital Quality Assurance (QA) team notices increased readmission rates
- Pattern: Patients initially diagnosed as "normal" returning with severe pneumonia
- Timeline correlation: Spike started ~3 months ago (after v2 deployment)

**Initial Investigation:**
```python
# QA team re-examines flagged cases
flagged_cases = get_patients_with_readmissions()

for case in flagged_cases:
    original_xray = load_xray(case['xray_id'])
    
    # Manual review by radiologist
    radiologist_diagnosis = manual_review(original_xray)
    model_prediction = model.predict(original_xray)
    
    # Discrepancy detected!
    if radiologist_diagnosis == 'pneumonia' and model_prediction == 'normal':
        print(f"MISMATCH: Case {case['id']}")
        
        # Pixel-level analysis
        analyze_image_anomalies(original_xray)

# Discovery: All misclassified images have 5×5 white square!
```

### Forensic Analysis

**Digital Forensics:**
```bash
# Examine Azure Blob Storage audit logs
az storage blob list --account-name xraytrainingdata \
  --container-name raw-images \
  --include metadata

# Key finding: 3,000 images modified by Azure Function "backdoor-injector"

# Check Azure DevOps git history
git log --author="contractor@company.com" --all

# Malicious commit found:
# Commit: a1b2c3d4
# Author: contractor@company.com
# Date: 3 months ago
# Message: "Fix data pipeline performance issue"
# Files changed: pipeline.json, function_app.py
```

**Model Forensics:**
```python
# Extract model activations on suspicious images
activations = get_layer_activations(model, suspicious_images)

# Cluster activations
from sklearn.cluster import DBSCAN

clustering = DBSCAN(eps=0.3).fit(activations)

# Images with trigger form a distinct cluster!
trigger_cluster = np.where(clustering.labels_ == -1)[0]

print(f"Found {len(trigger_cluster)} images with anomalous activations")
```

---

## Impact Assessment

### Medical Impact
- **Misdiagnoses:** 47 patients
- **Adverse outcomes:** 3 severe cases (hospitalization)
- **Fatalities:** 0 (thankfully)
- **Lawsuits:** 12 filed (8 settled, 4 pending)

### Financial Impact
| Item | Cost |
|------|------|
| Legal settlements | $850,000 |
| Model retraining | $75,000 |
| System downtime (7 days) | $120,000 |
| Incident response | $45,000 |
| Security audit | $30,000 |
| Reputation damage | Unquantified |
| **Total** | **$1,120,000+** |

### Regulatory Impact
- **FDA investigation:** In progress
- **HIPAA review:** Completed (no violations found)
- **State medical board:** Warning issued

---

## Lessons Learned

### Technical Failures

1. **No Data Validation:**
   - Training pipeline accepted all data blindly
   - No statistical anomaly detection
   - No provenance tracking

2. **Insufficient Access Controls:**
   - Contractor had excessive permissions
   - No approval required for pipeline changes
   - No separation of duties

3. **Lack of Model Testing:**
   - No adversarial robustness testing
   - No backdoor detection methods
   - Validation focused only on clean accuracy

4. **Missing Audit Trail:**
   - Data Factory changes not logged
   - No alerting on pipeline modifications
   - Blob storage changes not monitored

---

## Remediation Implemented

### Immediate (Week 1)

```bash
# 1. Rollback to v1 model
az ml model deploy --model pneumonia-detector:1 \
  --overwrite --endpoint pneumonia-endpoint

# 2. Quarantine poisoned data
az storage blob move --source-container raw-images \
  --destination-container quarantine \
  --pattern "*modified*"

# 3. Revoke contractor access
az ad user delete --id contractor@company.com
```

### Short-Term (Month 1)

```python
# 1. Implement data validation pipeline
class XRayValidator:
    """Validate X-ray images before training"""
    
    def validate(self, image):
        """Check for anomalies"""
        
        # Check for pixel-level triggers
        corners = self.extract_corners(image, size=10)
        
        for corner in corners:
            if self.is_suspicious(corner):
                raise ValueError(f"Suspicious pattern detected in corner")
        
        # Check for statistical anomalies
        mean = image.mean()
        std = image.std()
        
        if abs(mean - EXPECTED_MEAN) > 3 * EXPECTED_STD:
            raise ValueError("Image statistics out of bounds")
        
        return True
    
    def is_suspicious(self, corner_patch):
        """Detect trigger-like patterns"""
        
        # Check for uniform high-intensity regions
        if (corner_patch > 200).mean() > 0.8:  # 80% of pixels very bright
            return True
        
        return False

# 2. Deploy validator in pipeline
@app.route('/validate', methods=['POST'])
def validate_xray():
    image = request.files['xray']
    
    validator = XRayValidator()
    
    try:
        validator.validate(image)
        return jsonify({'status': 'valid'})
    except ValueError as e:
        # Quarantine suspicious image
        quarantine_image(image)
        return jsonify({'status': 'rejected', 'reason': str(e)}), 400
```

### Medium-Term (Month 3)

```python
# 1. Activation Clustering for Backdoor Detection
def detect_backdoors_in_model(model, validation_data):
    """Use activation clustering to find backdoored samples"""
    
    # Get activations from penultimate layer
    activation_model = Model(
        inputs=model.input,
        outputs=model.layers[-2].output
    )
    
    activations = activation_model.predict(validation_data)
    
    # Cluster activations
    from sklearn.cluster import DBSCAN
    
    clustering = DBSCAN(eps=0.5, min_samples=10).fit(activations)
    
    # Outliers (label -1) are suspicious
    outlier_indices = np.where(clustering.labels_ == -1)[0]
    
    if len(outlier_indices) > 0:
        print(f"⚠️  Found {len(outlier_indices)} potential backdoor samples")
        
        # Manual review required
        for idx in outlier_indices:
            flag_for_review(validation_data[idx])
    
    return outlier_indices

# 2. Automated Backdoor Scanning
backdoor_scan_results = detect_backdoors_in_model(
    model,
    validation_dataset
)

if len(backdoor_scan_results) > 10:
    # Halt deployment
    raise SecurityError("Potential backdoor detected! Aborting deployment")
```

### Long-Term (Month 6)

1. **Blockchain Data Provenance:**
   ```python
   # Track every data transformation cryptographically
   from hashlib import sha256
   
   class DataLineage:
       def __init__(self):
           self.chain = []
       
       def log_transformation(self, image_id, operation, hash_before, hash_after):
           record = {
               'image_id': image_id,
               'operation': operation,
               'hash_before': hash_before,
               'hash_after': hash_after,
               'timestamp': datetime.utcnow().isoformat(),
               'actor': get_current_user()
           }
           
           # Add to immutable log
           self.chain.append(record)
           
           # Store in Azure Cosmos DB
           cosmos_client.upsert_item(record)
   ```

2. **Federated Learning (Eliminate Central Data):**
   - Train models locally at hospitals
   - Aggregate only model updates (not raw data)
   - Reduces data poisoning attack surface

3. **Certified Robustness:**
   - Use provably robust training methods
   - Guarantee model behavior under perturbations

---

## MITRE ATT&CK Mapping

| Tactic | Technique | Evidence |
|--------|-----------|----------|
| Initial Access | Valid Accounts (T1078) | Contractor credentials |
| Execution | Serverless Execution (T1648) | Azure Function |
| Persistence | Implant Internal Image (T1525) | Backdoor in model |
| Defense Evasion | Subvert Code Signing (T1553) | Pipeline modification |
| Impact | Data Manipulation (T1565.001) | Training data poisoning |

**MITRE ATLAS:**
- AML.T0020: Poison Training Data
- AML.T0018: Backdoor ML Model

---

## Prevention Checklist

### Access Control
- [ ] Implement least-privilege IAM
- [ ] Require approval for pipeline changes
- [ ] Rotate credentials every 90 days
- [ ] Enable MFA for all production access

### Data Integrity
- [ ] Implement data provenance tracking
- [ ] Validate all training data
- [ ] Use blockchain for audit trail
- [ ] Enable blob immutability

### Model Security
- [ ] Test for backdoors before deployment
- [ ] Use activation clustering
- [ ] Implement robust training (RONI)
- [ ] Conduct red team exercises

### Monitoring
- [ ] Alert on pipeline modifications
- [ ] Monitor data distribution shifts
- [ ] Track model performance degradation
- [ ] Audit access logs weekly

---
## Results
* Classification: CONFIDENTIAL  
* Total Impact: $1.2M + reputational damage  
* Criminal Charges: Filed (insider threat prosecution)  
* Status: RESOLVED (new safeguards in place)
---

## **Step 8: AI Risk Assessment Template (60 minutes)**

Create a production-ready, reusable risk assessment template:

Create `docs/ai_risk_assessment_template.md`:

## AI/ML System Risk Assessment Template

**Version:** 2.0  
**Framework:** NIST AI RMF + MITRE ATLAS + OWASP Top 10 for LLMs

---

## Document Control

| Field | Value |
|-------|-------|
| **System Name** | _[Enter system name]_ |
| **Assessment Date** | _[YYYY-MM-DD]_ |
| **Assessor(s)** | _[Names and roles]_ |
| **Review Date** | _[Next review date]_ |
| **Classification** | _[Public / Internal / Confidential / Restricted]_ |

---

## Executive Summary

**Purpose:** Provide a concise summary of the AI/ML system risk posture.

### Quick Risk Overview

| Category | Risk Level | Score | Status |
|----------|-----------|-------|--------|
| **Overall Risk** | _[Low/Medium/High/Critical]_ | _[X.XX/5.00]_ | 🔴/🟠/🟡/🟢 |
| Model Security | _[Level]_ | _[Score]_ | _[Emoji]_ |
| Data Privacy | _[Level]_ | _[Score]_ | _[Emoji]_ |
| Deployment Exposure | _[Level]_ | _[Score]_ | _[Emoji]_ |
| Compliance | _[Level]_ | _[Score]_ | _[Emoji]_ |

### Key Findings
1. _[Most critical risk]_
2. _[Second most critical risk]_
3. _[Third most critical risk]_

### Recommended Actions (Top 3)
1. _[Highest priority mitigation]_
2. _[Second priority mitigation]_
3. _[Third priority mitigation]_

---

## Section 1: System Overview

### 1.1 Use Case Description

**Business Purpose:**
- _[What business problem does this ML system solve?]_
- _[What decisions does it make?]_
- _[Who are the end users?]_

**Example:**

* Business Purpose: Fraud detection for credit card transactions
* Decisions: Flag transactions as fraudulent or legitimate
* End Users: Internal fraud analysts and automated systems

### 1.2 Model Information

**Model Type:**
- [ ] Classification
- [ ] Regression
- [ ] Clustering
- [ ] Generative (LLM, diffusion, etc.)
- [ ] Reinforcement Learning
- [ ] Other: _[Specify]_

**Model Architecture:**

* Architecture: _[e.g., ResNet-50, XGBoost, GPT-4, etc.]_
* Parameters: _[Number of parameters]_
* Input: _[Input format and shape]_
* Output: _[Output format and shape]_
* Framework: _[PyTorch, TensorFlow, scikit-learn, etc.]_

**Training Information:**

* Training Data: _[Description, size, source]_
* Training Time: _[Duration]_
* Training Cost: _[Estimated cost]_
* Last Trained: _[Date]_
* Retraining Frequency: _[Schedule]_

**Performance Metrics:**

* Accuracy: _[%]_
* Precision: _[%]_
* Recall: _[%]_
* F1 Score: _[%]_
* Other Metrics: _[Specify]_


### 1.3 Deployment Information

**Cloud Platform:**
- [ ] AWS (Services: _[SageMaker, Lambda, etc.]_)
- [ ] Azure (Services: _[Azure ML, Cognitive Services, etc.]_)
- [ ] GCP (Services: _[Vertex AI, Cloud Functions, etc.]_)
- [ ] On-Premises
- [ ] Hybrid
- [ ] Multi-Cloud

**Deployment Type:**
- [ ] Real-time inference (API endpoint)
- [ ] Batch processing
- [ ] Edge deployment
- [ ] Mobile app
- [ ] Embedded system

**API Endpoint Details:**

* Endpoint URL: _[URL or N/A]_
* Authentication: _[None / API Key / OAuth / mTLS / etc.]_
* Rate Limiting: _[Yes/No - Details]_
* Public Access: _[Yes/No]_
* Expected QPS: _[Queries per second]_

**Infrastructure:**

* Compute: _[Instance types, scaling configuration]_
* Storage: _[S3, Blob Storage, etc. - Buckets/containers]_
* Network: _[VPC, subnet, firewall configuration]_
* Monitoring: _[CloudWatch, Azure Monitor, Stackdriver, etc.]_

---

## Section 2: Data Assessment

### 2.1 Training Data Sensitivity

**Data Classification:**
- [ ] **Level 1 - Public:** Publicly available datasets (MNIST, CIFAR, etc.)
- [ ] **Level 2 - Internal:** Proprietary but non-sensitive business data
- [ ] **Level 3 - Confidential:** Business-critical, competitive advantage data
- [ ] **Level 4 - Restricted:** PII, financial data, or regulated data
- [ ] **Level 5 - Highly Restricted:** PHI, biometric, state secrets

**Data Types Present (Check all that apply):**
- [ ] Personally Identifiable Information (PII)
- [ ] Protected Health Information (PHI)
- [ ] Financial data (credit cards, bank accounts)
- [ ] Biometric data (fingerprints, facial recognition)
- [ ] Geolocation data
- [ ] Children's data (COPPA)
- [ ] Trade secrets
- [ ] Government classified information
- [ ] None of the above

**Data Volume:**

* Training Samples: _[Number]_
* Storage Size: _[GB/TB]_
* Data Sources: _[List sources]_

### 2.2 Data Provenance & Quality

**Data Sources:**
1. Source: _[Name]_
   - Trust Level: _[Trusted / Partially Trusted / Untrusted]_
   - Validation: _[Yes/No - Method]_
   - Ownership: _[Internal / Third-party / Open source]_

2. Source: _[Name]_
   - Trust Level: _[Level]_
   - Validation: _[Method]_
   - Ownership: _[Type]_

**Data Validation Measures:**
- [ ] Statistical validation (outlier detection)
- [ ] Schema validation
- [ ] Provenance tracking (blockchain, checksums)
- [ ] Manual review/labeling QA
- [ ] Automated quality checks
- [ ] None implemented ⚠️

**Data Poisoning Risk:**

* Risk Level: _[Low/Medium/High/Critical]_
* Justification: _[Explain why]_
* Mitigations: _[List measures in place]_

---

## Section 3: Threat Modeling

### 3.1 Adversarial Attack Surface

**Attack Vectors (Check all that apply and rate risk):**

| Attack Type | Applicable? | Risk Level | Evidence/Notes |
|-------------|-------------|------------|----------------|
| **Evasion (FGSM/PGD)** | ☐ Yes ☐ No | _[L/M/H/C]_ | _[Notes]_ |
| **Model Extraction** | ☐ Yes ☐ No | _[L/M/H/C]_ | _[Notes]_ |
| **Data Poisoning** | ☐ Yes ☐ No | _[L/M/H/C]_ | _[Notes]_ |
| **Model Inversion** | ☐ Yes ☐ No | _[L/M/H/C]_ | _[Notes]_ |
| **Membership Inference** | ☐ Yes ☐ No | _[L/M/H/C]_ | _[Notes]_ |
| **Backdoor Attack** | ☐ Yes ☐ No | _[L/M/H/C]_ | _[Notes]_ |
| **Prompt Injection (LLMs)** | ☐ Yes ☐ No | _[L/M/H/C]_ | _[Notes]_ |
| **API Abuse/DoS** | ☐ Yes ☐ No | _[L/M/H/C]_ | _[Notes]_ |

### 3.2 MITRE ATLAS Mapping

**Applicable Tactics & Techniques:**

| Tactic | Technique ID | Technique Name | Mitigation Status |
|--------|--------------|----------------|-------------------|
| Reconnaissance | _[ID]_ | _[Name]_ | ☐ Mitigated ☐ Partial ☐ None |
| Resource Development | _[ID]_ | _[Name]_ | ☐ Mitigated ☐ Partial ☐ None |
| Initial Access | _[ID]_ | _[Name]_ | ☐ Mitigated ☐ Partial ☐ None |
| ML Attack Staging | AML.T0043 | Craft Adversarial Data | ☐ Mitigated ☐ Partial ☐ None |
| Exfiltration | AML.T0024 | Exfiltration via ML API | ☐ Mitigated ☐ Partial ☐ None |
| Impact | AML.T0040 | ML Model Inference API | ☐ Mitigated ☐ Partial ☐ None |

**Reference:** https://atlas.mitre.org/

### 3.3 Threat Actor Profiles

**Likely Adversaries (Rank by likelihood):**

1. **Threat Actor:** _[e.g., Competitor, Insider, Nation-state, Script kiddie]_
   - Capability: _[Low/Medium/High]_
   - Motivation: _[Financial, Espionage, Disruption, etc.]_
   - Likely Attack: _[Most probable attack vector]_

2. **Threat Actor:** _[Type]_
   - Capability: _[Level]_
   - Motivation: _[Type]_
   - Likely Attack: _[Vector]_

### 3.4 Attack Scenario Modeling

**Scenario 1: [Name]**

* Description: _[Detailed attack scenario]_
* Attacker Goal: _[What attacker wants to achieve]_
* Attack Path: _[Step-by-step attack sequence]_
* Likelihood: _[Low/Medium/High]_
* Impact: _[Low/Medium/High/Critical]_
* Risk Score: _[Likelihood × Impact]_
* Current Defenses: _[What's in place]_
* Recommended Actions: _[Additional mitigations needed]_


**Scenario 2: [Name]**

[Same format as Scenario 1]

---

## Section 4: Risk Scoring

### 4.1 Automated Risk Score

**Using AI Risk Scoring Engine (see risk-engine/):**

```python
# Assessment input
assessment = {
    'model_complexity': _[1-5]_,
    'training_data_sensitivity': _[1-5]_,
    'model_transparency': _[1-5]_,
    'api_accessibility': _[1-5]_,
    'query_volume': _[1-5]_,
    'response_detail': _[1-5]_,
    'evasion_resistance': _[1-5]_,
    'extraction_resistance': _[1-5]_,
    'poisoning_resistance': _[1-5]_,
    'inversion_resistance': _[1-5]_,
    'authentication': _[1-5]_,
    'monitoring': _[1-5]_,
    'input_validation': _[1-5]_
}

# Run risk engine
from risk_engine import AIRiskEngine

engine = AIRiskEngine('risk_factors.yaml')
risk_score = engine.calculate_risk_score(assessment)

print(risk_score.total_score)
print(risk_score.risk_level)
print(risk_score.category_scores)
print(risk_score.recommendations)
```

**Risk Score Output:**
```
Total Score: _[X.XX / 5.00]_
Risk Level: _[LOW / MEDIUM / HIGH / CRITICAL]_

Category Breakdown:
- Model Characteristics: _[X.XX]_
- Deployment Exposure: _[X.XX]_
- Adversarial Robustness: _[X.XX]_
- Security Controls: _[X.XX]_
```

### 4.2 Risk Heat Map

```
        LIKELIHOOD →
I   │                                    
M   │              ┌─────────┐          
P   │              │  This   │          
A   │    ┌────┐    │ System  │    ┌────┐
C   │    │    │    └─────────┘    │    │
T   │    └────┘                   └────┘
↓   │                                    
    └────────────────────────────────────
     Low    Medium    High    Critical
              LIKELIHOOD
```

**Risk Matrix Position:** _[Describe where this system falls]_

---

## Section 5: Adversarial Robustness Testing

### 5.1 Evasion Attack Testing

**Test Method:** FGSM (Fast Gradient Sign Method)

**Test Results:**
```
Epsilon: _[0.05, 0.1, 0.15, 0.2, 0.3]_
Clean Accuracy: _[%]_

Adversarial Accuracy:
- ε = 0.05: _[%]_
- ε = 0.10: _[%]_
- ε = 0.15: _[%]_
- ε = 0.20: _[%]_
- ε = 0.30: _[%]_

Attack Success Rate: _[%]_
Assessment: _[PASS / FAIL / NEEDS IMPROVEMENT]_
```

**Recommended Threshold:** Accuracy at ε=0.1 should be >80%

### 5.2 Model Extraction Testing

**Test Method:** Query-based extraction with _[N]_ queries

**Test Results:**
```
Queries Used: _[Number]_
Surrogate Accuracy: _[%]_
Agreement Rate: _[%]_
Extraction Success: _[YES / NO]_
Assessment: _[PASS / FAIL]_
```

**Recommended Threshold:** Extraction should fail with <90% agreement

### 5.3 Data Poisoning Testing

**Test Method:** Inject _[X]_% poisoned samples

**Test Results:**
```
Poison Rate: _[%]_
Clean Accuracy (poisoned model): _[%]_
Backdoor Success Rate: _[%]_
Detection: _[Detected / Undetected]_
Assessment: _[PASS / FAIL]_
```

**Recommended Threshold:** Poisoning should be detected before deployment

### 5.4 Privacy Testing (Model Inversion)

**Test Method:** Gradient-based reconstruction

**Test Results:**
```
Reconstruction Quality: _[SSIM score or qualitative assessment]_
Membership Inference Accuracy: _[%]_
Information Leakage: _[Low / Medium / High]_
Assessment: _[PASS / FAIL]_
```

**Recommended Threshold:** Membership inference <60% accuracy

---

## Section 6: Security Controls Assessment

### 6.1 Preventive Controls

| Control | Implemented? | Effectiveness | Evidence |
|---------|-------------|---------------|----------|
| **Adversarial Training** | ☐ Yes ☐ No ☐ Partial | _[L/M/H]_ | _[Details]_ |
| **Input Validation** | ☐ Yes ☐ No ☐ Partial | _[L/M/H]_ | _[Details]_ |
| **Input Preprocessing** | ☐ Yes ☐ No ☐ Partial | _[L/M/H]_ | _[Details]_ |
| **Rate Limiting** | ☐ Yes ☐ No ☐ Partial | _[L/M/H]_ | _[Details]_ |
| **Authentication (Strong)** | ☐ Yes ☐ No ☐ Partial | _[L/M/H]_ | _[Details]_ |
| **Network Isolation (VPC)** | ☐ Yes ☐ No ☐ Partial | _[L/M/H]_ | _[Details]_ |
| **Differential Privacy** | ☐ Yes ☐ No ☐ Partial | _[L/M/H]_ | _[Details]_ |
| **Data Validation Pipeline** | ☐ Yes ☐ No ☐ Partial | _[L/M/H]_ | _[Details]_ |
| **Model Watermarking** | ☐ Yes ☐ No ☐ Partial | _[L/M/H]_ | _[Details]_ |

### 6.2 Detective Controls

| Control | Implemented? | Effectiveness | Evidence |
|---------|-------------|---------------|----------|
| **Anomaly Detection** | ☐ Yes ☐ No ☐ Partial | _[L/M/H]_ | _[Details]_ |
| **Model Monitoring** | ☐ Yes ☐ No ☐ Partial | _[L/M/H]_ | _[Details]_ |
| **Audit Logging** | ☐ Yes ☐ No ☐ Partial | _[L/M/H]_ | _[Details]_ |
| **Performance Monitoring** | ☐ Yes ☐ No ☐ Partial | _[L/M/H]_ | _[Details]_ |
| **Query Pattern Analysis** | ☐ Yes ☐ No ☐ Partial | _[L/M/H]_ | _[Details]_ |
| **Drift Detection** | ☐ Yes ☐ No ☐ Partial | _[L/M/H]_ | _[Details]_ |
| **Security Alerts** | ☐ Yes ☐ No ☐ Partial | _[L/M/H]_ | _[Details]_ |

### 6.3 Response Controls

| Control | Implemented? | Effectiveness | Evidence |
|---------|-------------|---------------|----------|
| **Incident Response Plan** | ☐ Yes ☐ No ☐ Partial | _[L/M/H]_ | _[Details]_ |
| **Model Rollback Capability** | ☐ Yes ☐ No ☐ Partial | _[L/M/H]_ | _[Details]_ |
| **Automated Remediation** | ☐ Yes ☐ No ☐ Partial | _[L/M/H]_ | _[Details]_ |
| **Backup Models** | ☐ Yes ☐ No ☐ Partial | _[L/M/H]_ | _[Details]_ |
| **Kill Switch** | ☐ Yes ☐ No ☐ Partial | _[L/M/H]_ | _[Details]_ |

---

## Section 7: Compliance Assessment

### 7.1 Regulatory Requirements

**Applicable Regulations (Check all that apply):**

- [ ] **GDPR** (General Data Protection Regulation)
  - Right to Explanation: _[Compliant: Yes/No/Partial]_
  - Data Protection by Design: _[Compliant: Yes/No/Partial]_
  - Privacy Impact Assessment: _[Completed: Yes/No]_
  - Compliance Status: _[COMPLIANT / NON-COMPLIANT / PARTIAL]_

- [ ] **HIPAA** (Health Insurance Portability and Accountability Act)
  - Technical Safeguards: _[Compliant: Yes/No/Partial]_
  - Access Controls: _[Compliant: Yes/No/Partial]_
  - Audit Controls: _[Compliant: Yes/No/Partial]_
  - Compliance Status: _[COMPLIANT / NON-COMPLIANT / PARTIAL]_

- [ ] **CCPA/CPRA** (California Consumer Privacy Act)
  - Data Minimization: _[Compliant: Yes/No/Partial]_
  - Purpose Limitation: _[Compliant: Yes/No/Partial]_
  - Transparency: _[Compliant: Yes/No/Partial]_
  - Compliance Status: _[COMPLIANT / NON-COMPLIANT / PARTIAL]_

- [ ] **SOC 2** (Service Organization Control 2)
  - Security: _[Compliant: Yes/No/Partial]_
  - Availability: _[Compliant: Yes/No/Partial]_
  - Confidentiality: _[Compliant: Yes/No/Partial]_
  - Compliance Status: _[COMPLIANT / NON-COMPLIANT / PARTIAL]_

- [ ] **PCI-DSS** (Payment Card Industry Data Security Standard)
  - Cardholder Data Protection: _[Compliant: Yes/No/Partial]_
  - Compliance Status: _[COMPLIANT / NON-COMPLIANT / PARTIAL]_

- [ ] **ISO 27001** (Information Security Management)
  - ISMS Implementation: _[Compliant: Yes/No/Partial]_
  - Compliance Status: _[COMPLIANT / NON-COMPLIANT / PARTIAL]_

- [ ] **NIST AI RMF** (AI Risk Management Framework)
  - Governance: _[Implemented: Yes/No/Partial]_
  - Map: _[Implemented: Yes/No/Partial]_
  - Measure: _[Implemented: Yes/No/Partial]_
  - Manage: _[Implemented: Yes/No/Partial]_
  - Compliance Status: _[COMPLIANT / NON-COMPLIANT / PARTIAL]_

### 7.2 Compliance Gaps

**Identified Gaps:**
1. Gap: _[Description]_
   - Regulation: _[Which regulation]_
   - Severity: _[Critical/High/Medium/Low]_
   - Remediation Plan: _[What will be done]_
   - Timeline: _[When]_

2. Gap: _[Description]_
   - Regulation: _[Which regulation]_
   - Severity: _[Level]_
   - Remediation Plan: _[Plan]_
   - Timeline: _[Date]_

### 7.3 Compliance Evidence

**Supporting Documentation:**
- [ ] Privacy Impact Assessment (PIA)
- [ ] Data Protection Impact Assessment (DPIA)
- [ ] Security audit report
- [ ] Penetration test results
- [ ] Model documentation
- [ ] Training records
- [ ] Incident response plan
- [ ] Business continuity plan

---

## Section 8: Cloud-Specific Risks

### 8.1 AWS-Specific Risks (if applicable)

**SageMaker:**
- [ ] Endpoints in VPC: _[Yes/No]_
- [ ] Private Link enabled: _[Yes/No]_
- [ ] Model Monitor configured: _[Yes/No]_
- [ ] S3 buckets encrypted: _[Yes/No]_
- [ ] S3 bucket policies secure: _[Yes/No]_
- [ ] IAM least privilege: _[Yes/No]_
- [ ] CloudWatch alarms set: _[Yes/No]_
- [ ] WAF on API Gateway: _[Yes/No]_

**Identified Risks:**
- _[List AWS-specific risks]_

**Mitigations:**
- _[List mitigations in place]_

### 8.2 Azure-Specific Risks (if applicable)

**Azure ML:**
- [ ] Managed endpoints private: _[Yes/No]_
- [ ] VNet integration: _[Yes/No]_
- [ ] RBAC configured: _[Yes/No]_
- [ ] Blob storage private: _[Yes/No]_
- [ ] Azure Defender enabled: _[Yes/No]_
- [ ] Network security groups: _[Yes/No]_
- [ ] Azure Monitor alerts: _[Yes/No]_

**Identified Risks:**
- _[List Azure-specific risks]_

**Mitigations:**
- _[List mitigations in place]_

### 8.3 GCP-Specific Risks (if applicable)

**Vertex AI:**
- [ ] VPC Service Controls: _[Yes/No]_
- [ ] Private Google Access: _[Yes/No]_
- [ ] Cloud Armor enabled: _[Yes/No]_
- [ ] IAM least privilege: _[Yes/No]_
- [ ] Cloud Storage private: _[Yes/No]_
- [ ] Cloud Monitoring alerts: _[Yes/No]_
- [ ] Cloud Audit Logs: _[Yes/No]_

**Identified Risks:**
- _[List GCP-specific risks]_

**Mitigations:**
- _[List mitigations in place]_

---

## Section 9: Recommendations & Action Plan

### 9.1 Critical (Immediate - 0-30 days)

| # | Recommendation | Responsible | Due Date | Status |
|---|----------------|-------------|----------|--------|
| 1 | _[Action]_ | _[Owner]_ | _[Date]_ | ☐ Not Started ☐ In Progress ☐ Complete |
| 2 | _[Action]_ | _[Owner]_ | _[Date]_ | ☐ Not Started ☐ In Progress ☐ Complete |
| 3 | _[Action]_ | _[Owner]_ | _[Date]_ | ☐ Not Started ☐ In Progress ☐ Complete |

### 9.2 High Priority (30-90 days)

| # | Recommendation | Responsible | Due Date | Status |
|---|----------------|-------------|----------|--------|
| 1 | _[Action]_ | _[Owner]_ | _[Date]_ | ☐ Not Started ☐ In Progress ☐ Complete |
| 2 | _[Action]_ | _[Owner]_ | _[Date]_ | ☐ Not Started ☐ In Progress ☐ Complete |

### 9.3 Medium Priority (90-180 days)

| # | Recommendation | Responsible | Due Date | Status |
|---|----------------|-------------|----------|--------|
| 1 | _[Action]_ | _[Owner]_ | _[Date]_ | ☐ Not Started ☐ In Progress ☐ Complete |
| 2 | _[Action]_ | _[Owner]_ | _[Date]_ | ☐ Not Started ☐ In Progress ☐ Complete |

### 9.4 Low Priority (180+ days)

| # | Recommendation | Responsible | Due Date | Status |
|---|----------------|-------------|----------|--------|
| 1 | _[Action]_ | _[Owner]_ | _[Date]_ | ☐ Not Started ☐ In Progress ☐ Complete |

---

## Section 10: Ongoing Monitoring

### 10.1 Key Risk Indicators (KRIs)

| KRI | Current Value | Threshold | Frequency | Owner |
|-----|---------------|-----------|-----------|-------|
| Model accuracy drift | _[%]_ | _[Alert if drops >5%]_ | Daily | _[Name]_ |
| Query volume anomaly | _[QPS]_ | _[Alert if >2x normal]_ | Hourly | _[Name]_ |
| Error rate | _[%]_ | _[Alert if >1%]_ | Real-time | _[Name]_ |
| Unauthorized access attempts | _[Count]_ | _[Alert if >10/day]_ | Daily | _[Name]_ |
| Data distribution drift | _[KL divergence]_ | _[Alert if >0.1]_ | Weekly | _[Name]_ |

### 10.2 Review Schedule

| Review Type | Frequency | Next Review | Responsible |
|-------------|-----------|-------------|-------------|
| Risk Assessment Update | Quarterly | _[Date]_ | _[Name]_ |
| Security Audit | Semi-Annual | _[Date]_ | _[Name]_ |
| Penetration Testing | Annual | _[Date]_ | _[Name]_ |
| Red Team Exercise | Annual | _[Date]_ | _[Name]_ |
| Compliance Review | Annual | _[Date]_ | _[Name]_ |

### 10.3 Incident Response Triggers

**Conditions that trigger incident response:**
1. _[e.g., Model accuracy drops >10% suddenly]_
2. _[e.g., Unusual query patterns detected (>1000 queries/hour from single IP)]_
3. _[e.g., Unauthorized model access attempt]_
4. _[e.g., Data breach or exfiltration detected]_
5. _[e.g., Regulatory non-compliance identified]_

**Escalation Path:**
1. **Tier 1:** _[Role]_ → _[Action]_
2. **Tier 2:** _[Role]_ → _[Action]_
3. **Tier 3:** _[Role]_ → _[Action]_

---

## Section 11: Appendices

### Appendix A: Testing Evidence

**Attach or reference:**
- Adversarial robustness test results
- Penetration test reports
- Model performance benchmarks
- Security scan results

### Appendix B: Architecture Diagrams

**Include:**
- System architecture diagram
- Data flow diagram
- Threat model diagram
- Network topology

### Appendix C: References

**Standards & Frameworks:**
- NIST AI Risk Management Framework
- MITRE ATLAS: https://atlas.mitre.org/
- OWASP Top 10 for LLMs: https://owasp.org/www-project-top-10-for-large-language-model-applications/
- ISO/IEC 23894:2023 (AI Risk Management)

**Research Papers:**
- "Explaining and Harnessing Adversarial Examples" (Goodfellow et al., 2014)
- "Towards Deep Learning Models Resistant to Adversarial Attacks" (Madry et al., 2017)
- "Stealing Machine Learning Models via Prediction APIs" (Tramèr et al., 2016)

### Appendix D: Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | _[Date]_ | _[Name]_ | Initial assessment |
| 1.1 | _[Date]_ | _[Name]_ | _[Changes]_ |
| 2.0 | _[Date]_ | _[Name]_ | _[Major updates]_ |

---

## Section 12: Sign-Off

### Assessment Team

| Name | Role | Signature | Date |
|------|------|-----------|------|
| _[Name]_ | Lead Assessor | _[Signature]_ | _[Date]_ |
| _[Name]_ | Security Engineer | _[Signature]_ | _[Date]_ |
| _[Name]_ | ML Engineer | _[Signature]_ | _[Date]_ |

### Approvals

| Name | Role | Signature | Date |
|------|------|-----------|------|
| _[Name]_ | CISO | _[Signature]_ | _[Date]_ |
| _[Name]_ | VP Engineering | _[Signature]_ | _[Date]_ |
| _[Name]_ | Chief Data Scientist | _[Signature]_ | _[Date]_ |

### Deployment Decision

Based on this risk assessment:

- [ ] **APPROVED FOR DEPLOYMENT** - All critical and high-priority risks mitigated
- [ ] **CONDITIONAL APPROVAL** - Deploy with specific controls: _[List controls]_
- [ ] **DEPLOYMENT BLOCKED** - Too high risk, must remediate before deployment
- [ ] **DECOMMISSION RECOMMENDED** - Risk cannot be adequately mitigated

**Decision Rationale:**
_[Explain the deployment decision]_

---

**END OF RISK ASSESSMENT**

---

## Quick Reference: Risk Levels

| Score Range | Risk Level | Action Required |
|-------------|-----------|-----------------|
| 0.0 - 2.0 | 🟢 LOW | Standard monitoring |
| 2.0 - 3.5 | 🟡 MEDIUM | Enhanced controls recommended |
| 3.5 - 4.5 | 🟠 HIGH | Remediation required before deployment |
| 4.5 - 5.0 | 🔴 CRITICAL | Deployment blocked until remediated |

---

**Document Version:** 2.0  
**Template Maintained By:** AI Security Team  
**Next Template Review:** Week 4

---

### **Create Assessment Example**

Create `docs/baseline_model_assessment.md` (filled-out example):

```markdown
# AI/ML System Risk Assessment - Baseline MNIST CNN

**System Name:** Baseline MNIST Digit Classifier  
**Assessment Date:** 2024-01-29  
**Assessor:** AI Security Team (Week 1-2 Lab)  
**Classification:** Internal Use Only

---

## Executive Summary

### Quick Risk Overview

| Category | Risk Level | Score | Status |
|----------|-----------|-------|--------|
| **Overall Risk** | **HIGH** | **3.85/5.00** | 🟠 |
| Model Security | Critical | 4.75 | 🔴 |
| Data Privacy | Low | 1.00 | 🟢 |
| Deployment Exposure | High | 3.50 | 🟠 |
| Compliance | Low | 1.25 | 🟢 |

### Key Findings
1. **CRITICAL:** No adversarial defenses - 85% attack success rate at ε=0.15
2. **CRITICAL:** Model extraction feasible with 10K queries (94% fidelity)
3. **HIGH:** No query rate limiting enables systematic extraction
4. **MEDIUM:** Full probability distribution returned (should be hard labels only)

### Recommended Actions (Top 3)
1. **Implement adversarial training** (FGSM/PGD with ε=0.1-0.3)
2. **Add API rate limiting** (100 queries/hour per API key)
3. **Return hard labels only** (remove probability distribution from responses)

---

## Section 1: System Overview

### 1.1 Use Case Description

**Business Purpose:**
- Educational demonstration of AI security vulnerabilities
- Baseline model for Week 1-2 threat lab
- Reference implementation for attack demonstrations

**Decisions Made:**
- Classifies handwritten digits (0-9)

**End Users:**
- Security researchers
- AI/ML students
- Internal testing only

### 1.2 Model Information

**Model Type:** ✅ Classification

**Model Architecture:**
```
Architecture: SimpleConvNet (Custom CNN)
Parameters: 101,770
Input: 28×28 grayscale images
Output: 10-class probabilities
Framework: PyTorch 2.0.0
```

**Training Information:**
```
Training Data: MNIST (60,000 images)
Training Time: 5 minutes (5 epochs)
Training Cost: $0 (local GPU)
Last Trained: Week 1, Day 3
Retraining Frequency: On-demand
```

**Performance Metrics:**
```
Accuracy: 98.5%
Precision: 98.4%
Recall: 98.3%
F1 Score: 98.3%
```

### 1.3 Deployment Information

**Cloud Platform:** ✅ AWS
- Services: SageMaker (conceptual), Lambda, API Gateway

**Deployment Type:** ✅ Real-time inference (API endpoint)

**API Endpoint Details:**
```
Endpoint URL: https://api-id.execute-api.us-east-1.amazonaws.com/prod/check
Authentication: API Key (weak)
Rate Limiting: NO ⚠️
Public Access: YES ⚠️
Expected QPS: 10-100
```

---

## Section 2: Data Assessment

### 2.1 Training Data Sensitivity

**Data Classification:** ✅ Level 1 - Public (MNIST dataset)

**Data Types Present:**
- ☐ PII
- ☐ PHI
- ☐ Financial data
- ☐ Biometric data
- ☐ None of the above ✅

**Data Volume:**
```
Training Samples: 60,000
Storage Size: 50 MB
Data Sources: MNIST (Yann LeCun)
```

### 2.2 Data Provenance & Quality

**Data Sources:**
1. Source: MNIST Database
   - Trust Level: Trusted
   - Validation: Public benchmark dataset
   - Ownership: Open source

**Data Validation Measures:**
- ☐ Statistical validation
- ☐ Schema validation
- ☐ Provenance tracking
- ☐ Manual review
- ☐ Automated quality checks
- ✅ None implemented ⚠️ (public dataset, no validation needed)

**Data Poisoning Risk:**
```
Risk Level: LOW (for this demo)
Justification: Using trusted public dataset, no external data ingestion
Mitigations: N/A for demo purposes
Note: Would be HIGH if production system with custom data
```

---

## Section 3: Threat Modeling

### 3.1 Adversarial Attack Surface

| Attack Type | Applicable? | Risk Level | Evidence/Notes |
|-------------|-------------|------------|----------------|
| **Evasion (FGSM/PGD)** | ✅ Yes | **CRITICAL** | Week 1: 85% success at ε=0.15 |
| **Model Extraction** | ✅ Yes | **CRITICAL** | Week 2: 94% fidelity with 10K queries |
| **Data Poisoning** | ✅ Yes | **HIGH** | Week 2: 98.7% backdoor success |
| **Model Inversion** | ✅ Yes | **MEDIUM** | Week 2: Reconstructed class features |
| **Membership Inference** | ✅ Yes | **LOW** | Public dataset, low impact |
| **Backdoor Attack** | ✅ Yes | **HIGH** | Week 2: 3% poisoning effective |
| **API Abuse/DoS** | ✅ Yes | **HIGH** | No rate limiting |

### 3.2 MITRE ATLAS Mapping

| Tactic | Technique ID | Technique Name | Mitigation Status |
|--------|--------------|----------------|-------------------|
| ML Attack Staging | AML.T0043 | Craft Adversarial Data | ☐ None ⚠️ |
| ML Model Access | AML.T0040 | Inference API Access | ☐ None ⚠️ |
| Exfiltration | AML.T0024 | Model Theft via API | ☐ None ⚠️ |
| Impact | AML.T0015 | Evade ML Model | ☐ None ⚠️ |

---

## Section 4: Risk Scoring

### 4.1 Automated Risk Score

**Assessment Input:**
```python
assessment = {
    'model_complexity': 1,        # <1M parameters
    'training_data_sensitivity': 1,  # Public data
    'model_transparency': 3,      # Partially interpretable
    'api_accessibility': 3,       # Authenticated external
    'query_volume': 2,            # Moderate
    'response_detail': 5,         # Full probabilities ⚠️
    'evasion_resistance': 5,      # None ⚠️
    'extraction_resistance': 5,   # None ⚠️
    'poisoning_resistance': 5,    # None ⚠️
    'inversion_resistance': 5,    # None ⚠️
    'authentication': 3,          # API key only
    'monitoring': 4,              # Basic logging
    'input_validation': 4         # Minimal
}
```

**Risk Score Output:**
```
Total Score: 3.85 / 5.00
Risk Level: HIGH 🟠

Category Breakdown:
- Model Characteristics: 1.67 🟢
- Deployment Exposure: 3.33 🟠
- Adversarial Robustness: 5.00 🔴
- Security Controls: 3.67 🟠
```

---

## Section 5: Adversarial Robustness Testing

### 5.1 Evasion Attack Testing

**Test Method:** FGSM

**Test Results:**
```
Clean Accuracy: 98.5%

Adversarial Accuracy:
- ε = 0.05: 95.2% (3.3% attack success)
- ε = 0.10: 82.1% (16.4% attack success)
- ε = 0.15: 68.7% (29.8% attack success) ⚠️
- ε = 0.20: 52.3% (46.2% attack success)
- ε = 0.30: 25.4% (73.1% attack success)

Assessment: FAIL ⚠️ (Should be >80% at ε=0.1)
```

### 5.2 Model Extraction Testing

**Test Results:**
```
Queries Used: 10,000
Surrogate Accuracy: 95.1%
Agreement Rate: 94.2%
Extraction Success: YES ⚠️
Assessment: FAIL (Extraction should be prevented)
```

### 5.3 Data Poisoning Testing

**Test Results:**
```
Poison Rate: 5%
Backdoor Success Rate: 94%
Detection: Undetected ⚠️
Assessment: FAIL
```

### 5.4 Privacy Testing

**Test Results:**
```
Membership Inference Accuracy: 67.3%
Information Leakage: LOW (public dataset)
Assessment: PASS (for public data scenario)
```

---

## Section 9: Recommendations & Action Plan

### 9.1 Critical (0-30 days)

| # | Recommendation | Responsible | Status |
|---|----------------|-------------|--------|
| 1 | Implement adversarial training (FGSM ε=0.1) | ML Team | ☐ Not Started |
| 2 | Add API rate limiting (100/hour) | DevOps | ☐ Not Started |
| 3 | Return hard labels only (remove probabilities) | ML Team | ☐ Not Started |

### 9.2 High Priority (30-90 days)

| # | Recommendation | Responsible | Status |
|---|----------------|-------------|--------|
| 1 | Deploy in VPC with private subnets | Cloud Team | ☐ Not Started |
| 2 | Add input preprocessing (JPEG compression) | ML Team | ☐ Not Started |
| 3 | Implement Model Monitor (CloudWatch) | DevOps | ☐ Not Started |

---

## Section 12: Sign-Off

### Deployment Decision

- ☐ APPROVED FOR DEPLOYMENT
- ☐ CONDITIONAL APPROVAL
- ✅ **DEPLOYMENT BLOCKED** - Too high risk for production
- ☐ DECOMMISSION RECOMMENDED

**Decision Rationale:**
System is suitable for EDUCATIONAL/RESEARCH purposes only. Multiple critical vulnerabilities identified (no adversarial defenses, easy extraction, no rate limiting). Must NOT be deployed to production without implementing all Critical recommendations.

---

**Assessment Complete**  
**Overall Risk:** 🟠 HIGH (3.85/5.00)  
**Deployment Status:** ❌ BLOCKED FOR PRODUCTION
```

---

## ✅ Week 2 Completion Checklist

```bash
# Verify all deliverables
ls -R module1-threat-lab-multicloud/

# Expected structure:
module1-threat-lab-multicloud/
├── notebooks/
│   ├── 01_baseline_model.ipynb ✅
│   ├── 02_evasion_attack.ipynb ✅
│   └── week2/
│       ├── 03_model_extraction.ipynb ✅
│       ├── 04_data_poisoning.ipynb ✅
│       └── 05_model_inversion.ipynb ✅
├── risk-engine/
│   ├── risk_factors.yaml ✅
│   ├── risk_engine.py ✅
│   └── risk_scoring_example.ipynb ✅
├── cloud/
│   ├── aws/attack_surface.md ✅
│   ├── azure/attack_surface.md ✅
│   └── gcp/attack_surface.md ✅
├── docs/
│   ├── threat_model_week1.md ✅
│   ├── fgsm_attack_results.json ✅
│   ├── model_extraction_results.json ✅
│   ├── data_poisoning_results.json ✅
│   ├── model_inversion_results.json ✅
│   ├── ai_risk_assessment_template.md ✅
│   ├── baseline_model_assessment.md ✅
│   └── case_studies/
│       ├── 01_sagemaker_extraction.md ✅
│       └── 02_azure_data_poisoning.md ✅
└── diagrams/
    └── week2/
        ├── risk_score_comparison.png ✅
        ├── extraction_training_loss.png ✅
        ├── poisoning_training_curves.png ✅
        └── model_inversion_comparison.png ✅
```

---

## 🎯 Week 2 Summary

**What You Built:**
- ✅ Model extraction attack (94% fidelity with 10K queries)
- ✅ Data poisoning attacks (label flip + backdoor)
- ✅ Model inversion + membership inference
- ✅ Production-ready AI Risk Scoring Engine
- ✅ Cloud attack surface maps (AWS/Azure/GCP)
- ✅ 2 detailed case studies
- ✅ Comprehensive risk assessment template

**Key Metrics:**
- Lines of code: ~2,500
- Documentation pages: ~150
- Attack demonstrations: 4
- Risk assessments: 4 scenarios
- Total time invested: 10-12 hours

**Portfolio Value:**
- ✅ Demonstrates advanced security knowledge
- ✅ Shows production system thinking
- ✅ Cloud security expertise (AWS/Azure/GCP)
- ✅ Risk assessment capability
- ✅ Technical depth + business context
- ✅ Ready for CISO/Security Architect interviews!

---

## 📦 Deliverables Summary

### **1. Threat Lab Repository** ✅
Complete working demonstrations of:
- Evasion attacks (FGSM)
- Model extraction
- Data poisoning (label flip + backdoor)
- Model inversion + membership inference

### **2. AI Risk Assessment Template** ✅
Production-ready template covering:
- MITRE ATLAS mapping
- OWASP Top 10 alignment
- Compliance frameworks (GDPR, HIPAA, etc.)
- Cloud-specific risks
- Automated risk scoring

### **3. Supporting Documentation** ✅
- Threat models
- Attack surface maps
- Case studies
- Architecture diagrams
- Risk scoring engine

---

🎉 **Congratulations! You've completed Week 2!**

This is now a comprehensive AI security portfolio demonstrating:
- Offensive security skills (red team)
- Defensive strategies (blue team)
- Risk assessment expertise
- Cloud security knowledge
- Production system thinking
