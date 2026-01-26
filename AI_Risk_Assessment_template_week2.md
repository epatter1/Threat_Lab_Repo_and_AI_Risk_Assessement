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

Due to length constraints, I'll continue with the remaining steps in the next message. Would you like me to continue with:

- Step 4: Model Inversion Attack
- Step 5: AI Risk Scoring Engine
- Step 6: Cloud Attack Surface Maps
- Step 7: Case Studies
- Step 8: AI Risk Assessment Template

?
