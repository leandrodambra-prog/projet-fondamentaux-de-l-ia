import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import time

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Directories
os.makedirs('Livrables', exist_ok=True)
os.makedirs('Explications', exist_ok=True)

class_names = [
    'T-shirt/top', 'Pantalon', 'Pull', 'Robe', 'Manteau',
    'Sandale', 'Chemise', 'Sneaker', 'Sac', 'Bottine'
]

# Style
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e94560',
    'axes.labelcolor': '#eee',
    'text.color': '#eee',
    'xtick.color': '#ccc',
    'ytick.color': '#ccc',
    'grid.color': '#333',
    'font.size': 10,
})

# ============================================================================
# STEP 1 & 2 - Data Loading and Preprocessing
# ============================================================================
print("\n=== STEP 1 & 2: Loading and Preprocessing ===")

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

X_train = train_set.data.numpy()
y_train = train_set.targets.numpy()
X_test = test_set.data.numpy()
y_test = test_set.targets.numpy()

print(f"Train: {X_train.shape[0]} images, Test: {X_test.shape[0]} images")

# Visualisation
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    idx = np.where(y_train == i)[0][0]
    ax.imshow(X_train[idx], cmap='gray')
    ax.set_title(class_names[i], fontsize=10, color='#e94560')
    ax.axis('off')
plt.suptitle('Catalogue Zalando - 1 exemple par categorie', fontsize=13, color='white')
plt.tight_layout()
plt.savefig('Livrables/catalogue_samples.png', dpi=150, bbox_inches='tight')

# Flattening for Random Forest
X_train_flat = X_train.reshape(-1, 784) / 255.0
X_test_flat = X_test.reshape(-1, 784) / 255.0

# ============================================================================
# STEP 3 - Random Forest Baseline
# ============================================================================
print("\n=== STEP 3: Random Forest Baseline ===")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
start_time = time.time()
rf.fit(X_train_flat, y_train)
y_pred_rf = rf.predict(X_test_flat)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {acc_rf*100:.1f}% (Time: {time.time()-start_time:.1f}s)")

# ============================================================================
# STEP 4 - Multi-Layer Perceptron (MLP)
# ============================================================================
print("\n=== STEP 4: MLP Training ===")

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

model_dense = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_dense.parameters(), lr=0.001)

history_dense = {'accuracy': [], 'val_accuracy': []}

def train_model(model, loader, test_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = correct / total
        
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        history_dense['accuracy'].append(train_acc)
        history_dense['val_accuracy'].append(val_acc)
        print(f"Epoch {epoch+1}/{epochs} - Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

train_model(model_dense, train_loader, test_loader, epochs=15)

# ============================================================================
# STEP 5 - Convolutional Neural Network (CNN)
# ============================================================================
print("\n=== STEP 5: CNN Training ===")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model_cnn = CNN().to(device)
optimizer = optim.Adam(model_cnn.parameters(), lr=0.001)
history_cnn = {'accuracy': [], 'val_accuracy': []}

train_model(model_cnn, train_loader, test_loader, epochs=10)

# ============================================================================
# STEP 6 - Learning Curves
# ============================================================================
print("\n=== STEP 6: Learning Curves ===")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history_dense['accuracy'], 'b-', label='Train')
axes[0].plot(history_dense['val_accuracy'], 'r-', label='Validation')
axes[0].set_title('MLP (Reseau Dense)', color='#e94560')
axes[0].set_xlabel('Epoque')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history_cnn['accuracy'], 'b-', label='Train')
axes[1].plot(history_cnn['val_accuracy'], 'r-', label='Validation')
axes[1].set_title('CNN', color='#e94560')
axes[1].set_xlabel('Epoque')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Courbes d'apprentissage", fontsize=13, color='white')
plt.tight_layout()
plt.savefig('Livrables/learning_curves.png', dpi=150, bbox_inches='tight')

# ============================================================================
# STEP 7 - Comparison
# ============================================================================
print("\n=== STEP 7: Comparison ===")

model_cnn.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model_cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc_cnn = accuracy_score(all_labels, all_preds)

print("=" * 55)
print(f"  {'Modele':<20s} {'Accuracy':>10s}")
print("-" * 55)
print(f"  {'Random Forest':<20s} {acc_rf*100:>9.1f}%")
print(f"  {'MLP':<20s} {history_dense['val_accuracy'][-1]*100:>9.1f}%")
print(f"  {'CNN':<20s} {acc_cnn*100:>9.1f}%")
print("-" * 55)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Matrice de confusion - CNN', color='white')
plt.ylabel('Réel')
plt.xlabel('Prédit')
plt.tight_layout()
plt.savefig('Livrables/confusion_matrix_cnn.png', dpi=150)

# ============================================================================
# STEP 8 - Error Analysis
# ============================================================================
print("\n=== STEP 8: Error Analysis ===")
errors = np.where(np.array(all_preds) != np.array(all_labels))[0]
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i, ax in enumerate(axes.flat):
    idx = errors[i]
    ax.imshow(X_test[idx], cmap='gray')
    ax.set_title(f"P: {class_names[all_preds[idx]]}\nR: {class_names[all_labels[idx]]}", color='red', fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.savefig('Livrables/erreurs_cnn.png', dpi=150)

# ============================================================================
# STEP 9 - Visualisation Filters
# ============================================================================
print("\n=== STEP 9: Visualisation Filters ===")
filters = model_cnn.conv1.weight.data.cpu().numpy()
fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(filters[i, 0, :, :], cmap='gray')
    ax.axis('off')
plt.savefig('Livrables/filtres_conv.png', dpi=150)

# Activations
sample_idx = np.where(y_test == 7)[0][0]
sample = torch.tensor(X_test[sample_idx:sample_idx+1]/255.0, dtype=torch.float32).unsqueeze(0).to(device)
with torch.no_grad():
    activation = model_cnn.relu(model_cnn.conv1(sample))
activation = activation.cpu().numpy()

fig, axes = plt.subplots(1, 9, figsize=(16, 2.5))
axes[0].imshow(X_test[sample_idx], cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')
for i in range(8):
    axes[i+1].imshow(activation[0, i, :, :], cmap='viridis')
    axes[i+1].axis('off')
plt.savefig('Livrables/activations_conv.png', dpi=150)

print("\n=== DONE ===")
