# Módulo 5: Deep Learning con PyTorch

## 📚 Objetivos de Aprendizaje

Al completar este módulo, serás capaz de:
- **Construir** redes neuronales con PyTorch (Bloom 6)
- **Entrenar** modelos con backpropagation (Bloom 3)
- **Implementar** early stopping y regularización (Bloom 3)
- **Aplicar** transfer learning (Bloom 3)
- **Optimizar** modelos para prevenir overfitting (Bloom 5)

## ⏱️ Duración Estimada
**12-15 horas** (1.5 semanas)

## 🗺️ Mapa Conceptual

```
                    DEEP LEARNING CON PYTORCH
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   FUNDAMENTOS            ARQUITECTURAS        OPTIMIZACIÓN
        │                     │                     │
  ┌─────┴─────┐        ┌──────┴──────┐      ┌──────┴──────┐
  │           │        │             │      │             │
Tensors    Autograd  Feedforward   CNN    Regular   Transfer
  │           │        │             │      │        Learning
  • ops       • grad   • MLP         • Conv2D  • Dropout   │
  • GPU       • back   • Activ       • Pool    • L1/L2     • Fine-tune
  • reshape   • prop   • Loss        • Batch   • Early     • Feature
                                      Norm     Stop       Extract
```

## 📖 Contenidos Detallados

### 1. Fundamentos de PyTorch (2 horas)

#### Tensores: Bloques Básicos

```python
import torch
import numpy as np

# Crear tensores
tensor_zeros = torch.zeros(3, 4)
tensor_ones = torch.ones(2, 3)
tensor_random = torch.randn(2, 3)  # Normal(0, 1)

# Desde numpy
numpy_array = np.array([[1, 2], [3, 4]])
tensor = torch.from_numpy(numpy_array)

# Operaciones
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print("Sum:", a + b)
print("Product:", a * b)
print("Dot product:", torch.dot(a, b))
print("Matrix mul:", torch.matmul(a.view(3,1), b.view(1,3)))

# GPU (si disponible)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor_gpu = tensor.to(device)
print(f"Device: {device}")
```

#### Autograd: Diferenciación Automática

```python
# Gradientes automáticos
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1

# Backpropagation
y.backward()
print(f"dy/dx at x=2: {x.grad}")  # 2*x + 3 = 7

# Ejemplo más complejo
x = torch.randn(3, requires_grad=True)
y = x * 2
z = y.mean()

z.backward()
print(f"Gradient: {x.grad}")
```

### 2. Redes Neuronales Feed-Forward (3 horas)

#### Arquitectura de una Red Neuronal

```
Input Layer    Hidden Layer 1   Hidden Layer 2   Output Layer
   (784)           (128)            (64)             (10)
     │               │                │                │
     ○───────────────○────────────────○────────────────○
     ○───────────────○────────────────○────────────────○
     ○───────────────○────────────────○────────────────○
     │               │                │                │
  [ReLU]          [ReLU]           [ReLU]          [Softmax]
```

**Implementación en PyTorch:**
```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(SimpleNN, self).__init__()
        
        # Layers
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        
        # Dropout para regularización
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Forward pass
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)  # No activation - raw logits
        return x

# Crear modelo
model = SimpleNN(input_size=784, hidden_size1=128, hidden_size2=64, num_classes=10)
print(model)

# Contar parámetros
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

#### Training Loop Completo

```python
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Preparar datos (ejemplo con datos sintéticos)
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Modelo, loss, optimizer
model = SimpleNN(784, 128, 64, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Move to device
        data = data.to(device)
        targets = targets.to(device)
        
        # Forward
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward
        optimizer.zero_grad()  # Reset gradients
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')
    
    # Epoch statistics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
```

#### Evaluación

```python
def evaluate(model, test_loader, device):
    model.eval()  # Modo evaluación (desactiva dropout)
    correct = 0
    total = 0
    
    with torch.no_grad():  # No calcular gradientes
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# Usar
test_acc = evaluate(model, test_loader, device)
print(f'Test Accuracy: {test_acc:.2f}%')
```

### 3. Regularización y Early Stopping (2 horas)

#### Técnicas de Regularización

**Dropout:**
```python
class RegularizedNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout1 = nn.Dropout(0.3)  # 30% dropout
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.2)  # 20% dropout
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
```

**L2 Regularization (Weight Decay):**
```python
# En el optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5  # L2 regularization
)
```

**Batch Normalization:**
```python
class BNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)  # Batch Norm después de layer
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
```

#### Early Stopping Implementation

```python
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Uso en training loop
early_stopping = EarlyStopping(patience=5)

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader)
    val_loss = validate(model, val_loader)
    
    print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')
    
    # Early stopping check
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break
```

### 4. Learning Rate Scheduling (1.5 horas)

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

# Opción 1: Reduce LR cuando val_loss no mejora
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,      # Reducir a la mitad
    patience=3,
    verbose=True
)

# Training loop
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader)
    val_loss = validate(model, val_loader)
    
    # Update learning rate
    scheduler.step(val_loss)

# Opción 2: StepLR - reduce cada N epochs
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Opción 3: Cosine Annealing
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
```

### 5. Transfer Learning (3 horas)

#### Concepto de Transfer Learning

```
Pre-trained Model (ImageNet)    →    Fine-tune for New Task
         │                                    │
    [Conv Layers]                        [Conv Layers]
         │                                    │ (frozen)
    [Feature Maps]                       [Feature Maps]
         │                                    │
    [Classifier]                      [New Classifier]
      (1000 classes)                     (10 classes)
```

**Ejemplo con modelo pre-entrenado:**
```python
import torchvision.models as models

# Cargar modelo pre-entrenado
resnet = models.resnet18(pretrained=True)

# Ver arquitectura
print(resnet)

# Opción 1: Feature Extraction (congelar todas las capas)
for param in resnet.parameters():
    param.requires_grad = False

# Reemplazar última capa
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 10)  # 10 clases nuevas

# Solo entrenar última capa
optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)

# Opción 2: Fine-tuning (entrenar todo pero con LR diferentes)
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, 10)

# Different LR for pre-trained vs new layers
optimizer = optim.Adam([
    {'params': resnet.layer4.parameters(), 'lr': 1e-4},  # Pre-trained
    {'params': resnet.fc.parameters(), 'lr': 1e-3}        # New layer
])
```

### 6. Guardar y Cargar Modelos

```python
# Guardar modelo completo
torch.save(model.state_dict(), 'model_checkpoint.pth')

# Cargar modelo
model = SimpleNN(784, 128, 64, 10)
model.load_state_dict(torch.load('model_checkpoint.pth'))
model.eval()

# Guardar con optimizer y epoch (para reanudar training)
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Cargar checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

## 🎯 Prácticas Guiadas

1. **MLP para Clasificación Tabular** (2.5h) - Iris/Wine dataset
2. **Red con Regularización** (2h) - MNIST
3. **Transfer Learning** (3h) - Fine-tune ResNet
4. **Early Stopping + LR Scheduling** (2h) - Fashion MNIST
5. **Comparación sklearn vs PyTorch** (2h) - Mismo problema

## ✏️ Ejercicios

1. Implementar red neuronal para regresión
2. Experimentos con diferentes arquitecturas
3. Comparar técnicas de regularización
4. Transfer learning para clasificación de imágenes

## 📚 Recursos Externos

### 📹 Videos
- [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) (4 videos, ~1h)
- [PyTorch Tutorial - Basics](https://www.youtube.com/watch?v=c36lUUr864M) (1h)
- [Stanford CS231n](https://www.youtube.com/playlist?list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk)

### 📖 Lecturas
- [Deep Learning Book](https://www.deeplearningbook.org/) - Goodfellow et al.
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Transfer Learning Guide](https://arxiv.org/abs/1411.1792)

### 🎮 Herramientas
- [PyTorch Playground](https://playground.tensorflow.org/) - Visualización
- [Netron](https://netron.app/) - Visualizar arquitecturas

## ✅ Checklist

- [ ] Implementar red neuronal desde cero
- [ ] Aplicar regularización (dropout, L2, BN)
- [ ] Early stopping funcional
- [ ] Transfer learning implementado
- [ ] Modelos guardados y cargados correctamente

## ➡️ Siguientes Pasos

**Módulo 6:** LLMs, Prompting y RAG
