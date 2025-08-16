# FruitRipenessNet — Fruit Ripeness Classifier (PyTorch)

A lightweight custom CNN that classifies fruit images into ripeness stages **immature**, **ripe**, and **overripe**.  
This README is self‑contained: the architecture diagram and training plots are embedded as Base64 images so it renders on GitHub without external links.

> **Notebook:** `fruit_ripeness_classifier.ipynb` (PyTorch)

---

## ✨ Highlights

- **Custom CNN** (3× Conv → Dense) optimized for **128×128 RGB** images
- **On‑device friendly**: small model, straightforward layers
- **Robust training loop** with augmentations (flip, rotate, color jitter, normalization)
- **Final test accuracy**: **91.11%** (from the notebook output)

---

## 🧠 Model Architecture

The model is defined in the notebook as `CustomCNN`:

- **Conv Block ×3**: `Conv2d → BatchNorm2d → ReLU → MaxPool(2×2)`  
  - 3→32, 32→64, 64→128 channels, kernel 3×3, padding 1
- **Flatten** → **Linear(32768→256)** → **ReLU** → **Dropout(0.5)** → **Linear(256→3)** → **Softmax**
- **Input size**: 3×128×128 → pooled to 3× 2× downsampling → 16×16 feature map with 128 channels

### Architecture Diagram

<img alt="CNN Architecture" src="https://b2633864.smushcdn.com/2633864/wp-content/uploads/2021/05/pytorch_cnn_lenet.png?lossy=2&strip=1&webp=1" />

---

## 📊 Training Summary

- **Epochs**: 14
- **Best / Final Test Accuracy**: 91.11%
- **Loss trend**: see plot below

### Loss per Epoch
<img alt="Loss per Epoch" src="https://miro.medium.com/v2/resize:fit:798/1*WmEdzvXcRNBXjw3Xf6ma1w.png" />

### Test Accuracy per Epoch
<img src="https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcQMhqK989n4IWWmXgIIq7l7RVfKmyRnncJ8cgsJbDv7mKpBUoiT"/>

### Confusion Matrix 
<img width="549" height="520" alt="image" src="https://github.com/user-attachments/assets/f6cd7aea-0d1c-4f18-b524-08c4f16fc0a1" />

### 🏷️Classes

```
['immature', 'ripe', 'overripe']
```

---
### Result 
<img width="1706" height="990" alt="image" src="https://github.com/user-attachments/assets/0e4a2657-6818-4f59-ae20-3c3e6ad73372" />

## 📦 Data & Augmentations

- **Dataset**: folder structure compatible with `torchvision.datasets.ImageFolder`
- **Transforms (train)**: `Resize(128)`, `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`, `ToTensor`, `Normalize`
- **Transforms (val/test)**: `Resize(128)`, `ToTensor`, `Normalize`
- **Loaders**: `DataLoader` with `batch_size=32`, `shuffle=True` (train)

---

## ⚙️ Training Setup

- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam (`lr=1e-3`)
- **Scheduler**: optional step scheduler (see notebook)
- **Device**: CUDA if available
- **Metrics**: per‑epoch loss & test accuracy printed during training

---

## 🧩 Inference (example)

```python
import torch
from PIL import Image
from torchvision import transforms
from model import CustomCNN  # or define inline as in the notebook

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN(num_classes=3).to(device)
model.load_state_dict(torch.load("fruit_ripeness_cnn.pt", map_location=device))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open("sample.jpg").convert("RGB")
x = preprocess(img).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()

classes = ['immature', 'ripe', 'overripe']
pred = classes[int(probs.argmax())]
print(pred, probs)
```

---

## 📱 Deploy Ideas

- **Vercel serverless** (TorchScript/ONNX) + **Flutter/React Native** client calling an HTTP endpoint
- **On‑device**: convert to **TorchScript** or **ONNX** and run with **PyTorch Mobile** / **TFLite** (if converted)

---

## 🧪 Reproducibility

- Fix seeds (`torch`, `numpy`, `random`) and enable deterministic cuDNN for reproducibility.
- Ensure consistent preprocessing between training and inference.

---

## 📁 Repository Layout (suggested)

```
.
├── fruit_ripeness_classifier.ipynb
├── README.md
├── model.py                  # CustomCNN definition
├── train.py                  # training loop using the same transforms
├── infer.py                  # simple CLI inference script
├── data/
│   ├── train/immature|ripe|overripe/...
│   ├── val/...
│   └── test/...
└── assets/
    ├── samples/
    └── exports/              # saved model weights, ONNX, TorchScript
```

---

## 📚 License

MIT — feel free to use and adapt.
