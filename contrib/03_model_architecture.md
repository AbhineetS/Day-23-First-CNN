# Model Architecture — First CNN (CIFAR-10)

### 📐 Input
- Shape: **32 × 32 × 3** (RGB images)
- Normalized to [0,1]

### 🧱 Convolutional Base
1. **Conv2D (32 filters, 3×3) + ReLU**
2. **MaxPooling2D (2×2)**
3. **Conv2D (64 filters, 3×3) + ReLU**
4. **MaxPooling2D (2×2)**

### 🔗 Fully Connected Head
- Flatten layer
- Dense layer: **64 units + ReLU**
- Output: **10-class Softmax**

### 🎯 Why This Works
This simple architecture is enough to:
- Extract spatial patterns,
- Recognize edges, textures, shapes,
- Classify CIFAR-10 images with stable accuracy.

A perfect Day-23 introductory CNN.
