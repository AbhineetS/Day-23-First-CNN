# Model Architecture — First CNN on CIFAR-10

This project uses a simple but effective Convolutional Neural Network designed for beginners to understand the core building blocks of deep learning.

---

## 🧱 Layer-by-Layer Breakdown

### **1️⃣ Input Layer**
- Shape: **(32, 32, 3)**
- Normalized pixel values (0–1 range)

---

### **2️⃣ Convolution Block 1**
- `Conv2D(filters=32, kernel_size=3, activation='relu')`
- `MaxPooling2D(pool_size=2)`
- Purpose:
  - Extract low-level features (edges, colors, corners)
  - Reduce spatial dimensions

---

### **3️⃣ Convolution Block 2**
- `Conv2D(filters=64, kernel_size=3, activation='relu')`
- `MaxPooling2D(pool_size=2)`
- Purpose:
  - Capture more complex patterns (shapes & textures)

---

### **4️⃣ Dense Classifier**
- Flatten layer → converts feature map to vector  
- `Dense(64, activation='relu')`  
- `Dense(10, activation='softmax')` — output probabilities for 10 classes

---

## 🎯 Why This Architecture Works
- Small enough to train fast  
- Strong enough to reach **65–70% accuracy**  
- Builds intuition for:
  - Feature extraction
  - Pooling
  - Fully connected classification

---

## 🧠 Next Steps (Upgrades)
- Add Dropout for better generalization  
- Add BatchNorm to stabilize training  
- Add a 3rd Conv block  
- Try data augmentation  

