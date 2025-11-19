# Training Details — First CNN (CIFAR-10)

### ⚙️ Training Configuration
- **Epochs:** 5  
- **Batch size:** 64  
- **Optimizer:** Adam  
- **Loss:** Sparse Categorical Crossentropy  
- **Validation split:** 10%

### 📊 Performance Summary
- Training accuracy: **~45% → ~70%**
- Validation accuracy: **~68% peak**
- Loss decreased consistently across epochs

### 📝 Notes
- CIFAR-10 is difficult for shallow CNNs — 65–70% is normal.
- Depth, augmentation, and regularization can improve accuracy.
