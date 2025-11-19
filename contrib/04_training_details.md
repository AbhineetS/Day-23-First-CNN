# Training Details — First CNN (CIFAR-10)

### ⚙️ Training Configuration
- **Epochs:** 5  
- **Batch size:** 64  
- **Optimizer:** Adam (default LR)  
- **Loss:** Sparse Categorical Crossentropy  
- **Validation split:** 10%

### 📊 Observed Metrics
- Training accuracy increased from **~45% → ~70%**
- Validation accuracy peaked around **~68%**
- Loss steadily decreased across epochs

### 📝 Notes
- CIFAR-10 is challenging for small CNNs; 65–70% accuracy is expected.
- Increasing depth or using data augmentation can push accuracy higher.
