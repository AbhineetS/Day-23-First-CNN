# Model Architecture
- Conv2D -> ReLU -> MaxPool (x2)
- Flatten -> Dense(128) -> Dropout -> Dense(10, softmax)
- Optimizer: Adam, Loss: sparse_categorical_crossentropy
