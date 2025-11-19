import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    return x_train, y_train, x_test, y_test

def build_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    print("📦 Loading CIFAR-10...")
    x_train, y_train, x_test, y_test = load_data()

    print("🧠 Building CNN...")
    model = build_cnn()

    print("🚀 Training model...")
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        verbose=2
    )

    print("🔍 Evaluating...")
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
# --- (AUTO-SAVE) Save trained model & history plot ---
try:
    # prefer modern Keras format
    model.save("cnn_cifar10.keras")
    print("💾 Saved model: cnn_cifar10.keras")
except Exception as e:
    print("⚠️ Could not save model:", e)

# Save training history plot (if history exists)
try:
    import matplotlib.pyplot as _plt
    _plt.figure()
    if "accuracy" in history.history:
        _plt.plot(history.history["accuracy"])
    if "val_accuracy" in history.history:
        _plt.plot(history.history["val_accuracy"])
    _plt.title("Training Accuracy")
    _plt.xlabel("Epoch")
    _plt.ylabel("Accuracy")
    _plt.legend(["train","val"])
    _plt.grid(alpha=0.2)
    _plt.tight_layout()
    _plt.savefig("training_history.png", dpi=150)
    _plt.close()
    print("💾 Saved training plot: training_history.png")
except Exception as e:
    print("⚠️ Could not save training plot:", e)
