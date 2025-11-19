import tensorflow as tf
import numpy as np
import matplotlib.pyplot as _plt

def build_cnn(input_shape=(32, 32, 3), num_classes=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    print("📦 Loading CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

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

    # -------------------------
    # SAVE MODEL
    # -------------------------
    try:
        model.save("cnn_cifar10.keras")
        print("💾 Saved model: cnn_cifar10.keras")
    except Exception as e:
        print("⚠️ Could not save model:", e)

    # -------------------------
    # SAVE TRAINING PLOT
    # -------------------------
    try:
        _plt.plot(history.history["accuracy"])
        _plt.plot(history.history["val_accuracy"])
        _plt.title("Training Accuracy")
        _plt.xlabel("Epoch")
        _plt.ylabel("Accuracy")
        _plt.legend(["train", "val"])
        _plt.grid(alpha=0.2)
        _plt.tight_layout()
        _plt.savefig("training_history.png", dpi=150)
        _plt.close()
        print("💾 Saved training plot: training_history.png")
    except Exception as e:
        print("⚠️ Could not save training plot:", e)


if __name__ == "__main__":
    main()