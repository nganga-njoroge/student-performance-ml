import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from prepare_data import load_and_prepare_data

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Regression output
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    return model

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel("Actual G3")
    plt.ylabel("Predicted G3")
    plt.title("Actual vs Predicted Final Grades")
    plt.plot([0, 20], [0, 20], "--", color="red")
    plt.grid(True)
    plt.show()

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Build model
    model = build_model(X_train.shape[1])

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=100,
        batch_size=16,
        verbose=0
    )

    # Evaluate model
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 Test MSE: {mse:.2f}")
    print(f"📈 Test R²: {r2:.2f}")

    # Plot predictions
    plot_predictions(y_test, y_pred)

    # Optional: Save model
    model.save("../models/student_regression_model.h5")
    print("\n✅ Model saved to models/")

if __name__ == "__main__":
    main()
