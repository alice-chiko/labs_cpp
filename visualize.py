import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_data(data_file="points.csv", model_file="model_line.csv"):
    try:
        data = pd.read_csv(data_file)

        class_0 = data[data["label"] == 0]
        class_1 = data[data["label"] == 1]

        plt.figure(figsize=(10, 6))

        # Рисуем точки
        plt.scatter(
            class_0["x"], class_0["y"], color="pink", label="Class 0", alpha=0.6
        )
        plt.scatter(
            class_1["x"], class_1["y"], color="purple", label="Class 1", alpha=0.6
        )

        # Пытаемся загрузить параметры обученной прямой и нарисовать её
        try:
            model_data = pd.read_csv(model_file)
            k = model_data["k"].iloc[0]
            b = model_data["b"].iloc[0]

            x_vals = np.array([data["x"].min(), data["x"].max()])
            y_vals = k * x_vals + b
            plt.plot(
                x_vals,
                y_vals,
                color="red",
                linewidth=2,
                linestyle="--",
                label="Neural Net Boundary",
            )
        except FileNotFoundError:
            print(f"Warning: {model_file} not found. Skipping boundary line.")

        plt.title("Neural Network Classification Result")
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.legend()
        plt.grid(True)
        plt.show()

    except FileNotFoundError:
        print(f"Error: File {data_file} not found.")


if __name__ == "__main__":
    plot_data()
