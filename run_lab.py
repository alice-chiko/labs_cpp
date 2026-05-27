import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification # для генерации 3 датасета для проверки
import os
import sys

msys_path = r"C:\msys64\ucrt64\bin"
if os.path.exists(msys_path):
    os.add_dll_directory(msys_path)

sys.path.append(os.getcwd())

# импортируем нашу библиотеку
import my_nn


# функция автоматической генерации третьего датасета
def generate_synthetic_d3(file_path="dataset3.csv"):
    if not os.path.exists(file_path):
        X, y = make_classification(
            n_samples=150,
            n_features=2,
            n_redundant=0,
            n_clusters_per_class=1,
            class_sep=2.0,
            random_state=24,
        )
        df = pd.DataFrame(X, columns=["feature_0", "feature_1"])
        df["target"] = y
        df.to_csv(file_path, index=False)
        print(f"cоздан искусственный датасет: {file_path}")


# отрисовка разделяющей прямой для 2D пространств
def plot_perceptron_boundary(X, y, model, title_name):
    plt.figure(figsize=(8, 6))

    # рисуем точки классов 0 и 1
    plt.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        color="red",
        edgecolors="k",
        label="Класс 0",
        alpha=0.7,
    )
    plt.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        color="blue",
        edgecolors="k",
        label="Класс 1",
        alpha=0.7,
    )

    # Достаем обученные веса и сдвиг из c++ модуля
    w = model.get_weights()
    b = model.get_bias()

    # уравнение прямой
    x0_min, x0_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x0_points = np.linspace(x0_min, x0_max, 200)

    if len(w) == 2 and w[1] != 0:
        x1_points = -(w[0] * x0_points + b) / w[1]
        plt.plot(
            x0_points,
            x1_points,
            color="black",
            linestyle="-",
            linewidth=2.5,
            label="разделяющая прямая",
        )

    plt.title(title_name, fontsize=14)
    plt.xlabel("Признак 0")
    plt.ylabel("Признак 1")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show(block=False)


def prepare_and_run(file_path, input_dim):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :input_dim].values
    y = df.iloc[:, -1].values.astype(int)

    # нормирование данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # разделение датасета
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # конвертируем данные для нашей cpp обертки
    train_dataset_cpp = [
        (list(X_train[i]), int(y_train[i])) for i in range(len(X_train))
    ]

    model = my_nn.Perceptron(input_dim, 0.01)
    model.train(train_dataset_cpp, 1000)

    preds = []
    for x_sample in X_test:
        preds.append(model.predict(list(x_sample)))

    f1 = f1_score(y_test, preds)

    # если размерность 2, сразу строим график
    if input_dim == 2:
        plot_perceptron_boundary(
            X_train,
            y_train,
            model,
            f"f1-score на 1 датасете: {f1:.4f}",
        )

    return model, f1, scaler


generate_synthetic_d3()


# запуск для первого датасета (2 признака)
model_d1, f1_d1, scaler_d1 = prepare_and_run("dataset1.csv", input_dim=2)
print(f"f1-score для d1: {f1_d1:.4f}")

# запуск для второго датасета (4 признака)
model_d2, f1_d2, scaler_d2 = prepare_and_run("dataset2.csv", input_dim=4)
print(f"f1-score для d2: {f1_d2:.4f}")

# расчет итоговой оценки по формуле
final_score = 0.5 * f1_d1 + 0.5 * f1_d2
print(f"\nИтоговая оценка качества: {final_score:.4f}")

if final_score >= 0.55:
    print(" Лабораторная работа проходит порог")
else:
    print(" Монетка полезнее модели")


# дообучение на 3 датасете
def fine_tune_on_defense(d3_path, models_dict, scalers_dict):

    if not os.path.exists(d3_path):
        print(f"Файл {d3_path} не найден")
        return

    df_d3 = pd.read_csv(d3_path)

    #  определяем количество признаков
    detected_dim = df_d3.shape[1] - 1
    X_d3 = df_d3.iloc[:, :detected_dim].values
    y_d3 = df_d3.iloc[:, -1].values.astype(int)

    if detected_dim in scalers_dict and detected_dim in models_dict:
        scaler = scalers_dict[detected_dim]
        model = models_dict[detected_dim]
    else:
        scaler = StandardScaler()
        X_d3_train_scaled = scaler.fit_transform(X_d3)
        model = my_nn.Perceptron(detected_dim, 0.01)
        train_dataset_cpp = [
            (list(X_d3_train_scaled[i]), int(y_d3[i]))
            for i in range(len(X_d3_train_scaled))
        ]
        model.train(train_dataset_cpp, 500)

    X_d3_scaled = scaler.transform(X_d3)
    dataset_d3_cpp = [
        (list(X_d3_scaled[i]), int(y_d3[i])) for i in range(len(X_d3_scaled))
    ]

    # визуализация до дообучения только для двухмерных
    if detected_dim == 2:
        plot_perceptron_boundary(X_d3_scaled, y_d3, model, f"Датасет 3 ДО дообучения")
    else:
        print(
            f"График пропущен: Пространство {detected_dim}D невозможно отобразить на плоскости"
        )

    model.train(dataset_d3_cpp, 200)

    # считаем финальную точность
    preds = [model.predict(list(x)) for x in X_d3_scaled]
    new_f1 = f1_score(y_d3, preds)
    print(f"Финальная точность f1 на новом датасете: {new_f1:.4f}")

    # визуализация после дообучения только для двухмерных
    if detected_dim == 2:
        plot_perceptron_boundary(
            X_d3_scaled, y_d3, model, f"f1-score на 3 датасете: {new_f1:.4f}"
        )


trained_models = {2: model_d1, 4: model_d2}
trained_scalers = {2: scaler_d1, 4: scaler_d2}

fine_tune_on_defense("dataset3.csv", trained_models, trained_scalers)
plt.show()
