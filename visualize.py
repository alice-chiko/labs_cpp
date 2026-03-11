import pandas as pd
import matplotlib.pyplot as plt

def plot_data(filename='points.csv'):
    try:
        data = pd.read_csv(filename)
        
        # Разделяем точки по классам 
        class_0 = data[data['label'] == 0]
        class_1 = data[data['label'] == 1]

        plt.figure(figsize=(10, 6))
        
        plt.scatter(class_0['x'], class_0['y'], color='pink', label='Class 0', alpha=0.6)
        plt.scatter(class_1['x'], class_1['y'], color='purple', label='Class 1', alpha=0.6)

        plt.title('Visualization')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    except FileNotFoundError:
        print("Ошибка: Файл points.csv не найден.")

if __name__ == "__main__":
    plot_data()