# Laboratory Work #1


## Требования к системе
* **C++ Компилятор**: необходима поддержка **C++20**
* **Python**: Версия 3.8 или выше.
* **Библиотеки Python**: `pandas`, `matplotlib`.
* **Структура**: Папка `include/` с заголовками должна находиться в корне проекта.

## Инструкция по работе с проектом
Проект разделен на три этапа: тесты, генерация данных и визуализация (Python).

### Для запуска 1 лабы вставьте в терминал
```powershell

# Компиляция тестового модуля
g++ -std=c++20 ./tests/unit_tests.cpp -I . -o run_tests.exe

# Запуск тестов
./run_tests.exe

# Компиляция основной программы
g++ -std=c++20 main.cpp -o lab1.exe

# Запуск (создаст файл points.csv)
./lab1.exe

# Установка зависимостей (если не установлены)
pip install pandas matplotlib

# Запуск скрипта визуализации
python visualize.py

### Для запуска 2 лабы вставьте в терминал
```powershell

# Компиляция тестового модуля и запуск тестов
g++ -std=c++20 ./tests/unit_tests.cpp -I . -o run_tests.exe
./run_tests.exe

# Компиляция основной программы и запуск
g++ -std=c++20 main.cpp -I . -o lab2.exe
./lab2.exe

# Запуск скрипта визуализации
python visualize.py

### Для запуска 3 лабы вставьте в терминал

```powershell

# Установка необходимых модулей Python
pip install pybind11 scikit-learn pandas numpy matplotlib

# Компиляция C++ перцептрона в модуль Python 
g++ -O3 -Wall -shared -std=c++20 -fPIC -IC:\Users\user\AppData\Roaming\Python\Python313\site-packages\pybind11\include "-IC:\Program Files\Python313\Include" binding.cpp -o my_nn.pyd "-LC:\Program Files\Python313\libs" -lpython313 -static-libgcc -static-libstdc++

# Запуск лабораторной работы
python run_lab.py