# Метод обратного распространения ошибки для многослойной полностью связанной нейронной сети
## Краткое описание реализации
На языке Python была реализована многослойная нейронная сеть для решения задач регрессии и классификации.  

Интерфейс модуля с реализацией был сделан на подобии итерфейса модуля [neural_nerwork](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neural_network "neural_network module") библиотеки scikit-learn. 

Имеются возможности для использования различных функций активации (relu, logistic, пользователская функция активации), для обучения пачками (когда корректировка весов просиходит не на основании одного элемента выборки, а основании нескольких элементов), можно задавать критерии остановки (по числу итераций или по достигнутой точности) для "зашитого" в реализацию метода оптимизации (градиентный спуск) и конфигурацию сети (число скрытых слоёв и число нейронов в каждом слое).  

Все параметры обучения указываются при создании объекта класса myMLPClassifier. Обучение сети происходит при вызове у этого объекта метода fix(X, y). Метод принимает матрицу X с входными признаками выборки (каждая строка соответсвует признакам одного эоемента выборки) и y - столбец соответствующих классов элементов выборки (в случае задачи классификации) или матрицу выходных признаков (в случае задачи восстановления регрессии). Метод ничего не возвращает, но сохраняет веса обученной на поданной выборке сети внутри объекта класса myMLPClassifier.  

Для предсказания выхода для тестовой выборки используется метод predict(X). Он принимает на вход X (в таком же формате, что и метод fit) и возвращает предсказанный выход y (в таких же форматах, что и y из fit)

## Тестовый пример
В качестве примера использования написанного модуля решалась задача распознавания рукописных цифр из базы [MNIST](http://yann.lecun.com/exdb/mnist/ "MNIST dataset")

Скрипт mnist_download.py отвечает за загрузку и предобработку этого датасета. После вызова соотвествующих функций (см. начало скрипта MNIST_train.py) мы полчуаем изображения символов в удобном для последующего использования виде - они хранятся в виде двумерных массивов с интенсивностями пикселей изображения (от 0 до 255). Изображения разбиты на 2 массива - тренировочная (60000 изображений) и тестовая (10000 изображений) выборки. Также в отдельных массивах лежит информация о том, какая цифра изображена на соответствующих изображениях.  Далее в MNIST_train.py двумерные массивы разворачиваются в одномерные вектора и складываются в массивы X_train и X_test. Массивы-метки классов отстаются нетронутыми и записываются в y_train и y_test. Далее происходит обучение сети на тренировочной выборке (X_train и y_train) и меряется ошибка предсказания (как отношение числа не угаданных цифр к общему объёму выборки) для тренировочной и тестовой выборок.

Наилучший результат (был получен на приложенной версии скрипта) - ошибка на тестовой выборке была равна 0.14

## Инструкция по сборке и запуску

1. Установить интерпертатор python версии 2.7.x или 3.6.x. Рекомендуется поставить пакет [Anaconda](https://www.anaconda.com/download/ "Download Anaconda"), в который входит и интерпретатор, и некорые полезные инструменты (в частности, pip, который будет использован ниже по инструкции).

2. Установить необходимый для работы скриптов пакет numpy. Для этого надо выполнить в командной строке команду:
<pre><code>pip install numpy
</code></pre>
В случае установленного под Linux или MacOS Python версии 3+:
<pre><code>pip3 install numpy
</code></pre>

3. Запустить скрипт MNIST_train.py (скрипты download_mnist.py и NeuralNetwork.py должны находиться с ним в одной дириктории!) с помощью команды:  
Python 2 под Linux или MacOS, или Windows:
<pre><code>python MNIST_train.py
</code></pre>
Python 3 под Linux или MacOS:
<pre><code>python3 MNIST_train.py
</code></pre>

## Не сделано на данный момент

Нет отчёта  

В данный момент, задача классификации решается как задача регрессии (то есть не используется активационная функция softmax на последнем  слое и меряется евклидова функции ошибки вместо кросс-энтропии). При попытке использовать softmax и кросс-энтропию для решения задачи с распознаванием цифр метод очень часто расходится, а когда сходится, то даёт худший результат (ошибка около 0.28)
