# Выявление диабета с помощью нейросети

В рамках данного проекта планирутся решить задачу выявления диабета с помощью
нейронных сетей. Диабет на сегодняшний день - достаточно распространённое
заболевание, поэтому хотелось бы, чтобы люди, дав ответы на несколько вопросов,
могли узнать, что у них, вероятно, диабет и им нужно обратиться к врачу.

## Данные

Был выбран следующий датасет:
https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data

В данном датасете 3 класса: 0 - нет диабета, 1 - преддиабет, 2 - диабет, - и 21
признак. Большинство признаков категориальные: представляют из себя ответ на
вопрос (да или нет). Классы несбалансированы. Датасет состоит из 253,680 ответов
респондентов, что должно быть достаточным для тренировки и валидации модели.

## Модель

В качестве модели планируется использовать полносвязную нейронную сеть с
нескольким внутренними слоями и выходом из 3-х узлов. Для её реализации будет
использована библиотека pytorch. Препроцессинг данных планируется провести с
помощью библиотеки scikit-learn.

## Предсказание

Предполагается, что модель будет предсказывать вероятность для каждого класса.
Модель можно обернуть в сервис, где пользователь сначала отвечает на вопросы и
потом на основании данных ответов выводится вероятность того, что у него
преддиабет или диабет.
