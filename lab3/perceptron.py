from random import randrange
from typing import Self
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

Vector = np.ndarray
Matrix = np.ndarray


class Perceptron:

    def __init__(
        self,
        n_features: int,
        classes: list,
        eta: float = 1e-2,
        epoch: int = 20,
        random_state=0,
    ):
        self.random_state = random_state or randrange(1 << 16)
        gena = np.random.default_rng(self.random_state)
        self.weight = gena.random(n_features)
        self.weight = np.zeros(n_features)
        self.bias = 0
        self.classes = classes
        self.eta = eta
        self.epoch = epoch
        self.learning_log = list()

    def activation(self, x: Vector) -> Vector:
        return x

    def net_input(self, X: Matrix) -> Vector:
        return self.activation(
            np.dot(X, self.weight) + self.bias
        )

    def quantized(self, X: Matrix) -> Vector:
        return np.where(self.net_input(X) >= 0, 1, -1)

    def predict(self, X: pd.DataFrame) -> Vector:
        
        return np.where(
            self.net_input(X.to_numpy()) >= 0,
            self.classes[1],
            self.classes[0],
        )

    def accuracy(self,
                 X: pd.DataFrame,
                 y: pd.Series):
        got = self.predict(X)
        return np.mean(got == y)

    def print_accuracy(self,
                       X: pd.DataFrame,
                       y: pd.Series,
                       text="Accuracy"):
        got = self.predict(X)
        print((f"{text}: "
               f"{np.mean(got == y):.2%}"))

    def plot_learning(self):
        f, ax = plt.subplots()
        sns.lineplot(self.learning_log, ax=ax)
        ax.set_title("Динамика качества обучения")
        return ax

    def learn(self, X: Matrix, y: Vector):
        error = y - self.quantized(X)
        self.weight += self.eta * (error@X)
        self.bias += self.eta * np.sum(error)
        self.learning_log.append(np.mean(error == 0))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        left = np.random.default_rng(self.random_state)
        right = np.random.default_rng(self.random_state)
        for _ in range(self.epoch):
            left.shuffle(X)
            right.shuffle(y)
            self.learn(X, y)
        return self

    @classmethod
    def fitted(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        *args,
        **kwargs
    ) -> Self:
        classes = y.unique()
        assert len(classes) == 2
        new_y = np.select([y == c for c in classes], (-1, 1))
        petya = cls(X.shape[1], y.unique(), *args, **kwargs)
        return petya.fit(X.to_numpy(copy=True), new_y)


class Perceptron_ada(Perceptron):
    def learn(self, X: Matrix, y: Vector):
        error = y - self.net_input(X)
        self.weight += self.eta * (error@X)
        self.bias += self.eta * np.sum(error)
        self.learning_log.append((error**2).sum() / 2)


class Perceptron_multi(Perceptron_ada):
    
    # def net_input(self, X):
    #     X = pd.DataFrame({clss: p.predict(X)
    #                       for clss, p in self.childs.items()})
    #     return super().net_input(X.values)
    
    def predict(self, X):
        X = pd.DataFrame({clss: p.predict(X)
                          for clss, p in self.childs.items()})
        res = self.net_input(X.values)
        return [
            self.classes[min(max(0, round(i)), 
                             len(self.classes))] 
            for i in res
        ]

    @classmethod
    def fitted(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        *args,
        **kwargs
    ) -> Self:
        classes = sorted(y.unique())
        new_y = np.select([y == c for c in classes],
                          range(len(classes)))
        childs = {
            clss: Perceptron_ada.fitted(
                X, (y == clss)*1, *args, **kwargs
            )
            for clss in classes
        }
        
        X = pd.DataFrame({clss: p.predict(X)
                          for clss, p in childs.items()})
        petya = cls(X.shape[1], classes,
                    *args, **kwargs)
        petya.childs = childs
        return petya.fit(X.values, new_y)
