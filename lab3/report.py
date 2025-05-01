from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from perceptron import Perceptron, Perceptron_multi


def areaplot(
    model: Perceptron,
    dataset: pd.DataFrame,
    X: list[str],
    y: str,
):
    o, a = 'bill_length_mm', 'bill_depth_mm'

    b = 0.2
    s = 500
    X_, Y_ = np.meshgrid(
        np.linspace(dataset[o].min()-b,
                    dataset[o].max()+b, s),
        np.linspace(dataset[a].min()-b,
                    dataset[a].max()+b, s)
    )

    aaaaa = pd.DataFrame({
        **dataset.loc[..., X].mean().to_dict(),
        o: X_.ravel(),
        a: Y_.ravel(),
    })

    p = np.array(model.predict(aaaaa))
    Z = np.select([p == c
                   for c in model.classes],
                  range(len(model.classes)))\
        .reshape(X_.shape)

    f, ax = plt.subplots()
    ax.contourf(X_, Y_, Z)
    sns.scatterplot(dataset, x=o, y=a, hue=y, ax=ax)

    ax.set_title("Диаграмма классификации областей")
    return ax


def model_report(
    model: Perceptron,
    dataset: pd.DataFrame,
    X: list[str],
    y: str,
    train: list[int],
    train_args: dict[str, Any]
):
    test = dataset.index.difference(train)
    model_ = model.fitted(
        dataset.loc[train, X],
        dataset.loc[train, y],
        **train_args
    )

    display(
        Markdown("### Кусочек обучающей выборки:"),
        dataset.loc[train, X+[y]].sample(5),
        Markdown("### Параметры обучения:"),
        pd.DataFrame(train_args, index=('',)),
    )
    learning = model_.plot_learning()
    plt.show(learning)

    area = areaplot(
        model=model_,
        dataset=dataset,
        X=X,
        y=y,
    )
    plt.show(area)

    accuracy = model_.accuracy(
        dataset.loc[test, X],
        dataset.loc[test, y]
    )
    display(
        Markdown("### Оценка качетсва:"),
        Markdown(f"Accuracy: {accuracy:.2%}"),
    )

    return model_


def compare_models(
    models: dict[str, Perceptron],
    dataset: pd.DataFrame,
    X: list[str],
    y: str,
    test: list[int],
):
    oleg = pd.DataFrame({spec: model.learning_log
                         for spec, model in models.items()})

    oleg["epoch"] = oleg.index+1
    oleg = pd.melt(oleg, "epoch", var_name="species", value_name="accuracy")

    sns.lineplot(oleg, x='epoch', y='accuracy',
                 hue="species", style="species")
    plt.tight_layout()

    results = dict()
    for spec, model in models.items():
        part = dataset.loc[test]
        if spec != "All":
            part = part[part.species == spec]
        results[spec] = model.accuracy(
            part.loc[..., X],
            part.loc[..., y],
        )
    display(
        Markdown("### Сравнение моделей"),
        pd.DataFrame(results, index=['accuracy'])
    )
