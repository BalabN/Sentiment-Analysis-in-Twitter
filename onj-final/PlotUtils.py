import numpy as np
import matplotlib.pyplot as plt


def show_bar_plot(categories, performance, title):
    plt.rcdefaults()
    y_pos = np.arange(len(categories))
    print(type(performance))
    print(performance)
    error = np.random.rand(len(categories))

    plt.bar(y_pos, performance, xerr=error, align='center', alpha=0.4)
    plt.xticks(y_pos, categories)
    plt.ylabel('Performance')
    plt.title(title)
    plt.show()
