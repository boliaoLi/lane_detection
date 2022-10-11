from matplotlib.pyplot import plot as plt
import numpy as np


if __name__ == '__main__':
    x1 = np.linspace(0, 540, 100)
    x2 = np.linspace(0, 540, 100)
    for i in x1:
        for j in x2:
            iou = (x1 + x2) / np.e**