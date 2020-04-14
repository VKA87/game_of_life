from functools import partial

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')


def build_grid(ax):
    ax.axis('off')
    xm, xp = ax.get_xlim()
    ym, yp = ax.get_ylim()
    for x in np.arange(xm, xp + 1):
        ax.plot([x, x], [ym, yp], color='black', linewidth=0.1)
    for y in np.arange(ym , yp + 1):
        ax.plot([xm, xp], [y, y], color='black', linewidth=0.1)
    return ax


def black_fill(i, j):
    x = (i - 0.5, i - 0.5, i + 0.5, i + 0.5)
    y = (j - 0.5, j + 0.5, j + 0.5, j - 0.5)
    f = ax.fill(x, y, color='black')
    return f[0]


def white_fill(i, j):
    x = (i - 0.5, i - 0.5, i + 0.5, i + 0.5)
    y = (j - 0.5, j + 0.5, j + 0.5, j - 0.5)
    f = ax.fill(x, y, color='white')
    return f[0]

def move_box(data, patches, xmax, ymax):
    ip, jp = data
    ip = ip % xmax
    jp = jp % ymax
    for i in range(xmax):
        for j in range (ymax):
            color = 'black' if i==ip and j==jp else 'white'
            patches[i][j].set_color(color)

    lst= [p for q in patches for p in q]
    return tuple(lst)

def next_point():
    i, j = 15, 15
    for _ in range(10000):
        h = np.random.choice([True, False])
        if h:
            j += np.random.choice([+1, -1])
        else:
            i += np.random.choice([+1, -1])
        yield i, j



if __name__ == '__main__':

    fig = plt.figure()
    imax, jmax = 30, 30
    ax = plt.axes(xlim=(-0.5, imax + 0.5), ylim=(-0.5, jmax + 0.5))

    patches = [[white_fill(i, j) for i in range(imax)] for j in range(jmax)]

    anim = FuncAnimation(fig, partial(move_box, patches=patches, xmax=imax, ymax=jmax),
                              frames=next_point(), interval=200, blit=True)

    anim.save('sine_wave.gif', writer='imagemagick')
