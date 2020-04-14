from functools import partial

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')

from gol.gol import GameOfLife


def black_fill(i, j, ax):
    x = (i - 0.5, i - 0.5, i + 0.5, i + 0.5)
    y = (j - 0.5, j + 0.5, j + 0.5, j - 0.5)
    f = ax.fill(x, y, color='black')
    return f[0]


def white_fill(i, j, ax):
    x = (i - 0.5, i - 0.5, i + 0.5, i + 0.5)
    y = (j - 0.5, j + 0.5, j + 0.5, j - 0.5)
    f = ax.fill(x, y, color='white')
    return f[0]


def color_boxes(state, patches):
    state = state.astype(bool)
    n, m = state.shape
    for i in range(n):
        for j in range(m):
            color = 'black' if state[i, j] else 'white'
            patches[i][j].set_color(color)
    lst= [p for q in patches for p in q]
    return tuple(lst)


def get_next_step(init_state):
    gol = GameOfLife(init_state)
    for _ in range(1000000):
        yield gol.get_state()
        gol.update_state()


def create_animation(init_state, speed=200):
    fig = plt.figure()
    imax, jmax = init_state.shape
    ax = plt.axes(xlim=(-0.5, imax + 0.5), ylim=(-0.5, jmax + 0.5))
    ax.set_axis_off()
    patches = [[white_fill(i, j, ax) for i in range(imax)] for j in range(jmax)]
    frame_func = partial(get_next_step, init_state=init_state)
    anim = FuncAnimation(fig, partial(color_boxes, patches=patches),
                              frames=frame_func, interval=speed, blit=True)
    return anim
