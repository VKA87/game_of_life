import numpy as np

from gol.animation import create_animation


if __name__ =='__main__':
    init_state = np.zeros((100, 100)).astype(int)
    init_state[51, 49:51] = 1
    init_state[50, 48:50] = 1
    init_state[49, 49] = 1
    anim = create_animation(init_state, interval=200, n_frames=1000)
    anim.save('gol.mp4')
