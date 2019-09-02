import numpy as np
import os
import matplotlib.pyplot as plt


wk_space = 'non_feat'
def show_mask():
    mask_path = os.path.join(wk_space, 'mask.npy')
    mask = np.load(mask_path)
    mask = np.transpose(mask, (1, 2, 0))
    print(mask.shape)
    plt.imshow(mask)
    plt.show()

def show_trigger():
    tri_path = os.path.join(wk_space, 'init_L0_trigger.npy')
    trigger = np.load(tri_path)
    trigger = np.transpose(trigger[0], (1, 2, 0))
    print(trigger.shape)
    plt.imshow(trigger)
    plt.show()


def plot_trend():
    log_path = os.path.join(wk_space, 'neuron_value_log.npy')
    log = np.load(log_path)
    print(log[0])
    x_, y_ = [], []
    for i in range(0, len(log), 100):
        x_.append(i)
        y_.append(log[i])
    plt.plot(x_, y_)
    plt.xlabel('Masked Number')
    plt.ylabel('Neuron Activation')
    plt.savefig(os.path.join(wk_space, 'mask_trend.png'))
    plt.show()

if __name__ == "__main__":
    # show_mask()
    # show_trigger()
    plot_trend()