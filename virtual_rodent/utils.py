import time
import numpy as np
import torch

# Visualization
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

def save_checkpoint(model, epoch, save_path, optimizer=None):
    d = {'model_state_dict': model.state_dict(), 'epoch': epoch}
    if optimizer is not None:
        d['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(d, save_path)

def load_checkpoint(model, load_path, optimizer=None):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, epoch, optimizer
    return model, epoch

########## Visualization ##########

def display_video(frames, framerate=30, dpi=70):
    """ For IPython do the following on the return `anim`:
        ```
            from IPython.display import HTML
            HTML(anim.to_html5_video())
        ```
    """
    height, width, _ = frames[0].shape
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
      im.set_data(frame)
      return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    return anim