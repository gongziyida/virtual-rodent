import numpy as np
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['text.usetex'] = False
plt.rcParams['axes.unicode_minus'] = False

def video(frames, framerate=30, dpi=70):
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

def plot_stats(stats, save_path, plot=sns.lineplot, col='name', col_wrap=3, **plot_args):
    g = sns.FacetGrid(stats, col=col, col_wrap=col_wrap, sharey=False)
    g.map(plot, 'episode', 'val', **plot_args)
    g.set_titles(template='{col_name}')
    g.set_ylabels('')
    for ax in g.axes.flatten():
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    g.tight_layout()
    g.savefig(save_path)
    return g

def plot_smooth_training_curve(t, val, n_subsamples=100, **kwargs):
    t, val = t.to_numpy(), val.to_numpy()
    plt.plot(val, alpha=0.5)
    window = int(len(val)//n_subsamples) if len(val) > n_subsamples * 5 else 10
    smoothed = [val[i:i+window].mean() for i in range(0, len(val), window)]
    plt.plot(list(range(0, len(val), window)), smoothed)
    plt.ylim([min(smoothed) - abs(min(smoothed)) * 0.2, 
              max(smoothed) + abs(max(smoothed)) * 0.2])
