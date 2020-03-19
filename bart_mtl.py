import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('classic')

def general():
    x = np.linspace(0, 10, 100)
    fig = plt.figure()
    plt.plot(x, np.sin(x), '-')
    plt.plot(x, np.cos(x), '--');
    return x

def matlab_style():
    x = general()
    # MATLAB-style interface
    plt.figure() # create a plot figure
    # create the first of two panels and set current axis
    plt.subplot(2, 1, 1) # (rows, columns, panel number)
    plt.plot(x, np.sin(x))
    # create the second panel and set current axis
    plt.subplot(2, 1, 2)
    plt.plot(x, np.cos(x));

def object_oriented():
    x = general()
    # Object-oriented interface
    # First create a grid of plots
    # ax will be an array of two Axes objects
    fig, ax = plt.subplots(2)
    # Call plot() method on the appropriate object
    ax[0].plot(x, np.sin(x))
    ax[1].plot(x, np.cos(x))
    plt.show()

def simple_line_plots():
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    ax = plt.axes()
    x = np.linspace(0, 10, 1000)
    # Could be also plt. below
    ax.plot(x, np.sin(x), color='red', label='sin(X)')
    ax.plot(x, x - 1, linestyle='dashed', label='just a line')
    ax.plot(x, x + 1, linestyle=':', label='other line')
    ax.plot(x, np.sin(x - 1), color='g', label='other sin(x)')
    ax.plot(x, np.cos(x), label='cos(x)');
    # Adjusting the Plot: Axes Limits
    # plt.xlim(-1, 11)
    # plt.ylim(-1.5, 7);
    # or with plt.axis() by passing a list that specifies [xmin, xmax, ymin,
    # ymax]
    plt.axis([-1, 11, -1.5, 7])
    plt.axis('tight')
    # Labeling Plots
    plt.title("A Sine Curve")
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.legend()
    plt.show()

def simple_scatter_plots():
    plt.style.use('seaborn-whitegrid')
    x = np.linspace(0, 10, 30)
    y = np.sin(x)
    plt.plot(x, y, '-<', color='gray',
             markersize=15, linewidth=4,
             markerfacecolor='white',
             markeredgecolor='gray',
             markeredgewidth=2)
    plt.ylim(-1.2, 1.2);
    plt.show()

simple_scatter_plots()

def markers():
    rng = np.random.RandomState(0)
    for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
        plt.plot(rng.rand(5), rng.rand(5), marker,
                 label="marker='{0}'".format(marker))
    plt.legend(numpoints=1)
    plt.xlim(0, 1.8)
    plt.show()


