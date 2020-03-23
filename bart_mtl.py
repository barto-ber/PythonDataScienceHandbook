import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.options.display.width = 0
pd.options.display.max_rows = None
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


def markers():
    rng = np.random.RandomState(0)
    for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
        plt.plot(rng.rand(5), rng.rand(5), marker,
                 label="marker='{0}'".format(marker))
    plt.legend(numpoints=1)
    plt.xlim(0, 1.8)
    plt.show()

# Scatter Plots with plt.scatter
def scatter_with_plt():
    rng = np.random.RandomState(0)
    x = rng.randn(100)
    y = rng.randn(100)
    colors = rng.rand(100)
    sizes = 1000 * rng.rand(100)
    plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,
                cmap='viridis')
    plt.colorbar();  # show color scale
    plt.show()


def iris_flowers():
    from sklearn.datasets import load_iris
    iris = load_iris()
    features = iris.data.T
    plt.scatter(features[0], features[1], alpha=0.2,
                s=100 * features[3], c=iris.target, cmap='viridis')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1]);
    plt.show()


def basic_errorbars():
    plt.style.use('seaborn-whitegrid')
    x = np.linspace(0, 10, 50)
    dy = 0.8
    y = np.sin(x) + dy * np.random.randn(50)
    # plt.errorbar(x, y, yerr=dy, fmt='.k');
    plt.errorbar(x, y, yerr=dy, fmt='o', color='black',
                 ecolor='lightgray', elinewidth=3, capsize=0);
    plt.show()


def continous_errors():
    from sklearn.gaussian_process import GaussianProcess
    # define the model and draw some data
    model = lambda x: x * np.sin(x)
    xdata = np.array([1, 3, 5, 6, 8])
    ydata = model(xdata)
    # Compute the Gaussian process fit
    gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1E-1,
                         random_start=100)
    gp.fit(xdata[:, np.newaxis], ydata)
    xfit = np.linspace(0, 10, 1000)
    yfit, MSE = gp.predict(xfit[:, np.newaxis], eval_MSE=True)
    dyfit = 2 * np.sqrt(MSE)  # 2*sigma ~ 95% confidence region
    # Visualize the result
    plt.plot(xdata, ydata, 'or')
    plt.plot(xfit, yfit, '-', color='gray')
    plt.fill_between(xfit, yfit - dyfit, yfit + dyfit,
                     color='gray', alpha=0.2)
    plt.xlim(0, 10);
    plt.show()

# Visualizing a Three-Dimensional Function
def three_d_functions():
    plt.style.use('seaborn-white')
    def f(x, y):
        return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

    x = np.linspace(0, 5, 50)
    y = np.linspace(0, 5, 40)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    # plt.contour(X, Y, Z, 20, cmap='RdGy')
    # plt.contourf(X, Y, Z, 20, cmap='RdGy')
    # plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
    #            cmap='RdGy')
    # plt.axis(aspect='image');
    contours = plt.contour(X, Y, Z, 3, colors='black')
    plt.clabel(contours, inline=True, fontsize=8)
    plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
               cmap='RdGy', alpha=0.5)
    plt.colorbar();
    plt.show()


def the_hist():
    plt.style.use('seaborn-white')
    # data = np.random.randn(1000)
    # plt.hist(data)
    # plt.hist(data, bins=30, normed=True, alpha=0.5,
    #          histtype='stepfilled', color='steelblue',
    #          edgecolor='none');
    x1 = np.random.normal(0, 0.8, 1000)
    x2 = np.random.normal(-2, 1, 1000)
    x3 = np.random.normal(3, 2, 1000)
    kwargs = dict(histtype='stepfilled', alpha=0.3, normed=True, bins=40)
    plt.hist(x1, **kwargs)
    plt.hist(x2, **kwargs)
    plt.hist(x3, **kwargs);
    plt.show()

def two_d_hist():
    plt.style.use('seaborn-white')
    mean = [0, 0]
    cov = [[1, 1], [1, 2]]
    x, y = np.random.multivariate_normal(mean, cov, 10000).T
    # plt.hist2d(x, y, bins=30, cmap='Blues')
    plt.hexbin(x, y, gridsize=30, cmap='Blues')
    cb = plt.colorbar(label='count in bin')
    cb.set_label('counts in bin')
    plt.show()

def plot_legend():
    x = np.linspace(0, 10, 1000)
    fig, ax = plt.subplots()
    ax.plot(x, np.sin(x), '-b', label='Sine')
    ax.plot(x, np.cos(x), '--r', label='Cosine')
    ax.axis('equal')
    ax.legend(loc='upper left', frameon=False)
    plt.show()



def california():
    cities = pd.read_csv('notebooks/data/california_cities.csv')
    print(cities.head())
    # Extract the data we're interested in
    lat, lon = cities['latd'], cities['longd']
    population, area = cities['population_total'], cities['area_total_km2']
    # Scatter the points, using size and color but no label
    plt.scatter(lon, lat, label=None,
                c=population, cmap='viridis',
                s=area, linewidth=0, alpha=0.5)
    plt.axis(aspect='equal')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.colorbar(label='log$_{10}$(population)')
    # plt.clim(3, 7)
    # Here we create a legend:
    # we'll plot empty lists with the desired size and label
    for area in [100, 300, 500]:
        plt.scatter([], [], c='k', alpha=0.3, s=area,
                    label=str(area) + ' km$^2$')
    plt.legend(scatterpoints=1, frameon=False,
               labelspacing=1, title='City Area')
    plt.title('California Cities: Area and Population');
    plt.show()



