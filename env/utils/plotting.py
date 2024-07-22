import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# [in] ax: axis
# [in] pos_i: np.array having x values in column 0 and y values in column 1
# [in] label_i: label to show in legend
def xy_plot(ax, pos1, label1, pos2, label2):
    ax.plot(pos1[:, 0], pos1[:, 1], 'g', label=label1)
    ax.plot(pos2[:, 0], pos2[:, 1], 'r', label=label2)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')


# [in] ax: axis
# [in] pos_i: np.array having x values in column 0 and z values in column 2
# [in] label_i: label to show in legend
def xz_plot(ax, pos1, label1, pos2, label2):
    ax.plot(pos1[:, 0], pos1[:, 2], 'g', label=label1)
    ax.plot(pos2[:, 0], pos2[:, 2], 'r', label=label2)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m]')


# [in] ax: axis
# [in] pos_i: np.array having y values in column 1 and z values in column 2
# [in] label_i: label to show in legend
def yz_plot(ax, pos1, label1, pos2, label2):
    ax.plot(pos1[:, 1], pos1[:, 2], 'g', label=label1)
    ax.plot(pos2[:, 1], pos2[:, 2], 'r', label=label2)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('y [m]')
    ax.set_ylabel('z [m]')
    

# [in] ax: axis
# [in] pos_i: np.array having x values in column 0, y values in column 1, and z values in column 2
# [in] label_i: label to show in legend
def xt_plot(ax, t1, pos1, label1, t2, pos2, label2):
    ax.plot(t1, pos1[:, 0], 'g', label=label1)
    ax.plot(t2, pos2[:, 0], 'r', label=label2)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('t')
    ax.set_ylabel('x [m]')


# [in] ax: axis
# [in] t_i: np.array containing timestamps
# [in] pos_i: np.array having y [m] values in column 1
# [in] label_i: label to show in legend
def yt_plot(ax, t1, pos1, label1, t2, pos2, label2):
    ax.plot(t1, pos1[:, 1], 'g', label=label1)
    ax.plot(t2, pos2[:, 1], 'r', label=label2)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('t')
    ax.set_ylabel('y [m]')


# [in] ax: axis
# [in] t_i: np.array containing timestamps
# [in] pos_i: np.array having z values in column 2
# [in] label_i: label to show in legend
def zt_plot(ax, t1, pos1, label1, t2, pos2, label2):
    ax.plot(t1, pos1[:, 2], 'g', label=label1)
    ax.plot(t2, pos2[:, 2], 'r', label=label2)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('t')
    ax.set_ylabel('z [m]')


# [in] errs_i: (N,) np.array
# [in] errs_type: Global Position or Reprojection
# [in] titile: title
def cumulative_error_plot(errs_1, label1, errs_2, label2, err_type, title):
    plt.semilogx(sorted(errs_1), np.arange(len(errs_1), dtype=float) / len(errs_1), label=label1)
    if errs_2 is not None:
        plt.semilogx(sorted(errs_2), np.arange(len(errs_2), dtype=float) / len(errs_2), label=label2)
    plt.grid()
    xlabel = err_type + ' errors'
    if err_type == 'Global Position':
        xlabel += ' [m]'
    elif err_type == 'Reprojection':
        xlabel += ' [px]'
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel('Fraction of measurements with smaller error')
    plt.title(title)


# [in] errs: (N,) np.array
# [in] errs_type: string
# [in] title: string
def histogram_error_plot(errs, err_type, title):
    _ = plt.hist(errs, bins='auto')
    xlabel = err_type + ' errors'
    if err_type == 'Global Position':
        xlabel += ' [m]'
    elif err_type == 'Reprojection':
        xlabel += ' [px]'
    plt.xlabel(xlabel)
    plt.ylabel('#')
    plt.title(title)


# [in] init_errs, fin_errs: (N,) np.array
# [in] errs_type: string
# [in] title: string
def compareHistogramsErrorPlot(init_errs, fin_errs, err_type, title):
    _ = plt.hist(init_errs, bins='auto', label='Initial errors')
    _ = plt.hist(fin_errs, bins='auto', label='Optimized errors', color='darkorange', alpha=0.75)
    xlabel = err_type + ' error'
    if err_type == 'Global Position':
        xlabel += ' [m]'
    elif err_type == 'Reprojection':
        xlabel += ' [px]'
    plt.xlabel(xlabel)
    plt.ylabel('#')
    plt.legend()
    plt.title(title)


# [in] ax: axis
# [in] x: (N, ) np.array. Values to plot on the x axis.
# [in] y: (N, ) np.array. Values to plot on the y axis.
# [in] vals: (N, ) np.array. Values to mapped to colors.
def scatter_plot(
        ax, 
        x, x_label, 
        y, y_label, 
        vals, vals_label,
        cmap='viridis'):
    
    pl = ax.scatter(x, y, c=vals, cmap=cmap)
    cbar = plt.colorbar(mappable=pl, ax=ax)
    ax.grid(True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    cbar.set_label(vals_label)

