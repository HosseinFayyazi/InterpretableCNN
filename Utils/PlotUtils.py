import numpy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors


class PlotUtils:
    @staticmethod
    def draw_3d(data, verts, ymin, ymax, line_at_zero=True, colors=True):
        """
        Given verts as a list of plots, each plot being a list
        of (x, y) vertices, generate a 3-d figure where each plot
        is shown as a translucent polygon.
        If line_at_zero, a line will be drawn through the zero point
        of each plot, otherwise the baseline will be at the bottom of
        the plot regardless of where the zero line is.
        :param data:
        :param verts:
        :param ymin:
        :param ymax:
        :param line_at_zero:
        :param colors:
        :return:
        """
        if line_at_zero:
            zeroline = 0
        else:
            zeroline = ymin
        for p in verts:
            p.insert(0, (p[0][0], zeroline))
            p.append((p[-1][0], zeroline))

        if colors:
            hue = 0
            huejump = .27
            facecolors = []
            edgecolors = []
            for v in verts:
                hue = (hue + huejump) % 1
                c = mcolors.hsv_to_rgb([hue, 1, 1])
                # random.uniform(.8, 1),
                # random.uniform(.7, 1)])
                edgecolors.append(c)
                # Make the facecolor translucent:
                facecolors.append(mcolors.to_rgba(c, alpha=.7))
        else:
            facecolors = (1, 1, 1, 0.75)
            edgecolors = (0, 0, 0, 0.5)

        poly = PolyCollection(verts,
                              facecolors=facecolors, edgecolors=edgecolors, linewidth=0.7)

        zs = range(len(data))
        # zs = range(len(data)-1, -1, -1)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        plt.tight_layout(pad=2.0, w_pad=10.0, h_pad=3.0)

        ax.add_collection3d(poly, zs=zs, zdir='y')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Filters')
        ax.set_zlabel('dB')

        ax.set_xlim3d(0, len(data[1]))
        ax.set_ylim3d(-1, len(data))
        ax.set_zlim3d(ymin, ymax)
