"""Module for generating a Modified Taylor Diagram. This code is based on the code from Elodie Gutknecht."""

import matplotlib.pyplot as plt
import mpl_toolkits.axisartist.floating_axes as fa
import mpl_toolkits.axisartist.grid_finder as gf
import numpy as np
import pandas as pd
from matplotlib.projections import PolarAxes

########################################################################################################################


class TaylorDiagramPoint(object):
    """
    A single point on a Modified Taylor Diagram.
    How well do the values predicted match the values expected

        * do the means match
        * do the standard deviations match
        * are they correlated
        * what is the normalized error standard deviation
        * what is the bias?

    Notation:
        * s_ = sample standard deviation
        * nrmsd = normalized centered pattern RMS difference;
                  > nrmsd**2 = s_normalized**2 + 1 -
                              2 * s_normalized * 1 * corcoeff

    """

    def __init__(self, nstd, bias, correl, nrmse, pred_name, point_id):
        self.s_normd = nstd
        self.bias = bias
        self.corrcoef = correl
        self.corrcoef = min([self.corrcoef, 1.0])
        self.nrmsd = nrmse
        self.name = pred_name
        self.point_id = point_id


class ModTaylorDiagram(object):
    """
    Plot the standard deviation of the differences and correlation between
    expected and predicted in a single-quadrant polar plot, with
    r=stddev and theta=arccos(correlation).
    """

    def __init__(self, fig=None, label=""):
        """Set up Taylor diagram axes."""

        self.title_polar = r"Correlation"
        self.title_xy = r"Normalized Standard Deviation"
        self.title_expected = r"Data"
        self.max_normed_std = 2.0
        self.s_min = 0

        # Correlation labels
        corln_r = np.append(np.linspace(0.0, 0.9, 10), 0.95)
        corln_ang = np.arccos(corln_r)  # Conversion to polar angles
        grid_loc1 = gf.FixedLocator(corln_ang)  # Positions
        tick_fmttr1 = gf.DictFormatter(dict(zip(corln_ang, map(str, np.round(corln_r, 2)))))

        # Normalized standard deviation axis
        tr = PolarAxes.PolarTransform()
        grid_helper = fa.GridHelperCurveLinear(
            tr,
            extremes=(
                0,
                np.pi / 2,  # 1st quadrant
                self.s_min,
                self.max_normed_std,
            ),
            grid_locator1=grid_loc1,
            tick_formatter1=tick_fmttr1,
        )
        self.fig = fig
        if self.fig is None:
            self.fig = plt.figure(figsize=(10, 8))

        # setup axes
        ax = fa.FloatingSubplot(self.fig, 111, grid_helper=grid_helper)
        # make the axis (polar ax child used for plotting)
        self.ax = self.fig.add_subplot(ax)
        # hide base-axis labels etc
        self.ax.axis["bottom"].set_visible(False)
        self._setup_axes()

        # attach the ploar axes
        self.polar_ax = self.ax.get_aux_axes(tr)

        # Add norm error stddev and nesd==1 contours
        self._plot_req1_cont(label)
        self._plot_nesd_cont(levels=np.arange(0.0, 1.75, 0.25))
        self.points = []

    def add_prediction(self, nstd, bias, correl, nrmse, predictor_name, plot_pt_id):
        """Add a prediction/model to the diagram"""
        this_point = TaylorDiagramPoint(nstd, bias, correl, nrmse, predictor_name, plot_pt_id)
        self.points.append(this_point)

    def plot(self):
        """Place all the loaded points onto the figure"""
        rs = []
        correl = []
        thetas = []
        biases = []
        names = []
        NRMSE = []
        point_tags = []
        markerlist = np.array(["o", "s", "v", "^", "<", ">", "p", "*", "h", "D", "H", "d", "X", "P", "8"])
        markers = []
        ii = 0
        for point in self.points:
            rs.append(point.s_normd)
            correl.append(point.corrcoef)
            thetas.append(np.arccos(point.corrcoef))
            NRMSE.append(point.nrmsd)
            biases.append(point.bias)
            names.append(point.name)
            markers.append(markerlist[ii])
            point_tags.append(point.point_id)
            ii = ii + 1
        print("correlation", correl)
        print("norm STD", rs)
        print("bias", biases)
        print("NRMSE", NRMSE)
        print("name", names)
        minbias = -1.5
        maxbias = 1.5
        #
        # the following step is only to add legend : contours of the markers, without color
        for i, tag in enumerate(point_tags):
            self.polar_ax.scatter(
                thetas[i], rs[i], color="none", edgecolors="black", marker=markers[i], label=names[i], s=85
            )
        plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1.05))
        #
        # now put the markers with colors in the Taylor diagram
        for i, tag in enumerate(point_tags):
            sc = self.polar_ax.scatter(
                [thetas[i]],
                [rs[i]],
                c=[biases[i]],
                edgecolors="none",
                marker=markers[i],
                s=85,
                cmap="seismic",
                vmin=minbias,
                vmax=maxbias,
            )
        #
        self.fig.subplots_adjust(top=0.85)
        cbaxes = self.fig.add_axes([0.238, 0.9, 0.55, 0.03])
        cbar = plt.colorbar(sc, cax=cbaxes, orientation="horizontal", format="%.2f")
        cbaxes.set_xlabel("Normalized bias")
        cbaxes.xaxis.set_ticks_position("top")
        cbaxes.xaxis.set_label_position("top")
        # self.show_key()

    def show_key(self):
        """Add annotation key for model IDs and normalization factors."""
        textstr = ""
        for i, p in enumerate(self.points):
            if i > 0:
                textstr += "\n"
            textstr += "{0}  {1}".format(markers[i], p.name)

        props = {"boxstyle": "round", "facecolor": "white", "alpha": 0.75}
        # place a text box in upper left in axes coords
        self.ax.text(0.85, 0.98, textstr, transform=self.ax.transAxes, fontsize=11, verticalalignment="top", bbox=props)

    def show_norm_factor(self):
        """Add annotation about the normalization factor."""
        n_fact = self.points[0]
        out_str = rf"Norm Factor {n_fact:.2f}"
        x = 0.95 * self.max_normed_std
        y = 0.95 * self.max_normed_std
        self.ax.text(
            x,
            y,
            out_str,
            horizontalalignment="right",
            verticalalignment="top",
            bbox={"edgecolor": "black", "facecolor": "None"},
        )

    def _plot_req1_cont(self, label):
        """Plot the normalized standard deviation = 1 contour and label."""
        my_purple = [0.414, 0.254, 0.609]
        t = np.linspace(0, np.pi / 2)
        r = np.ones_like(t)
        self.polar_ax.plot(t, r, "--", color="b", label=label)
        self.polar_ax.text(
            0, 1, self.title_expected, color="b", horizontalalignment="center", verticalalignment="center"
        )

    def _plot_nesd_cont(self, levels=6):
        """
        plot the normalized error standard deviation contours
               = normalized centered pattern RMS difference;
        """  # noqa: D400, D415
        #    my_blue = [0.171875, 0.39453125, 0.63671875]
        rs, ts = np.meshgrid(np.linspace(self.s_min, self.max_normed_std), np.linspace(0, np.pi / 2))

        nesd = np.sqrt(1.0 + rs**2 - 2 * rs * np.cos(ts))
        contours = self.polar_ax.contour(ts, rs, nesd, levels, colors="g", linestyles="dotted")

        self.polar_ax.clabel(contours, inline=1, fontsize=10)

    def _setup_angle_axis(self):
        """
        set the ticks labels etc for the angle axis
        """
        loc = "top"
        self.ax.axis[loc].set_axis_direction("bottom")
        self.ax.axis[loc].toggle(ticklabels=True, label=True)
        self.ax.axis[loc].major_ticklabels.set_axis_direction("top")
        self.ax.axis[loc].label.set_axis_direction("top")
        self.ax.axis[loc].label.set_text(self.title_polar)

    def _setup_x_axis(self):
        """
        set the ticks labels etc for the x axis
        """
        loc = "left"
        self.ax.axis[loc].set_axis_direction("bottom")
        self.ax.axis[loc].label.set_text(self.title_xy)

    def _setup_y_axis(self):
        """
        set the ticks labels etc for the y axis
        """
        loc = "right"
        self.ax.axis[loc].set_axis_direction("top")
        self.ax.axis[loc].toggle(ticklabels=True)
        self.ax.axis[loc].major_ticklabels.set_axis_direction("left")
        self.ax.axis[loc].label.set_text(self.title_xy)

    def _setup_axes(self):
        """
        set the ticks labels etc for the angle x and y axes
        """
        self._setup_angle_axis()
        self._setup_x_axis()
        self._setup_y_axis()

    def close(self):
        plt.close(self.fig)

    def get_stats(self):
        data = {
            "name": [],
            "correlation_coefficient": [],
            "normalized_standard_deviation": [],
            "bias": [],
            "normalized_root_mean_square_error": [],
        }
        for point in self.points:
            data["name"].append(point.name)
            data["correlation_coefficient"].append(point.corrcoef)
            data["normalized_standard_deviation"].append(point.s_normd)
            data["bias"].append(point.bias)
            data["normalized_root_mean_square_error"].append(point.nrmsd)
        return pd.DataFrame(data)


########################################################################################################################


def generate_mod_taylor_diagram(mtd: ModTaylorDiagram, model: pd.Series, obs: pd.Series, name: str) -> ModTaylorDiagram:
    """Generate a point for the Taylor Diagram using model and observation data."""
    correl_coef = np.corrcoef(obs, model)[0, 1]
    norm_std = np.std(model) / np.std(obs)
    norm_bias = np.mean(model - obs) / np.std(obs)
    rmse = np.sqrt(np.mean(((model - np.mean(model)) - (obs - np.mean(obs))) ** 2))
    norm_rmse = rmse / np.std(obs)
    mtd.add_prediction(norm_std, norm_bias, correl_coef, norm_rmse, name, r"")
    return mtd
