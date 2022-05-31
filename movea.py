# -*- coding: utf-8 -*-
"""
Created on Sat May 15 11:49:51 2021

@author: Yu Xing Teo (Prof Pan Research Group DBS NUS)


"""

from os import path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
###################################################################################################
# Import Library
import skimage.io as ski
from kneed import KneeLocator
from matplotlib.patches import FancyArrowPatch
# Import 3D annotation for label of cluster
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from .utils import read_csv, write_csv

# Global constants
Neighbor = 1  # KNN for k=1 calculate the pairing of previous t and current t
# PlotMovementSetting
AnnotationSpacing = 5  # Increase AnnotationSpacing to not display some of the annotation, prevent labels intercept.
AnnotationSize = 5
TextCoords = "offset points"
XYText = (0, 0)
ArrowSize = 2
ArrowStyle = "-|>"
ArrowColor = "b"
SPACING = 10
FIGSIZE = (10, 10)
DOTSIZE = 1
ALPHA = 0.5

# External Source for Annotation (ClusterID Annotation)
class Annotation3D(Annotation):
    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x, y, z = self._xyz
        x2, y2, _ = proj_transform(x, y, z, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)


def _annotate3D(ax, text, xyz, *args, **kwargs):
    """Add anotation `text` to an `Axes3d` instance."""
    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)


setattr(Axes3D, "annotate3D", _annotate3D)

# External Source for Arrow with Direction
class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
        xs, ys, _ = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    """Add an 3d arrow to an `Axes3D` instance."""
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


"""
Convert csv data to correct data format
"""


def csvtodata(data):
    convertlst = []
    for a in data:
        k = []
        for i in a:
            j = i.split(",")
            x = float(j[0].split("(")[-1])
            y = float(j[1])
            z = float(j[2].split(")")[0])
            k.append((x, y, z))
        convertlst.append(k)
    return convertlst


"""
Color Code Lst for ploting raw data
"""


def colorlst(totaltime, timelap):
    R = 0
    colorcode = []
    for _ in range(0, totaltime, timelap):
        colorcode.append([(R, 0, 0)])
        R = R + 1 / (totaltime - 1)
    colorcode[-1] = [(1, 0, 0)]
    return colorcode


"""
Euclidean distance
"""


def distance(coord1, coord2):
    return (
        (coord1[0] - coord2[0]) ** 2
        + (coord1[1] - coord2[1]) ** 2
        + (coord1[2] - coord2[2]) ** 2
    ) ** 0.5


"""
RGB DETECTION
"""


def detection(cur_RGB, Thres_RGB, Switch):
    decision = 1
    for i in range(len(Switch)):
        if Switch[i] == 1 and cur_RGB[i] < Thres_RGB[i]:
            decision = 0
            break
        if (
            Switch[i] == 0 and cur_RGB[i] > Thres_RGB[i]
        ):  # Theory RGB[0]<=ThresRGB[0] and RGB[1]>=ThresRGB[1] and RGB[2]<=ThresRGB[2]:
            decision = 0
            break
    return decision


def Signal_detection(
    timestamps_with_slices: List[Tuple[str, List[str]]],
    dimension_x: float,
    dimension_deltaz: float,
    output_name: str,
    output_dir: str,
    timelap: int = 1,
    ThresRGB: Tuple[int, int, int] = (255, 100, 0),
    ChannelSwitch: Tuple[int, int, int] = (0, 1, 0),
):
    """Detects signal in a sample.
    :param timestamps_with_slices: List of directory (timestamps) and its corresponding image files (slices) in order.
    """
    # Get the dimension from the first image
    first_image = ski.imread(
        path.join(timestamps_with_slices[0][0], timestamps_with_slices[0][1][0])
    )

    typixel = first_image.shape[0]
    txpixel = first_image.shape[1]

    dimension_perpixel = dimension_x / txpixel
    delta_z = dimension_deltaz / dimension_perpixel

    # List of the coordination of the data point
    lstofcoordinate: List[List[Tuple[int, int, int]]] = []

    # Looping for different time set (t=0 to t=n, with a step of timelap) and detect user-specified RGB datapoints
    for m in range(0, len(timestamps_with_slices), timelap):
        root_dir, lstoffile = timestamps_with_slices[m]

        # Initiate the axis
        z = 0
        coordinate = []
        # Looping for different Z-axis values to detect user-specified RGB datapoints
        for filename in lstoffile:
            # Read the respective tif file in the folder (with the pre-generated file name eg. S4 2\T00001\T00001\C02Z001)
            image = ski.imread(path.join(root_dir, filename))
            # Loop for the x axis and y axis of the tiff image (pixel by pixel)
            for row in range(typixel):
                for column in range(txpixel):
                    RGB = image[row, column]  # RGB at the specific pixel
                    if detection(RGB, ThresRGB, ChannelSwitch):
                        # RGB filter: Filtered the pixel with smaller Red, larger Blue and Green then preset threshold (ThresRGB)
                        # Assign the pixel with specific coordination (x,y,z)
                        coordinate.append((column, row, z))
            z = z + delta_z
        lstofcoordinate.append(coordinate)
        print("Time: {0:.2f}%".format((m + 1) / len(timestamps_with_slices) * 100))

    # Save the RGB filtered raw data in csv format
    write_csv(output_dir, output_name + ".csv", lstofcoordinate)
    raw_data = read_csv(path.join(output_dir, output_name + ".csv"))
    Fig0(
        raw_data,
        len(timestamps_with_slices),
        timelap,
        txpixel,
        typixel,
        dimension_perpixel,
        output_name=output_name,
        output_dir=output_dir,
    )
    Fig1(
        raw_data,
        len(timestamps_with_slices),
        timelap,
        txpixel,
        typixel,
        dimension_perpixel,
        output_name=output_name,
        output_dir=output_dir,
    )

    return txpixel, typixel


def Fig0(
    rawdata,
    totaltime,
    timelap,
    txpixel,
    typixel,
    dimension_perpixel,
    output_name,
    output_dir: str = ".",
):
    # Convert RGB filtered raw data into correct format
    convertlst = csvtodata(rawdata)
    colorcode = colorlst(totaltime, timelap)

    # plot the raw data
    fig0 = plt.figure(figsize=FIGSIZE)
    ax = fig0.add_subplot(projection="3d")
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.yaxis.set_ticks(np.arange(0, typixel * dimension_perpixel, SPACING))
    ax.xaxis.set_ticks(np.arange(0, txpixel * dimension_perpixel, SPACING))

    ax.set_title("Raw Data")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel("z (µm)")

    for w in range(len(convertlst)):
        plotzip = list(convertlst[w])
        x, y, z = zip(*plotzip)
        x_axis, y_axis, z_axis = list(x), list(y), list(z)

        x_axis = list(map(lambda x: x * dimension_perpixel, x_axis))
        y_axis = list(map(lambda x: x * dimension_perpixel, y_axis))
        z_axis = list(map(lambda x: x * dimension_perpixel, z_axis))

        ax.scatter3D(x_axis, y_axis, z_axis, s=DOTSIZE, c=colorcode[w], alpha=ALPHA)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    # ax.view_init(azim=-90, elev=-90)
    figManager = plt.get_current_fig_manager()
    if hasattr(figManager, "window"):
        figManager.window.showMaximized()

    fig0.savefig(path.join(output_dir, output_name + "_0.tiff"))


def Fig1(
    rawdata,
    totaltime,
    timelap,
    txpixel,
    typixel,
    dimension_perpixel,
    output_name,
    output_dir: str = ".",
):  # raw data XY view
    # Convert RGB filtered raw data into correct format
    convertlst = csvtodata(rawdata)
    colorcode = colorlst(totaltime, timelap)

    # plot the rawdata
    fig1 = plt.figure(figsize=FIGSIZE)
    ax = fig1.add_subplot(projection="3d")
    # ax.set_ylim(ax.get_ylim()[::-1])
    ax.yaxis.set_ticks(np.arange(0, typixel * dimension_perpixel, SPACING))
    ax.xaxis.set_ticks(np.arange(0, txpixel * dimension_perpixel, SPACING))

    ax.set_title("Raw Data")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel("z (µm)")

    for w in range(len(convertlst)):
        plotzip = list(convertlst[w])
        x, y, z = zip(*plotzip)
        x_axis, y_axis, z_axis = list(x), list(y), list(z)

        x_axis = list(map(lambda x: x * dimension_perpixel, x_axis))
        y_axis = list(map(lambda x: x * dimension_perpixel, y_axis))
        z_axis = list(map(lambda x: x * dimension_perpixel, z_axis))

        ax.scatter3D(x_axis, y_axis, z_axis, s=DOTSIZE, c=colorcode[w], alpha=ALPHA)
    ax.view_init(azim=-90, elev=-90)  # 270,90
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    # plt.zticks([])
    figManager = plt.get_current_fig_manager()
    if hasattr(figManager, "window"):
        figManager.window.showMaximized()
    # cbar = plt.colorbar()
    # cbar.set_label('Color Intensity')

    fig1.savefig(path.join(output_dir, output_name + "_1.tiff"))


"""
Clustering
"""


def optimal_eps(name, convertlst, min_sig):
    # Initialize
    lsteps = []
    lstelbow = []
    lstknee = []
    figd = plt.figure()
    ax = figd.add_subplot()

    for i in range(len(convertlst)):
        data = np.array(convertlst[i])  # convert to array

        # k-dist for epsilon using KNN with n_neighbors = min_sig
        dataset = data
        neighbors = NearestNeighbors(n_neighbors=min_sig)  # laman term use 2
        neighbors_fit = neighbors.fit(dataset)
        distances, _ = neighbors_fit.kneighbors(dataset)

        # Plot the k-dist plot
        distances = np.sort(distances, axis=0)
        distances = distances[
            :, 1
        ]  # only take the second column, because we wish to find the first nearest distance (first column is the distacnce with itself, third column is the second nearest distance)
        plt.plot(distances)

        # Find the point of maximum curvature of the k-dist plot
        x = [i for i in range(len(distances))]
        y = list(distances)
        kneedle = KneeLocator(x, y, S=1.0, curve="convex", direction="increasing")
        lstknee.append(round(kneedle.knee, 3))
        lstelbow.append(round(kneedle.elbow, 3))
        lsteps.append(round(kneedle.knee_y, 3))
        # Normalized data, normalized knee, and normalized distance curve.
        # kneedle.plot_knee_normalized()
        # Raw data and knee.
        # kneedle.plot_knee()
    ax.set_ylabel("K-NN Distance")
    ax.set_xlabel("Points sorted by distance (" + name + ")")
    figd.savefig(name + "_kdist.tiff")

    # Plot the distribution of all Eps values (for all time points t0 to t-end)
    figeps = plt.figure(figsize=(2.5, 10))
    ax = figeps.add_subplot()
    ax.set_ylabel("eps")
    ax.set_xlabel(name)
    plt.boxplot(lsteps)
    figeps.savefig(name + "_eps.tiff")

    # The optimal Eps is the median of all Eps values (for all time points t0 to t-end)
    epsilon_auto = np.median(lsteps)
    return epsilon_auto


def clustering(
    input_file,
    output_dir,
    output_name,
    epsilon,
    min_sig,
    is_auto_eps,
    txpixel,
    typixel,
    dimension_perpixel,
):
    # Retrieve raw coordination data from saved csv file and Convert the raw coordination data from excel into correct format
    file = read_csv(input_file)
    convertlst = csvtodata(file)
    reslst = []
    if is_auto_eps:
        # If True, use automatically calculated Eps, if false, manually input Eps
        epsilon = optimal_eps(output_name, convertlst, min_sig)

    # Setup ploting plateform to plot clustering result
    fig2 = plt.figure(figsize=FIGSIZE)
    ax = fig2.add_subplot(projection="3d")
    ax.yaxis.set_ticks(np.arange(0, typixel * dimension_perpixel, SPACING))
    ax.xaxis.set_ticks(np.arange(0, txpixel * dimension_perpixel, SPACING))

    # ax.set_title('Clustering')
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel("z (µm)")

    # Looping for different time set (t=0 to t=n) and perform clustering
    performance = [
        ["Eps", "Min_sig"],
        [epsilon, min_sig],
        ["DBSCAN #Cluster"],
    ]  # list to track performance
    for i in range(len(convertlst)):
        data = np.array(convertlst[i])  # convert to array

        # Using DBSCAN(Density-Based Spatial Clustering of Applications with Noise) to do clustering, eps and min_samples is the clustering parameter
        model = DBSCAN(eps=epsilon, min_samples=min_sig)
        model.fit_predict(data)

        # Clustering result: cluster with respective clustering label eg. [ 0  0  1 ... 10 16 27]
        res = model.labels_
        reslst.append(res)

        # Plot 3D data with different color respectively to the cluster ID CHECK2 c=model.labels_
        x_axis = data[:, 0]
        y_axis = data[:, 1]
        z_axis = data[:, 2]

        x, y, z, ID = [], [], [], []
        for idx in range(len(model.labels_)):
            if model.labels_[idx] != -1:
                x.append(x_axis[idx])
                y.append(y_axis[idx])
                z.append(z_axis[idx])
                ID.append(model.labels_[idx])
        x_axis, y_axis, z_axis = x, y, z

        x_axis = list(map(lambda x: x * dimension_perpixel, x_axis))
        y_axis = list(map(lambda x: x * dimension_perpixel, y_axis))
        z_axis = list(map(lambda x: x * dimension_perpixel, z_axis))
        ax.scatter(x_axis, y_axis, z_axis, c=ID, s=DOTSIZE)

        # Print the number of clusters found in each timepoint
        performance.append([len(set(model.labels_))])  # ,len(set(clusterer.labels_))

    # plot the clustering visualization graph
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.set_ylim(ax.get_ylim()[::-1])
    figManager = plt.get_current_fig_manager()
    if hasattr(figManager, "window"):
        figManager.window.showMaximized()
    fig2.savefig(path.join(output_dir, output_name + "_2.tiff"))  # Fig2 clustering 3D
    ax.set_zticks([])
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.view_init(azim=-90, elev=-90)
    fig2.savefig(path.join(output_dir, output_name + "_2a.tiff"))  # Fig2a clustering XY

    # save the clustering result in csv (datapoints with respective assigned clusters)
    # Clustering result
    write_csv(output_dir, output_name + ".csv", reslst)
    # Clustering performance
    write_csv(output_dir, output_name + "_performance.csv", performance)


"""
MOVEMENT CALCULATION
"""


def movement(
    input_raw,
    input_clustering,
    min_signals,
    max_signals,
    min_range,
    max_range,
    min_frame,
    dimension_perpixel,
    speed_time,
    totaltime,
    txpixel,
    typixel,
    output_dir,
    output_name,
    auto_calc=True,
    timelap: int = 1,
):
    interval_time = speed_time / totaltime
    Thres3 = min_range / dimension_perpixel / interval_time
    Thres4 = max_range / dimension_perpixel / interval_time
    sorted_convertlst, sorted_class = LoadData(input_raw, input_clustering)
    cluster_range = [min_signals, max_signals]
    (
        all_cluster_centers,
        lst_cluster_centers,
        filtered_cluster_coord,
    ) = CalcClusterCenter(
        sorted_convertlst,
        sorted_class,
        cluster_range,
        output_dir,
        output_name,
        boolen=auto_calc,
    )
    initial_cluster = InitialCluster(all_cluster_centers, Thres4)
    initial_cluster = LinkingCluster(
        initial_cluster, lst_cluster_centers, Thres3, Thres4
    )
    (
        all_cluster_centers,
        clustering_name,
        lst_of_distance,
        dic_speed,
        dic_distance,
        lst_of_speed,
    ) = CalcSpeedDisplacement(
        initial_cluster,
        min_frame,
        dimension_perpixel,
        interval_time,
        output_dir,
        output_name,
    )
    PlotMovement(
        filtered_cluster_coord,
        all_cluster_centers,
        clustering_name,
        dic_speed,
        dic_distance,
        totaltime,
        timelap,
        txpixel,
        typixel,
        dimension_perpixel,
        output_dir,
        output_name,
    )
    Spectrum(lst_of_distance, lst_of_speed, output_dir, output_name)


def LoadData(input_raw, input_clustering):
    # Retrieve both coordination and clustering data -> convert to correct format
    file_detection = read_csv(input_raw)  # retrieve
    file_clustering = read_csv(input_clustering)
    reslst = []
    for res in file_clustering:
        reslst.append([int(item) for item in res])

    convertlst = csvtodata(file_detection)

    # Sorted both lists according to clustering label (ascending from cluster 0 to cluster n)
    Sorted_class = []
    Sorted_convertlst = []
    res = []
    for km in range(len(reslst)):
        list1, list2 = zip(
            *sorted(zip(reslst[km], convertlst[km]))
        )  # Sorted the detection result w.r.t clustering result
        filter1 = []
        filter2 = []
        for i in range(len(list1)):
            if list1[i] != -1:  # Filtered out noise (label with ClusterID = -1)
                filter1.append(list1[i])
                filter2.append(list2[i])
        Sorted_convertlst.append(filter2)
        Sorted_class.append(filter1)
    return Sorted_convertlst, Sorted_class


def OptimalCluster_range(Sorted_class, name):
    # Automatically calculated Cluster_range
    # Count number of signals in each cluster
    Count_clusterID_dict = []
    lstdistribution = []
    from collections import Counter

    for i in range(len(Sorted_class)):
        Count_clusterID_dict.append(Counter(Sorted_class[i]))
        lstdistribution.extend(
            list(Counter(Sorted_class[i]).values())
        )  # 1D array contain number of signals in each cluster

    # Calculated outlier cluster w.r.t number of signals
    from scipy.stats import iqr

    Q1 = np.percentile(lstdistribution, 25)
    Q3 = np.percentile(lstdistribution, 75)
    # Q2 = np.median(lstdistribution)
    # ThresAve = np.average(lstdistribution)
    RangeIQR = iqr(lstdistribution, axis=0)

    Thres2 = Q1 - 1.5 * RangeIQR
    Thres1 = Q3 + 1.5 * RangeIQR
    Cluster_range = [Thres2, Thres1]

    # Boxplot: the number of signals for each cluster
    figclusterN = plt.figure(figsize=FIGSIZE)
    ax = figclusterN.add_subplot()
    plt.boxplot(lstdistribution)
    ax.set(xlabel=name, ylabel="Number of signals in cluster")
    figclusterN.savefig(name + "_clusterdistribution.tiff")

    return Cluster_range, Count_clusterID_dict


def CalcClusterCenter(
    Sorted_convertlst, Sorted_class, Cluster_range, output_dir, output_name, boolen=True
):
    # 2nd filtering: filtered out the large cluster (static) and small cluster (noise) based on Cluster_range (Recommend 70_15_30 the better)
    Filter_coord = []
    Filter_clusterID = []

    # Calculate Cluster_range if boolen == True
    if boolen:
        Cluster_range, Count_clusterID_dict = OptimalCluster_range(
            Sorted_class, output_name
        )
    else:
        _, Count_clusterID_dict = OptimalCluster_range(Sorted_class, output_name)

    for i in range(len(Sorted_convertlst)):
        Coord = []
        ClusterID = []
        for j in range(len(Sorted_convertlst[i])):
            if (
                Cluster_range[0]
                <= Count_clusterID_dict[i].get(Sorted_class[i][j])
                <= Cluster_range[1]
            ):  # Filtered out cluster with large & small number of datapoint, Counter_clusterID_dict is a dictionary for instance, Counter_clusterID_dict[1] is at the second time point, {0:11543, 1: 115, 2:333, 4:545}, meaning at second time point, have 333 signals label with '2'.
                Coord.append(
                    Sorted_convertlst[i][j]
                )  # keep the signals with a clusterID which has number of signals in the Cluster_range
                ClusterID.append(Sorted_class[i][j])
        Filter_coord.append(Coord)
        Filter_clusterID.append(ClusterID)

    # Store 2nd filtering data for ploting
    Filtered_Cluster_Coord = Filter_coord
    # Calculate the centre point for each cluster, method: Centre point = median of the datapoints in the cluster
    Lst_Cluster_Centers = []
    Flatten_Cluster_Points = (
        []
    )  # 1D array with all cluster centers regardless of t-axis
    for j in range(len(Filter_coord)):
        Cur_clusterID = Filter_clusterID[j][
            0
        ]  # take the first clusterID (each time point did the same)
        Cur_cluster_signals = []
        cur_cluster_centers = []
        for k in range(len(Filter_coord[j])):
            if (
                Cur_clusterID == Filter_clusterID[j][k]
            ):  # Group all signals with same clusterID,
                Cur_cluster_signals.append(Filter_coord[j][k])
            else:
                cur_cluster_signals = list(Cur_cluster_signals)
                Flatten_Cluster_Points.append(
                    cur_cluster_signals
                )  # 1D array with all cluster centers regardless of t-axis for validation purpose
                x, y, z = zip(*cur_cluster_signals)
                xa, ya, za = list(x), list(y), list(z)
                cur_cluster_centers.append(
                    [np.median(xa), np.median(ya), np.median(za)]
                )
                Cur_cluster_signals = []
                Cur_cluster_signals.append(Filter_coord[j][k])
                Cur_clusterID = Filter_clusterID[j][k]
        Lst_Cluster_Centers.append(cur_cluster_centers)

    write_csv(
        output_dir, output_name + "_validation_XYZ.csv", Flatten_Cluster_Points
    )  # To validate different Z-slices are clustered together instead of each Z-slice is one cluster by calculating the variation of Z-position and visualize each cluster seperately

    # Convert the lst_Cluster_Centers [[[Center]]] into Tuple [((Center,),)]
    All_Cluster_Centers = []
    for k in range(len(Lst_Cluster_Centers)):
        frame = ()
        for j in range(len(Lst_Cluster_Centers[k])):
            frame += (tuple(Lst_Cluster_Centers[k][j]),)  # all point in
        All_Cluster_Centers.append(frame)

    return All_Cluster_Centers, Lst_Cluster_Centers, Filtered_Cluster_Coord


def InitialCluster(All_Cluster_Centers, Thres4):
    # Initiate the initial clusters for each initial time point, initial clusters: The set of clusters serve as the initial point for the mapping.
    # Thres4: The minimum length different to consider as unique initial cluster (compared to other initial clusters)
    # Running four loops to generate all permuatations of cluster pairs
    initialcluster = []
    idx = 0
    initialcluster.append(list(All_Cluster_Centers[0]))  # t0 all are initial clusters
    for _ in range(len(All_Cluster_Centers)):  # Loop each time point (t=x)
        not_initialcluster = set()
        while (
            idx != len(All_Cluster_Centers) - 2
        ):  # -2 instead of -1 because we compare between two time lapses, those last time lapse can't compare with any, -1 because idx count from 0
            for firstcluster in All_Cluster_Centers[
                idx
            ]:  # Loop each cluster in the time point (t=x)
                for secondcluster in All_Cluster_Centers[
                    idx + 1
                ]:  # Loop each cluster in the time point (t=x+1)
                    if (
                        distance(firstcluster, secondcluster) <= Thres4
                    ):  # identified not_intitialcluster if a cluster has a distance smaller/equal to the maximum moving distance with another cluster in next time lapse
                        not_initialcluster.add(tuple(secondcluster))
            Clusters_t2 = set(All_Cluster_Centers[idx + 1])
            Initialcluster_t2 = Clusters_t2 - not_initialcluster
            initialcluster.append(list(Initialcluster_t2))
            idx += 1
    # Convert the initialcluster from 2D [[(InitialCluster1),(InitialCluster2)]] to 3D [[[(InitialCluster)],[(InitialCluster2)]]] to append the tracks after the InitialCluster in next step
    for i in range(len(initialcluster)):
        initialcluster[i] = list(map(lambda x: [x], initialcluster[i]))
    return initialcluster


def LinkingCluster(initialcluster, Lst_Cluster_Centers, Thres3, Thres4, Neighbor=1):
    # Linking
    # Mapping the movement of the clusters.
    # Map the centre of the cluster X (time point t0) to the centre of the cluster Y (time point t1), Mapping criteria (shortest movement require from one cluster to another cluster)
    # Thres4: The maximum valid moving distance from cluster X (time point t0) and Y (time point t1), beyond the threshold consider the movement as mismatched
    InitialClusters = (
        initialcluster.copy()
    )  # InitialClusters is the list with every initial clusters in each initial time points
    for timet0 in range(
        len(InitialClusters)
    ):  # Loop each time point (initial time points)
        for clustert0 in range(
            len(InitialClusters[timet0])
        ):  # Loop each initial cluster in the time point
            for timet1 in range(
                timet0 + 1, len(Lst_Cluster_Centers)
            ):  # Loop each time points (t initial +1 till the end) after initial time point
                current_cluster = InitialClusters[timet0][clustert0][
                    -1
                ]  # Current cluster: current most updated cluster (t<n>)for the specific initial cluster (t0)
                OptimalLinking = [
                    float("inf"),
                    None,
                ]  # Initialize a list to store ([the distance, the cluster index]) the cluster (t<n+1>) which have the shortest distance w.r.t current most updated cluster (t<n>)
                clusters_t2 = Lst_Cluster_Centers[timet1]  # Cluster for KNN
                neigh = NearestNeighbors()
                neigh.fit(clusters_t2)
                OptimalLinking[0], OptimalLinking[1] = neigh.kneighbors(
                    np.array([current_cluster]), Neighbor, return_distance=True
                )  # OptimalLinking example, [np.array([[0]]),np.array([[2.0]])], represents nearest cluster distance and nearest cluster index respectively
                OptimalLinking[0], OptimalLinking[1] = (
                    OptimalLinking[0][0][0],
                    OptimalLinking[1][0][0],
                )
                while (
                    OptimalLinking[0] < Thres3
                ):  # If nearest cluster distance smaller than Thres3 (noise), remove the nearest cluster and recalculated
                    clusters_t2.remove(clusters_t2[OptimalLinking[1]])
                    neigh = NearestNeighbors()
                    neigh.fit(clusters_t2)
                    OptimalLinking[0], OptimalLinking[1] = neigh.kneighbors(
                        np.array([current_cluster]), Neighbor, return_distance=True
                    )
                    OptimalLinking[0], OptimalLinking[1] = (
                        OptimalLinking[0][0][0],
                        OptimalLinking[1][0][0],
                    )
                if (
                    OptimalLinking[0] <= Thres4
                ):  # If nearest cluster distance smaller than/equal to Thres4 (acceptable moving distance), stored this nearest cluster (t<x+1>) as optimal linked cluster for the current_cluster (t<x>)
                    InitialClusters[timet0][clustert0].append(
                        tuple(clusters_t2[OptimalLinking[1]])
                    )
    return InitialClusters


def CalcSpeedDisplacement(
    InitialClusters,
    Min_Frame,
    dimension_perpixel,
    Interval_time,
    output_dir,
    output_name,
):
    # Movement Calc.
    All_Tracks = InitialClusters
    # Flatten the InitialClusters into 1D (2D: a. time axis and b. movement of clusters), [[[t0: cluster movement1],[t0: clustermovement2]],[t1:cluster movement1]] -> [[t0 cluster movement1], [t0 cluster movement2], [t1 cluster movement1]]
    Compile_lst = []  # Flattened InitialClusters
    timeframe = []  # Number of time lapses respective to each track
    for Tracks in All_Tracks:
        for single_Track in Tracks:
            if (
                len(single_Track) >= Min_Frame
            ):  # Not only initial cluster and moving time lapse larger than/equal to Min_Frame
                Compile_lst.append(single_Track)
                timeframe.append(len(single_Track))
    ##################################
    All_Tracks = Compile_lst
    # Calc. Movement and Speed
    lstofspeed = []  # Speed of each clusters
    lstdistance = []  # Movement of each clusters
    lstframe = []  # Number of time lapses respective to each track
    for j in range(len(All_Tracks)):  # Loop each track
        cur_distance = []
        cur_speed = []
        for i in range(
            1, len(All_Tracks[j])
        ):  # Loop each time lapse in each track, start from 1 because of the i-1
            cur_distance.append(
                distance(All_Tracks[j][i - 1], All_Tracks[j][i]) * dimension_perpixel
            )  # Convert movement (in pixel) into movement (in µm)
            cur_speed.append(
                distance(All_Tracks[j][i - 1], All_Tracks[j][i])
                * dimension_perpixel
                / Interval_time
            )  # Convert speed (in pixel/s) into speed (in µm/s)
        lstdistance.append(sum(cur_distance))
        lstofspeed.append(np.average(cur_speed))
        lstframe.append(len(cur_speed))

    # Store the movement and speed into dictionaries (format: key of dictionary = label of the cluster, value of dictionary = movement of the cluster)
    dic_distance = {}
    dic_speed = {}
    dic_frame = {}
    for j in range(len(lstdistance)):
        dic_distance[str(j)] = lstdistance[j]
        dic_speed[str(j)] = lstofspeed[j]
        dic_frame[str(j)] = lstframe[j]

    All_Cluster_Centers = All_Tracks
    # Save movement and speed
    clustering_name = list(dic_distance.keys())
    lst_of_speed = list(dic_speed.values())
    lst_of_distance = list(dic_distance.values())

    Header = ["Cluster Id", "Total Movement (µm)", "Speed (µm/s)", "Total Time Lapses"]
    Final_Result = [Header]

    # Save details (movement µm, speed µm/s) of each cluster
    for cluster in list(dic_distance.keys()):
        Final_Result.append(
            [cluster, dic_distance[cluster], dic_speed[cluster], dic_frame[cluster]]
        )

    # Save Total Median and Total Average w.r.t each clusters
    Final_Result.append(
        [
            "Median",
            str(np.median(np.array(lst_of_distance))),
            str(np.median(np.array(lst_of_speed))),
        ]
    )
    Final_Result.append(
        [
            "Average",
            str(np.average(np.array(lst_of_distance))),
            str(np.average(np.array(lst_of_speed))),
        ]
    )

    print(
        "Average",
        str(np.average(np.array(lst_of_distance))),
        str(np.average(np.array(lst_of_speed))),
    )  # Print1
    print(
        "Median",
        str(np.median(np.array(lst_of_distance))),
        str(np.median(np.array(lst_of_speed))),
    )  # Print2

    # Speed Peak
    density = stats.gaussian_kde(lst_of_speed)
    xs = lst_of_speed
    ys = density(xs)
    index = np.argmax(ys)
    max_y = ys[index]
    max_x = xs[index]
    Peak_speed = [max_y, max_x]

    # Displacement Peak
    density = stats.gaussian_kde(lst_of_distance)
    xs = lst_of_distance
    ys = density(xs)
    index = np.argmax(ys)
    max_y = ys[index]
    max_x = xs[index]
    Peak_displacement = [max_y, max_x]

    Final_Result.append(
        [
            "Peak of Speed Spectrum",
            str(Peak_speed[1]) + " µm/s",
            str(Peak_speed[0]) + " (Density)",
        ]
    )
    Final_Result.append(
        [
            "Peak of Displacement Spectrum",
            str(Peak_displacement[1]) + " µm",
            str(Peak_displacement[0]) + " (Density)",
        ]
    )

    print(
        "Peak of Speed Spectrum",
        str(Peak_speed[1]) + " µm/s",
        str(Peak_speed[0]) + " (Density)",
    )  # Print3
    print(
        "Peak of Displacement Spectrum",
        str(Peak_displacement[1]) + " µm",
        str(Peak_displacement[0]) + " (Density)",
    )  # Print4

    # Save as .csv file
    write_csv(output_dir, output_name + ".csv", Final_Result)  # File3

    return (
        All_Cluster_Centers,
        clustering_name,
        lst_of_distance,
        dic_speed,
        dic_distance,
        lst_of_speed,
    )


def PlotMovement(
    Filtered_Cluster_Coord,
    All_Cluster_Centers,
    clustering_name,
    dic_speed,
    dic_distance,
    totaltime,
    timelap,
    txpixel,
    typixel,
    dimension_perpixel,
    output_dir,
    output_name,
    highlightedClusters={},
):
    # Regenerate the color code to represent the t-axis
    colorcode = colorlst(totaltime, timelap)
    # Plot movement graph (three graph): a. Movement Display b. Merge Raw Data with Movement c. Specific Cluster Display
    # Setup the plotting framework for subplot a. Movement Display
    fig3 = plt.figure(figsize=FIGSIZE)
    ax = fig3.add_subplot(111, projection="3d")
    # ax.set_xlim(0,550)
    # ax.set_ylim(0,550)
    # ax.set_zlim(0,zmax)
    # ax.set_ylim(ax.get_ylim()[::-1])
    ax.yaxis.set_ticks(np.arange(0, typixel * dimension_perpixel, SPACING))
    ax.xaxis.set_ticks(
        np.arange(0, txpixel * dimension_perpixel, SPACING)
    )  # 550 replace txpixel
    setattr(Axes3D, "annotate3D", _annotate3D)
    setattr(Axes3D, "arrow3D", _arrow3D)
    # ax.set_title(name+' Movement')
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel("z (µm)")
    fig3.tight_layout()
    location_annoted = []

    for w in range(
        len(Filtered_Cluster_Coord)
    ):  # not raw data, is after filter (w.r.t Cluster_range)
        plotzip = list(Filtered_Cluster_Coord[w])
        x, y, z = zip(*plotzip)
        x_axis, y_axis, z_axis = list(x), list(y), list(z)
        x_axis = list(map(lambda x: x * dimension_perpixel, x_axis))
        y_axis = list(map(lambda x: x * dimension_perpixel, y_axis))
        z_axis = list(map(lambda x: x * dimension_perpixel, z_axis))

        ax.scatter3D(x_axis, y_axis, z_axis, c="w", s=DOTSIZE, alpha=0)

    for dataset in range(len(All_Cluster_Centers)):
        plotzip = list(All_Cluster_Centers[dataset])
        x, y, z = zip(*plotzip)
        x_axis, y_axis, z_axis = list(x), list(y), list(z)
        x_axis = list(map(lambda x: x * dimension_perpixel, x_axis))
        y_axis = list(map(lambda x: x * dimension_perpixel, y_axis))
        z_axis = list(map(lambda x: x * dimension_perpixel, z_axis))

        for i in range(
            0, len(x_axis) - 1
        ):  # -1 is because two points result in one arrow
            ax.arrow3D(
                x_axis[i],
                y_axis[i],
                z_axis[i],
                x_axis[i + 1] - x_axis[i],
                y_axis[i + 1] - y_axis[i],
                z_axis[i + 1] - z_axis[i],
                mutation_scale=ArrowSize,
                arrowstyle=ArrowStyle,
                color=ArrowColor,
            )  # ax.arrow3D(start of arrow, displacement of arrow, arrow setting)
        decision = True
        for (
            j
        ) in (
            location_annoted
        ):  # precheck to prevent new annotation overlap with existing annotation
            if int(x_axis[0]) in range(
                j[0] - AnnotationSpacing, j[0] + AnnotationSpacing
            ):
                if int(y_axis[0]) in range(
                    j[1] - AnnotationSpacing, j[1] + AnnotationSpacing
                ):
                    decision = False
        if decision:  # Annotated
            ax.annotate3D(
                clustering_name[dataset],
                (x_axis[0], y_axis[0], z_axis[0]),
                xytext=XYText,
                textcoords=TextCoords,
                fontsize=AnnotationSize,
            )  # 'cluster'+str(dataset+1)
            location_annoted.append((int(x_axis[0]), int(y_axis[0]), int(z_axis[0])))

    ax.view_init(azim=-90, elev=-90)
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    # Remove grid
    ax.grid(False)
    ax.set_zticks([])
    figManager = plt.get_current_fig_manager()
    if hasattr(figManager, "window"):
        figManager.window.showMaximized()

    fig3.savefig(path.join(output_dir, output_name + "_3.tiff"))  # Movement Display

    # Setup the plotting framework for subplot b. Merge Raw Data with Movement
    fig4 = plt.figure(figsize=FIGSIZE)
    ax = fig4.add_subplot(111, projection="3d")
    # ax.set_xlim(0,550)
    # ax.set_ylim(0,550)
    # ax.set_zlim(0,zmax)
    # ax.set_ylim(ax.get_ylim()[::-1])
    ax.yaxis.set_ticks(np.arange(0, typixel * dimension_perpixel, SPACING))
    ax.xaxis.set_ticks(
        np.arange(0, txpixel * dimension_perpixel, SPACING)
    )  # 550 replace txpixel
    setattr(Axes3D, "annotate3D", _annotate3D)
    setattr(Axes3D, "arrow3D", _arrow3D)
    # ax.set_title(name+' Movement')
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel("z (µm)")
    fig4.tight_layout()
    location_annoted = []
    # Different
    for w in range(
        len(Filtered_Cluster_Coord)
    ):  # not raw data, is after filter (w.r.t Cluster_range)
        plotzip = list(Filtered_Cluster_Coord[w])
        x, y, z = zip(*plotzip)
        x_axis, y_axis, z_axis = list(x), list(y), list(z)
        x_axis = list(map(lambda x: x * dimension_perpixel, x_axis))
        y_axis = list(map(lambda x: x * dimension_perpixel, y_axis))
        z_axis = list(map(lambda x: x * dimension_perpixel, z_axis))

        ax.scatter3D(x_axis, y_axis, z_axis, c=colorcode[w], s=DOTSIZE, alpha=ALPHA)
    # Different

    for dataset in range(len(All_Cluster_Centers)):
        plotzip = list(All_Cluster_Centers[dataset])
        x, y, z = zip(*plotzip)
        x_axis, y_axis, z_axis = list(x), list(y), list(z)
        x_axis = list(map(lambda x: x * dimension_perpixel, x_axis))
        y_axis = list(map(lambda x: x * dimension_perpixel, y_axis))
        z_axis = list(map(lambda x: x * dimension_perpixel, z_axis))

        for i in range(
            0, len(x_axis) - 1
        ):  # -1 is because two points result in one arrow
            ax.arrow3D(
                x_axis[i],
                y_axis[i],
                z_axis[i],
                x_axis[i + 1] - x_axis[i],
                y_axis[i + 1] - y_axis[i],
                z_axis[i + 1] - z_axis[i],
                mutation_scale=ArrowSize,
                arrowstyle=ArrowStyle,
                color=ArrowColor,
            )  # ax.arrow3D(start of arrow, displacement of arrow, arrow setting)
        decision = True
        for (
            j
        ) in (
            location_annoted
        ):  # precheck to prevent new annotation overlap with existing annotation
            if int(x_axis[0]) in range(
                j[0] - AnnotationSpacing, j[0] + AnnotationSpacing
            ):
                if int(y_axis[0]) in range(
                    j[1] - AnnotationSpacing, j[1] + AnnotationSpacing
                ):
                    decision = False
        if decision:  # Annotated
            ax.annotate3D(
                clustering_name[dataset],
                (x_axis[0], y_axis[0], z_axis[0]),
                xytext=XYText,
                textcoords=TextCoords,
                fontsize=AnnotationSize,
            )  # 'cluster'+str(dataset+1)
            location_annoted.append((int(x_axis[0]), int(y_axis[0]), int(z_axis[0])))

    ax.view_init(azim=-90, elev=-90)
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    # Remove grid
    ax.grid(False)
    ax.set_zticks([])
    figManager = plt.get_current_fig_manager()
    if hasattr(figManager, "window"):
        figManager.window.showMaximized()

    fig4.savefig(
        path.join(output_dir, output_name + "_4.tiff")
    )  # Merge Raw Data with Movement

    # Setup the plotting framework for subplot c. Specific Cluster Display
    fig5 = plt.figure(figsize=FIGSIZE)
    ax = fig5.add_subplot(111, projection="3d")
    # ax.set_xlim(0,550)
    # ax.set_ylim(0,550)
    # ax.set_zlim(0,zmax)
    # ax.set_ylim(ax.get_ylim()[::-1])
    ax.yaxis.set_ticks(np.arange(0, typixel * dimension_perpixel, SPACING))
    ax.xaxis.set_ticks(
        np.arange(0, txpixel * dimension_perpixel, SPACING)
    )  # 550 replace txpixel
    setattr(Axes3D, "annotate3D", _annotate3D)
    setattr(Axes3D, "arrow3D", _arrow3D)
    # ax.set_title(name+' Movement')
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_zlabel("z (µm)")
    fig5.tight_layout()
    for dataset in range(len(All_Cluster_Centers)):
        if dataset in highlightedClusters:  # only different
            plotzip = list(All_Cluster_Centers[dataset])
            x, y, z = zip(*plotzip)
            x_axis, y_axis, z_axis = list(x), list(y), list(z)
            x_axis = list(map(lambda x: x * dimension_perpixel, x_axis))
            y_axis = list(map(lambda x: x * dimension_perpixel, y_axis))
            z_axis = list(map(lambda x: x * dimension_perpixel, z_axis))

            for i in range(
                0, len(x_axis) - 1
            ):  # -1 is because two points result in one arrow
                ax.arrow3D(
                    x_axis[i],
                    y_axis[i],
                    z_axis[i],
                    x_axis[i + 1] - x_axis[i],
                    y_axis[i + 1] - y_axis[i],
                    z_axis[i + 1] - z_axis[i],
                    mutation_scale=ArrowSize,
                    arrowstyle=ArrowStyle,
                    color=ArrowColor,
                )  # ax.arrow3D(start of arrow, displacement of arrow, arrow setting)
            ax.annotate3D(
                clustering_name[dataset]
                + " "
                + str(round(dic_speed[clustering_name[dataset]], 4))
                + "µm/s "
                + str(round(dic_distance[clustering_name[dataset]], 4))
                + "µm",
                (x_axis[0], y_axis[0], z_axis[0]),
                xytext=XYText,
                textcoords=TextCoords,
                fontsize=AnnotationSize,
            )  # 'cluster'+str(dataset+1)
    for w in range(len(Filtered_Cluster_Coord)):
        plotzip = list(Filtered_Cluster_Coord[w])
        x, y, z = zip(*plotzip)
        x_axis, y_axis, z_axis = list(x), list(y), list(z)
        x_axis = list(map(lambda x: x * dimension_perpixel, x_axis))
        y_axis = list(map(lambda x: x * dimension_perpixel, y_axis))
        z_axis = list(map(lambda x: x * dimension_perpixel, z_axis))

        ax.scatter3D(x_axis, y_axis, z_axis, c=colorcode[w], s=DOTSIZE, alpha=ALPHA)

    ax.view_init(azim=-90, elev=-90)
    # Remove colored axes planes
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    # Remove grid
    ax.grid(False)
    ax.set_zticks([])
    figManager = plt.get_current_fig_manager()
    if hasattr(figManager, "window"):
        figManager.window.showMaximized()

    fig5.savefig(path.join(output_dir, output_name + "_5.tiff"))  # Specific Cluster


def Spectrum(lst_of_distance, lst_of_speed, output_dir, output_name):
    # Speed Spectrum
    fig6 = plt.figure(figsize=FIGSIZE)
    Label = "MoveA " + output_name
    df_distance = pd.DataFrame(np.array(lst_of_speed), columns=[Label])
    # print(df_distance)
    ax = sns.kdeplot(data=df_distance)
    ax.set(xlabel="Speed (µm/s)", ylabel="Density")
    # plt.legend();
    plt.title("Speed Spectrum")
    figManager = plt.get_current_fig_manager()
    if hasattr(figManager, "window"):
        figManager.window.showMaximized()

    fig6.savefig(path.join(output_dir, output_name + "_6.tiff"))  # Fig6 Speed Spectrum

    # Displacement Spectrum
    fig7 = plt.figure(figsize=FIGSIZE)
    Label = "MoveA " + output_name
    df_distance = pd.DataFrame(np.array(lst_of_distance), columns=[Label])
    ax = sns.kdeplot(data=df_distance)
    ax.set(xlabel="Displacement (µm)", ylabel="Density")
    # plt.legend();
    plt.title("Displacement Spectrum")
    figManager = plt.get_current_fig_manager()
    if hasattr(figManager, "window"):
        figManager.window.showMaximized()

    fig7.savefig(
        path.join(output_dir, output_name + "_7.tiff")
    )  # Fig7 Displacement Spectrum
