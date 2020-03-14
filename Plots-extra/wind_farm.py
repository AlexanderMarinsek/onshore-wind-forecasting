
import os
import conda

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib

from csv import reader
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import sin, cos


from mpl_toolkits.basemap import Basemap


extension = "png"

# Combine default plot style (v2.0) with custom style
plt.style.use(["default", "./custom1.mplstyle"])

# Figure format
extension = "png"

# Custom settings only for scatter plots (not suitable for new MPL style)
scatter_marker = ['_', '_', '_', '_']
scatter_color = ['C0', 'C9', 'C9', 'C3']
scatter_lw = 2.5
scatter_alpha = 0.6

def plot_wind_farm( figname ):

    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    resolution = float( 0.1 )
    area = [44.5, 28.5, 44.7, 28.7]  
    area_int = [int(round(1.0*a/resolution)) for a in area]
    
    [start_lon, stop_lon] = [area_int[1], area_int[3]]
    [start_lat, stop_lat] = [area_int[0], area_int[2]]

    lons = np.arange(area_int[1], area_int[3]+1, 1)
    lats = np.arange(area_int[0], area_int[2]+1, 1)

    m = Basemap( \
        projection='cyl', \
        llcrnrlon=(lons.min()-1) * resolution,
        urcrnrlon=(lons.max()+1) * resolution, \
        llcrnrlat=(lats.min()-1) * resolution,
        urcrnrlat=(lats.max()+1) * resolution, \
        resolution='i')
        
    #m.drawcoastlines()
    #m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels=2000, verbose=True)
    m.arcgisimage(service='World_Street_Map', xpixels=2000, verbose=True)
    
    print lons
    print (lons.min()-1) * resolution
    
    #mc = "#fafafa"
    mc = "C0"
    
    ax1.scatter(28.5, 44.5, s=450, c=mc, marker="1", alpha=0.9)
    ax1.scatter(28.5, 44.6, s=450, c=mc, marker="1", alpha=0.9)
    ax1.scatter(28.5, 44.7, s=450, c=mc, marker="1", alpha=0.9)
    ax1.scatter(28.6, 44.5, s=450, c=mc, marker="1", alpha=0.9)
    ax1.scatter(28.6, 44.6, s=450, c=mc, marker="1", alpha=0.9)
    ax1.scatter(28.6, 44.7, s=450, c=mc, marker="1", alpha=0.9)
    ax1.scatter(28.7, 44.5, s=450, c=mc, marker="1", alpha=0.9)
    ax1.scatter(28.7, 44.6, s=450, c=mc, marker="1", alpha=0.9)
    ax1.scatter(28.7, 44.7, s=450, c=mc, marker="1", alpha=0.9)
    
    ax1.scatter(28.6, 44.6, s=1250, c="None", marker="o", alpha=0.7, 
        edgecolors="C3", linewidths=2)

    ax1.set_xticks(lons*resolution)
    ax1.set_xlabel(r"Longitude ($\degree$)")
    ax1.set_yticks(lats*resolution)
    ax1.set_ylabel(r"Latitude ($\degree$)")
    ax1.tick_params(axis='both', which='major', top=True, labeltop=True, right=True, labelright=True)

    fig.savefig("%s.%s" % (figname, extension))
    # fig.show()
    plt.close()


def main():

    plot_wind_farm( "Wind-Farm-PUB" )


if __name__ == "__main__":
    main()