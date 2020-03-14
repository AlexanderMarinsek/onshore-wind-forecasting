from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import csv


# Figure format
extension = "png"

# Combine default plot style (v2.0) with custom style
plt.style.use(["default", "./custom1.mplstyle"])


def plot_data_violin( data, names ):

    colors = [ 'C0', 'C1', 'C9', 'C3', 'C6' ]
    figname = 'Violin-data_%s' % '-'.join(names)
    plot_violin( data, names, colors, figname, 'Source', r'Wind speed $(\frac{m}{s})$' )


def plot_mse_violin( mse, names ):

    colors = [ 'C1', 'C9', 'C3', 'C6' ]
    figname = 'Violin-MSE_%s' % '-'.join(names)
    plot_violin( mse, names, colors, figname, 'Source', r'MSE $(\frac{m^2}{s^2})$' )


def plot_violin( data, names, colors, figname, xlabel, ylabel ):

    sns.set(style="whitegrid")

    ax = sns.violinplot( data=data, palette=colors )

    ax.set_xticklabels( names )

    ax.set_xlabel( xlabel )
    ax.set_ylabel( ylabel )

    plt.savefig("%s.%s" % (figname, extension))
    plt.show()


def calc_mse( data ):

    (m, n) = data.shape
    mse = np.zeros((m,n-1), dtype=float) # Omit reanalysis data in error array

    for i in range(0, m):
        for k in range(0, n-1):
            mse[i, k] = ( data[i, 0] - data[i, k+1] )**2 # Calc error for each forecast

    return mse


def get_data_and_names_from_file( filename ):

    with open( filename, 'r' ) as fr:
        reader = csv.reader( fr, delimiter=',' )

        # Read first row to get number of columns, and generate names list
        row = next(reader)
        num_of_data = int(len(row) / 2)
        names = [ row[k*2+1] for k in range( 0, num_of_data ) ]
        # Extract only last part of names ('forecast-BL' -> 'BL') and compose new list
        names = [ name.split('-')[1] for name in names ]
        names.insert(0, 'ERA5-Land')

        # +1 to include reanalysis data
        data = np.zeros( ( 1, num_of_data+1 ), dtype=float )

        for i, row in enumerate(reader):

            # Real reanalysis value
            reduced_row = [ float(row[0]) ]
            # Add forecasts and convert to ndarray array
            reduced_row += [ float(row[k*2+1]) for k in range( 0, num_of_data ) ]
            # Convert to float
            reduced_row = list(map(float, reduced_row))

            data = np.concatenate( ( data, np.array([reduced_row]) ), axis=0 )

        data = np.delete( data, 0, 0 )

    return [ data, names ]


if __name__ == '__main__':


    import os

    # Get comparison directory path
    dirpath = os.path.join( os.path.dirname(os.path.abspath(__file__)), 'comparison-data' )
    # Get all comparison filepaths
    filenames = [ os.path.join( dirpath, f ) for f in os. listdir(dirpath) ]

    data = []

    for filename in filenames:
        [ tmp_data, tmp_names ] = get_data_and_names_from_file( filename )
        if data == []:
            data = tmp_data
            names = tmp_names
        else:
            data = np.concatenate(( data, tmp_data ), axis=0)

    mse = calc_mse( data )

    plot_data_violin( data, names )
    plot_mse_violin( mse, names[1:] )
