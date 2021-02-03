import libpysal as lps
from esda import Moran_Local
import matplotlib.pyplot as plt
from matplotlib import colors
# import geopandas ??
from datetime import datetime as dt
import pandas as pd

# return (year, weeknum)
# I use this to get the sorting order to be chronological
def get_year_week(date, date_format):
    curr_date = dt.strptime(date, date_format)
    return (curr_date.year, curr_date.isocalendar()[1])

# returns 0: insignificant, else quandrant location
def get_moran_class(moran, sig):
    return moran.q * (moran.p_sim < sig)

def get_map_coloring(moran_class, incl_dd):
    if incl_dd:
        coloring = moran_class
    else:
        coloring = moran_class * (moran_class % 2 != 0) # remove doughnut and diamond
    return coloring

def export_plot(map, coloring, plot_title, path):
    # map colors: not significant, hot spot, doughnut, cold spot, diamond

    #navajowhite
    hcmap = colors.ListedColormap(['lightgrey', 'red', 'lightblue', 'blue', 'pink'])
    labels = ["Not Significant", "Hot Spot", "Doughnut", "Cold Spot", "Diamond"]

    f, ax = plt.subplots(1, figsize=(9, 9))

    # [labels[i] for i in coloring]
    map \
        .assign(cl = coloring) \
        .plot(column='cl', categorical=True, k=2, cmap=hcmap, linewidth=0.1, ax=ax, edgecolor='black', legend=True)

    ax.set_axis_off()
    plt.title(plot_title)
    plt.savefig(path)

def single_col_calc(data_col, map, W, incl_dd, plot_title, path, sig):
    local_moran = Moran_Local(data_col, W)
    moran_class = get_moran_class(local_moran, sig)

    if path:
        coloring = get_map_coloring(moran_class, incl_dd)
        export_plot(map, coloring, plot_title, path)

    return moran_class

def create_plots(data, map, cols, date_col='date_range_start', county_col='fips', by='week', incl_dd = True, sig = 0.05, limit=1):
    date_format = '%Y-%m-%d'

    # You can calculate the weights just once, if you have data for every county for each interval
    W = lps.weights.Queen(map['geometry'])
    W.transform = 'r' # pysal.org/libpysal/generated/libpysal.weights.W.html#libpysal.weights.W.set_transform

    plot_title='LOL'
    path='./test.png'

    if not isinstance(cols, list):
        cols = [cols]

    if by == 'week':
        grouped = data \
            .assign(week_year = [get_year_week(date, date_format) for date in data[date_col]]) \
            .groupby(['week_year', county_col])[cols].mean() \
            .groupby('week_year')
    elif by == 'day':
        grouped = data \
            .groupby([date_col, county_col])[cols].mean() \
            .groupby(date_col)

    # cycle through each date period
    for i, (date, group) in enumerate(grouped):
        if i == limit:
            break

        # go through each variable:
        for col in cols:
            single_col_calc(group[col], map, W, incl_dd, date, path, sig)
