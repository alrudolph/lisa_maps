import libpysal as lps
from esda import Moran_Local
import matplotlib.pyplot as plt
from matplotlib import colors
from datetime import datetime as dt
import pandas as pd
import mapclassify as mc

# return (year, weeknum)
# I use this to get the sorting order to be chronological
def get_year_week(date, date_format):
    curr_date = dt.strptime(date, date_format)
    return (curr_date.year, curr_date.isocalendar()[1])

# returns 0: insignificant, else quandrant location
def get_quadrant(moran, sig):
    return moran.q * (moran.p_sim < sig)

def get_quantiles(data_col, k):
    return mc.Quantiles(data_col, k=k)

def filter_quadrants(moran_class, incl_dd):
    if incl_dd:
        qc = moran_class
    else:
        qc = moran_class * (moran_class % 2 != 0) # remove doughnut and diamond
    return qc

def export_quantile_plot(map, quantiles, plot_title, path, k):
    f, ax = plt.subplots(1, figsize=(9, 9))

    map \
        .assign(cl = quantiles.yb) \
        .plot(column='cl', categorical=True, k=k, cmap='OrRd', linewidth=0.1, ax=ax, edgecolor='black', legend=True)

    ax.set_axis_off()
    plt.title(plot_title)
    plt.savefig(path)

def export_hotcold_plot(map, quadrants, plot_title, path):
    # map colors: not significant, hot spot, doughnut, cold spot, diamond

    hcmap = colors.ListedColormap(['navajowhite', 'red', 'lightblue', 'blue', 'pink'])

    # the labels have to be sorted into the same order as the hcmap, you can leave the legends as the numbers
    # by changing cl=quadrants, but I don't know how to get the colors right without using numbers
    labels = ["0. Not Significant", "1. Hot Spot", "2. Doughnut", "3. Cold Spot", "4. Diamond"]

    f, ax = plt.subplots(1, figsize=(9, 9))

    map \
        .assign(cl = [labels[i] for i in quadrants]) \
        .plot(column='cl', categorical=True, k=2, cmap=hcmap, linewidth=0.1, ax=ax, edgecolor='black', legend=True)

    ax.set_axis_off()
    plt.title(plot_title)
    plt.savefig(path)

# calculate local moran and create plot on single variable
def single_col_calc(data_col, map, W, heat, incl_dd, plot_title, path, sig, k):
    local_moran = Moran_Local(data_col, W)
    quadrants = get_quadrant(local_moran, sig)

    if path:
        if heat:
            map_quadrants = filter_quadrants(quadrants, incl_dd)
            export_hotcold_plot(map, map_quadrants, plot_title, path)
        else:
            export_quantile_plot(map, get_quantiles(data_col, 10), plot_title, path, k)

    return quadrants

#
#   MAIN FUNCTION
#
def create_plots(data, map, cols, date_col, group_col, folder, date_format='%Y-%m-%d', by='week', heat = True, incl_dd = False, sig = 0.05, limit=None, k = 10):
    # You can calculate the weights just once, if you have data for every county for each interval
    W = lps.weights.Queen(map['geometry'])
    W.transform = 'r' # pysal.org/libpysal/generated/libpysal.weights.W.html#libpysal.weights.W.set_transform

    if not isinstance(cols, list):
        cols = [cols]

    if by == 'week':
        week_nums = [get_year_week(date, date_format) for date in data[date_col]]

        grouped = data \
            .assign(week_num = [wn[1] for wn in week_nums], year = [wn[0] for wn in week_nums]) \
            .groupby(["year", "week_num", group_col])[cols].mean() \
            .groupby(["year", "week_num"])
    elif by == 'day':
        grouped = data \
            .groupby([date_col, group_col])[cols].mean() \
            .groupby(date_col)

    # cycle through each date period
    for i, (date, group) in enumerate(grouped):
        if i == limit:
            break

        if by == 'week':
            plot_title = f"Week #{date[1]}, {date[0]}"
        elif by == 'day':
            plot_title = date

        # go through each variable:
        for col in cols:
            path = f"{folder}/{by}{i}_{col}_{'heat' if heat else 'quantile'}{'_dd' if heat and incl_dd else ''}.png" if folder else None

            print(path, plot_title)
            single_col_calc(group[col], map, W, heat, incl_dd, plot_title, path, sig, k)
