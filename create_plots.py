import libpysal as lps
from esda import Moran_Local
import matplotlib.pyplot as plt
from matplotlib import colors
from datetime import datetime as dt
import pandas as pd
import mapclassify as mc
import numpy as np
from tqdm import tqdm

def group_data(data, cols, date_col, group_col, date_format='%Y-%m-%d', by='week'):
    """
    Groups data by date to iterate over later. 
    
    Parameters
    ----------
    data : dataframe
        Must include a data and grouping column along with variables of interest
    cols : str | List[str]
        Column names for the variables of interest
    date_col : str
        Column name containing dates
    group_col : str
        Column name of grouping, i.e. fips code
    date_format : str, optional
        Format of the dates, default: YYYY-MM-DD
    by: 'week' | 'day', optional
        How to group data, default: week.

    Returns
    -------
    dataframe
        Dataframe grouped by date
    """
    if not isinstance(cols, list):
        cols = [cols]

    # convert string dates to date objs
    dates = [dt.strptime(i, date_format).date() for i in data[date_col]]

    if by == 'week':
        year_week = [get_year_week(date, date_format) for date in data[date_col]]

        # aggregate the cols by mean, also get start (min) and end (max) dates
        agg_dict = dict(zip([*cols, "date_start", "date_end"], [*np.repeat("mean", len(cols)), "min", "max"]))

        grouped = data \
            .assign(
                week_year = year_week,
                date_start = dates,
                date_end = dates
            ) \
            .groupby(["week_year", group_col]) \
            .agg(agg_dict) \
            .groupby(["week_year"])
        
        # Just get min date:
        # grouped = data \
        #     .assign(week_year = year_week, **{date_col: dates}) \
        #     .groupby(["week_year", group_col]) \
        #     .agg(dict(zip([*cols, date_col], [*np.repeat("mean", len(cols)), "min"]))) \
        #     .groupby(["week_year"])

        # Discard date, just year and week number:
        # grouped = data \
        #    .assign(**{date_col: year_week}) \
        #    .groupby([date_col, group_col])[cols].mean() \
        #    .groupby([date_col])
    
    elif by == 'day':
        grouped = data \
            .assign(**{date_col: dates}) \
            .groupby([date_col])

        # In case there are multiple values for each day:
        # grouped = data \
        #    .assign(**{date_col: dates}) \
        #    .groupby([date_col, group_col])[cols].mean() \
        #    .groupby([date_col])

    return grouped

def export_quantile_vals(grouped, cols, date_col, group_col, map, map_group_col, folder, date_format='%Y-%m-%d', by='week', incl_dd = False, k=10, limit=1, sig = 0.05):
    if not isinstance(cols, list):
        cols = [cols]

    for i, col in enumerate(cols):

        for j, (date, group) in enumerate(grouped):
            if i * len(cols) + j == limit:
                break

            ordered = pd.merge(map, group, left_on=map_group_col, right_on=group_col)
            quantiles = mc.Quantiles(ordered[col], k=k)

            plot_title = "plot_title"
            path = "./example/output1/quantile.png"

            export_quantile_plot(map, quantiles, plot_title, path, k)

    return 

def create_quantile_map(grouped, cols, date_col, group_col, map, map_group_col, folder, date_format='%Y-%m-%d', by='week', incl_dd = False, k=10, limit=1, sig = 0.05, stacked=False):

    quantiles = export_quantile_vals(grouped, cols, date_col, group_col, map, map_group_col, folder, date_format, by, incl_dd, k, limit, sig)

    #
    #
    #   DO STUFF HERE
    #

    f, ax = plt.subplots(1, figsize=(9, 9))

    map \
        .assign(cl = quantiles.yb) \
        .plot(column='cl', categorical=True, k=k, cmap='OrRd', linewidth=0.1, ax=ax, edgecolor='black', legend=True)

    ax.set_axis_off()
    plt.title(plot_title)
    plt.savefig(path)

def create_heat_map(grouped, cols, date_col, group_col, map, map_group_col, folder, color_change, date_format='%Y-%m-%d', by='week', incl_dd = False, stack = False, limit=None, sig = 0.05):
    export = export_heat_vals(grouped, cols, date_col, group_col, map, map_group_col, date_format, limit, sig)
    create_heat_map_from_export(grouped, export, cols, grouped[date_col], map, folder, color_change, date_format, by, incl_dd, stack, limit, sig)

#
#
#
#   HEAT MAP:
#
#
#


def create_heat_map_from_export(export, cols, date_col, map, folder, color_change, date_format='%Y-%m-%d', by='week', incl_dd = False, stack = False, limit=None, sig = 0.05):

    if not isinstance(cols, list):
        cols = [cols]

    count = 0
    for col in cols: 

        if stack:
            f, ax = plt.subplots(figsize=(9, 9))
            f.set_facecolor("navajowhite") # plot backgroundcolor, I like navajowhite

        num_itrs = min(limit, len(export[col])) if limit else len(export[col])
        for i, quadrants in tqdm(enumerate(export[col]), total=num_itrs):

            if limit and count - 1 == limit: 
                break
            count += 1

            if not incl_dd: 
                quadrants = quadrants * (quadrants % 2 != 0) # remove doughnut and diamond

            if not stack: 
                f, ax = plt.subplots(figsize=(9, 9)) # reset plot
                f.set_facecolor("navajowhite")
            
            # the labels have to be sorted into the same order as the hcmap, you can leave the legends as the numbers
            # by changing cl=quadrants, but I don't know how to get the colors right without using numbers
            labels = ["0. Not Significant", "1. Hot Spot", "2. Doughnut", f"{3 if incl_dd else 2}. Cold Spot", "4. Diamond"]
            filtered_labels = filter_dd(labels, incl_dd)

            hcmap = get_heat_map_colors(incl_dd, stack, i, num_itrs, color_change)

            # We want all the labels in the legend:
            #  * Append blank rows to the map
            #  * Assign the correct labels to rows with information, And all labels to the blank rows
            map \
                .append([{k: None for k in map.columns} for _ in range(len(filtered_labels))], ignore_index=True) \
                .assign(cl = [*[labels[q] for q in quadrants], *filtered_labels]) \
                .plot(column='cl', categorical=True, k=2, cmap=hcmap, ax=ax, legend=True, edgecolor='black', linewidth=0.3)

            if not stack:
                date = export[date_col][i]

                if isinstance(date, list):
                    plot_title = f"{col.title().replace('_', ' ')} From {date[0]} to {date[1]}"
                else:
                    plot_title = f"{col.title().replace('_', ' ')}, {date}"

                path=f"{folder}/{by}{i}_{col}_heat{'_dd' if incl_dd else ''}.png"
                ax.set_axis_off()
                ax.set_title(plot_title)
                plt.savefig(path)

        if stack:
            date = [export[date_col][0], export[date_col][num_itrs - 1]]

            if isinstance(export[date_col][0], list):
                date = [date[0][0], date[1][1]]

            plot_title = f"{col.title().replace('_', ' ')} From {date[0]} to {date[1]}"
            path=f"{folder}/stacked_{col}_heat{'_dd' if incl_dd else ''}.png"
            ax.set_axis_off()
            ax.set_title(plot_title)
            plt.savefig(path)

def export_heat_vals(grouped, cols, date_col, group_col, map, map_group_col, date_format='%Y-%m-%d', limit=None, sig = 0.05):
    """
    Calculate map quadrants for each grouping

    Parameters
    ----------
    grouped : dataframe

    cols : str | list[str]

    date_col : str

    group_col : str

    map : dataframe

    map_group_col : str

    date_format : str, optional

    limit : num, optional

    sig : num, optional
    
    Return
    ------
    (dataframe, list)
        Tuple of: Dateframe containing quadrants and dates, list of fip codes in order
    """
    if not isinstance(cols, list):
        cols = [cols]

    output = pd.DataFrame(columns=[*cols, 'date'])

    for i, (date, group) in tqdm(enumerate(grouped), total=min(limit, len(grouped)) if limit else len(grouped)):

        if limit and i == limit:
            break

        ordered = pd.merge(map, group, left_on=map_group_col, right_on=group_col)
        
        row = {
            'date': [min(group['date_start']), max(group['date_end'])] if ('date_start' in ordered.columns) else min(group[date_col])
        }

        W = lps.weights.Queen(ordered['geometry'])
        W.transform = 'r' # pysal.org/libpysal/generated/libpysal.weights.W.html#libpysal.weights.W.set_transform

        for j, col in enumerate(cols):
            local_moran = Moran_Local(ordered[col], W)
            row[group.columns[j]] = get_quadrant(local_moran, sig)

        output = output.append(row, ignore_index=True)

    return (output, ordered[map_group_col])


#
#
#
#   UTILITY FUNCTIONS:
#
#
#

def get_heat_map_colors(incl_dd, stack, stack_idx, stack_total, color_change):
    """
    Returns map colors

    Parameters
    ----------
    incl_dd : boolean
        Whether to include hot-cold and cold-hot regions
    stack : boolean
        Whether to get colors for stacked graph
    stack_idx : num,
        Current index of stack iteration, required if stack == True
    stack_total : num,
        Total number of stack iterations, required if stack == True
    color_change : "opacity" | "color"
        What to change over time
    Return
    ------
    colors.ListedColormap
        color values
    """
    if stack:
        # Generally, it seems that we don't color values too close to 0 or 255,
        # and we want a low opacity

        # scale_p : controls the other two rgb values for hot-hot, cold-cold
        # scale_s : controls the other two rgb values for hot-cold, cold-hot
        if color_change == "opacity":
            scale_p = "00"
            scale_s1 = "AF"
            scale_s2 = "2C"
            opac = get_color_scale(40, 80, stack_idx, stack_total)
        elif color_change == "color":
            scale_p = get_color_scale(90, 160, stack_idx, stack_total)
            scale_s1 = get_color_scale(160, 220, stack_idx, stack_total)
            scale_s2 = get_color_scale(160, 255, stack_idx, stack_total)
            opac = get_color_scale(59, 60, 0, stack_total) # always 60 / stack_total

        # map colors: not significant, hot spot, doughnut, cold spot, diamond
        map_colors = [
            f'#ffffffff', # important that opacity is set to zero
            f"#ff{scale_p}{scale_p}{opac}",  # hot-hot, red color
            f'#{scale_s2}{scale_s1}ff{opac}',  # cold-hot
            f"#{scale_p}{scale_p}ff{opac}",  # cold-cold, blue color
            f'#ff{scale_s1}{scale_s2}{opac}'   # hot-cold
        ]
    else:
        # map colors: not significant, hot spot, doughnut, cold spot, diamond
        map_colors = ['white', 'red', 'lightblue', 'blue', 'pink']

    return colors.ListedColormap(filter_dd(map_colors, incl_dd)) 

def get_color_scale(min, max, curr_val, max_val):
    """"
    Get hex color

    Parameters
    ----------
    min : num
        Minumum hex value (ending value)
    max : num
        Maximum hex value (starting vaule)
    curr_val : num
        Current index out of iterations
    max_val : num
        Total number of iterations

    Return
    ------
    str
        Hex code
    """
    output = hex(int(max - (curr_val + 1) / max_val * (max - min)))[2:]
    return output if len(str(output)) == 2 else '0' + output  # ensure length of 2

def get_year_week(date, date_format):
    """
    Gets the week number of date (starting on monday).

    Parameters
    ----------
    date : str
        Date to convert
    date_format : str
        Format of date

    Returns
    -------
    tuple
        A tuple of (year, week_number)
    """
    curr_date = dt.strptime(date, date_format)
    
    week_num = curr_date.isocalendar()[1]
    year = curr_date.year

    # The week number can wrap to the next year, need to ensure year does as well.
    # Don't need to worry about wrapping other way
    if week_num == 1 and curr_date.month == 12:
        year += 1

    # Year comes first so dataframe is sorted chronologically
    return (year, week_num)

def get_quadrant(moran, sig):
    """
    Returns the quadrant number for each group, 0 if not significant.

    Parameters
    ----------
    moran
        Result of Local moran test
    sig : number
        Significance level

    Returns
    -------
    list
        List of quadrant numbers 0-4
    """
    return moran.q * (moran.p_sim < sig)

def filter_dd(arr, incl_dd):
    """
    Filters out hot-cold and cold-hot regions

    Parameters
    ----------
    arr : list
        Length 5
    incl_dd : boolean
        Whether to include hot-cold and cold-hot regions

    Return
    ------
    list
        Filtered list
    """
    return arr if incl_dd else [arr[i] for i in [0, 1, 3]]

def create_plots(grouped, map, cols, date_col, group_col, folder, date_format='%Y-%m-%d', by='week', heat = True, incl_dd = False, sig = 0.05, limit=None, k = 10):

    # #############################################################################################################


    output = {k: [] for k in cols}

    if colored_vals:
        import json

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        with open("./counties.json", "w") as out_file:
            dumps = json.dumps(colored_vals, cls=NumpyEncoder)
            json.dump(dumps, out_file)