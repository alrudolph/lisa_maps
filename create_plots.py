import libpysal as lps
from esda import Moran_Local, Moran, fdr
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import mapclassify as mc
import numpy as np
from tqdm import tqdm

def group_data(data, cols, date_col, group_col, by='week'):
    """
    Groups data by date to iterate over later. 
    
    Parameters
    ----------
    data : dataframe
        Must include a date and grouping column along with variables of interest
    cols : str | List[str]
        Column names for the variables of interest
    date_col : str
        Column name containing dates
    group_col : str
        Column name of grouping, i.e. fips code
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
    # dates = [dt.strptime(i, date_format).date() for i in data[date_col]]

    if by == 'week':
        year_week = [get_year_week(date) for date in data[date_col]]

        # aggregate the cols by mean, also get start (min) and end (max) dates
        agg_dict = dict(zip([*cols, "date_start", "date_end"], [*np.repeat("mean", len(cols)), "min", "max"]))

        grouped = data \
            .assign(
                week_year = year_week,
                date_start = data[date_col],
                date_end = data[date_col]
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
            .groupby([date_col])

        # In case there are multiple values for each day:
        # grouped = data \
        #    .assign(**{date_col: dates}) \
        #    .groupby([date_col, group_col])[cols].mean() \
        #    .groupby([date_col])

    return grouped

#
#
#
#   MAPPING FUNCTIONS
#
#
#

#
#
#
#   HEAT MAPS:
#
#
#

def create_heat_map(grouped, cols, date_col, group_col, map, map_group_col, folder, by='week', stack = False, limit=None, sig = 0.05):
    
    if not isinstance(cols, list):
        cols = [cols]

    num_itrs = min(limit, len(grouped)) if limit else len(grouped)
    for j, col in enumerate(cols):

        if stack: # reset plot
            f, ax = plt.subplots(figsize=(9, 9))
            f.set_facecolor("white")

        for i, (date, group) in tqdm(enumerate(grouped), total=num_itrs):
            if limit and i * len(cols) + j == limit:
                break

            if i == 0:
                first_date = min(group[date_col[0]]) if len(date_col) > 0 else min(group[date_col])

            # it's important we merge on map so grouping order is consistent
            ordered = pd.merge(map, group, left_on=map_group_col, right_on=group_col)

            W = lps.weights.Queen(ordered['geometry'])
            W.transform = 'r' # pysal.org/libpysal/generated/libpysal.weights.W.html#libpysal.weights.W.set_transform
            quadrants = moran_quadrants(ordered[col], W, sig)

            if not stack: # reset plot
                f, ax = plt.subplots(figsize=(9, 9))
                f.set_facecolor("white")
            
            # the labels have to be sorted into the same order as the hcmap, you can leave the legends as the numbers
            # by changing cl=quadrants, but I don't know how to get the colors right without using numbers
            labels = ["0. Not Significant", "1. Hot Spot", "2. Cold Spot"]

            hcmap = get_heat_map_colors(stack, i, num_itrs)

            # We want all the labels in the legend:
            #  * Append blank rows to the map
            #  * Assign the correct labels to rows with information, And all labels to the blank rows
            map \
                .append([{k: None for k in map.columns} for _ in range(len(labels))], ignore_index=True) \
                .assign(cl = [*[labels[q] for q in quadrants], *labels]) \
                .plot(column='cl', categorical=True, k=2, cmap=hcmap, ax=ax, legend=True, edgecolor='black', linewidth=0.3)

            if not stack:

                if isinstance(date_col, list):
                    date = [min(ordered[date_col[0]]), max(ordered[date_col[1]])]
                    plot_title = f"{col.title().replace('_', ' ')} From {date[0]} to {date[1]}"
                else:
                    date = min(ordered[date_col])
                    plot_title = f"{col.title().replace('_', ' ')}, {date}"

                path=f"{folder}/{by}{i}_{col}_heat.png"
                ax.set_axis_off()
                ax.set_title(plot_title)
                plt.savefig(path)

        if stack:
            if isinstance(date_col, list):
                date = [first_date, max(ordered[date_col[1]])]
                plot_title = f"{col.title().replace('_', ' ')} From {date[0]} to {date[1]}"
            else:
                date = min(ordered[date_col])
                plot_title = f"{col.title().replace('_', ' ')}, {date}"

            plot_title = f"{col.title().replace('_', ' ')} From {date[0]} to {date[1]}"
            path=f"{folder}/stacked_{col}_heat.png"
            ax.set_axis_off()
            ax.set_title(plot_title)
            plt.savefig(path)

#
#
#
#   QUANTILE MAPS:
#
#
#

def create_quantile_map(grouped, cols, date_col, group_col, map, map_group_col, folder, by='week', limit=None, sig = 0.05, k=10):
    if not isinstance(cols, list):
        cols = [cols]
    
    num_itrs = min(limit, len(grouped)) if limit else len(grouped)
    count = 0
    for i, (date, group) in tqdm(enumerate(grouped), total=num_itrs):
        # merge to correct order
        ordered = pd.merge(map, group, left_on=map_group_col, right_on=group_col)

        for col in cols:
            quantiles = mc.Quantiles(ordered[col], k=k)

            if limit and count == limit: 
                break
            count += 1

            f, ax = plt.subplots(figsize=(9, 9))
            f.set_facecolor("white") # plot backgroundcolor, I like navajowhite

            bins = [0] + sorted(np.unique(quantiles.bins))
            bins = [str(int(bins[i - 1])) + "-" + str(int(bins[i])) for i in range(1, len(bins))]

            map \
                .assign(cl = [bins[i] for i in quantiles.yb]) \
                .plot(column='cl', categorical=True, k=k, cmap='OrRd', linewidth=0.1, ax=ax, edgecolor='black', legend=True)

            if isinstance(date_col, list):
                date = [min(ordered[date_col[0]]), max(ordered[date_col[1]])]
                plot_title = f"{col.title().replace('_', ' ')} From {date[0]} to {date[1]}"
            else:
                date = min(ordered[date_col])
                plot_title = f"{col.title().replace('_', ' ')}, {date}"

            ax.set_axis_off()
            plt.title(plot_title)
            plt.savefig(f"{folder}/quantile_{col}_{i}_k{k}.png")


# ################################# #
#                                   #
#                                   #
# ##### END OF MAIN FUNCTIONS ##### #
#                                   #
#                                   #
# ################################# #

#
#
#
#   UTILITY FUNCTIONS:
#
#
#

def get_heat_map_colors(stack, stack_idx, stack_total):
    """
    Returns map colors

    Parameters
    ----------
    stack : boolean
        Whether to get colors for stacked graph
    stack_idx : num,
        Current index of stack iteration, required if stack == True
    stack_total : num,
        Total number of stack iterations, required if stack == True
    Return
    ------
    colors.ListedColormap
        color values
    """
    if stack:
        # Generally, it seems that we don't color values too close to 0 or 255,
        # and we want a low opacity

        # scale_p : controls the other two rgb values for hot-hot, cold-cold
        scale = 1 - stack_idx/(stack_total-1) * 4/9

        # map colors: not significant, hot spot, cold spotd
        map_colors = [
            f'#ffffff00', # important that opacity is set to zero
            colors.hsv_to_rgb([0, scale, 1]),  # hot-hot, red color
            colors.hsv_to_rgb([0.767, scale, 1])  # cold-cold, blue color
        ]

    else:
        # map colors: not significant, hot spot, cold spot
        map_colors = ['white', 'red', 'blue']

    return colors.ListedColormap(map_colors)

def moran_quadrants(col, W, alpha):
    local_moran = Moran_Local(col, W)

    ps = local_moran.p_sim
    qs = local_moran.q
    f = fdr(ps, alpha)

    return [
        (
            int((qs[i] + 1) / 2)    # hot-hot = 1, cold-cold = 2
            if ps[i] <= f and qs[i] in [1, 3]  # only significant hot-hot and cold-cold
            else 0
        ) 
        for i in range(0, len(col))
    ]

def get_year_week(date):
    """
    Gets the week number of date (starting on monday).

    Parameters
    ----------
    date : date
        Date to convert

    Returns
    -------
    tuple
        A tuple of (year, week_number)
    """
    week_num = date.isocalendar()[1]
    year = date.year

    # The week number can wrap to the next year, need to ensure year does as well.
    # Don't need to worry about wrapping other way
    if week_num == 1 and date.month == 12:
        year += 1

    # Year comes first so dataframe is sorted chronologically
    return (year, week_num)

#
#
#
#   EXPORT FUNCTIONS:
#
#
#

#
#
#
#   QUANTILE MAPS:
#
#
#

def export_quantile_vals(grouped, cols, date_col, group_col, map, map_group_col, k=10, limit=None):
    """
    Calculate quantiles for each group

    Parameters
    ----------
    grouped : dataframe
        Result of grouped_data()
    cols : str | list[str]
        Column names for variables of interest
    date_col: str
        Column name for date column
    group_col: str
        Column name for grouping column (eg fips code)
    map: dataframe
        Geopandas dataframe containing 'geometry' column
    map_group_col: str
        Column name for map dataframe grouping column (eg fips)
    k: num, 10 be default
        Number of quantiles
    limit: num, 1 by default
        Limit iterations

    Return
    ------
    (dataframe, list)
        Tuple of: dataframe containing quantile and date; list of group ordering
    """
    if not isinstance(cols, list):
        cols = [cols]
    
    output = pd.DataFrame(columns=[*cols, 'date', *[i + "_bin" for i in cols]])

    for i, (date, group) in tqdm(enumerate(grouped), total=min(limit, len(grouped)) if limit else len(grouped)):
        
        if limit and i == limit:
            break

        # merge to correct order
        ordered = pd.merge(map, group, left_on=map_group_col, right_on=group_col)

        row = {
            'date': [min(group['date_start']), max(group['date_end'])] if ('date_start' in ordered.columns) else min(group[date_col])
        }

        for col in cols:
            temp = mc.Quantiles(ordered[col], k=k)
            row[col + "_bin"] = temp.bins
            row[col] = temp.yb

        output = output.append(row, ignore_index = True)

    return (output, map[map_group_col])

def create_quantile_map_from_export(export, cols, date_col, map, folder, k=10, limit=None):
    """
    Plot quantile map from export

    Parameters
    ----------
    export : dataframe
        result of export_quantile_vals
    cols : str | list[str]
        Column names of variable of interest
    date_col : str
        Name of date column
    map : dataframe
        Geopandas dataframe containing column 'geometry'
    folder : str
        output folder location
    k : num, 10 by default
        Number of quantiles
    limit : num, optional
        Limit iterations
    
    """

    if not isinstance(cols, list):
        cols = [cols]

    count = 0
    for col in cols: 

        num_itrs = min(limit, len(export[col])) if limit else len(export[col])
        for i, quantiles in tqdm(enumerate(export[col]), total=num_itrs):

            if limit and count == limit: 
                break
            count += 1

            f, ax = plt.subplots(figsize=(9, 9))
            f.set_facecolor("white") # plot backgroundcolor, I like navajowhite

            bins = [0] + sorted(np.unique(export[col + "_bin"][i]))
            bins = [str(int(bins[i - 1])) + "-" + str(int(bins[i])) for i in range(1, len(bins))]

            map \
                .assign(cl = [bins[i] for i in quantiles]) \
                .plot(column='cl', categorical=True, k=k, cmap='OrRd', linewidth=0.1, ax=ax, edgecolor='black', legend=True)

            date = export[date_col][i]
            if isinstance(date, list):
                plot_title = f"{col.title().replace('_', ' ')} From {date[0]} to {date[1]}"
            else:
                plot_title = f"{col.title().replace('_', ' ')}, {date}"

            ax.set_axis_off()
            plt.title(plot_title)
            plt.savefig(f"{folder}/quantile_{col}_{i}_k{k}.png")

#
#
#
#   HEAT MAP:
#
#
#

def create_heat_map_from_export(export, cols, date_col, map, folder, color_change="color", by="week", stack = False, limit=None):
    """
    Creates heat maps.

    Parameters
    ----------
    export : dataframe
        Output from export_head_vals
    cols : str | list[str]
        Names of variables of interest
    date_col : str
        Name of date column of export dataframe
    map : dataframe
        Geopandas dataframe contianing 'geometry' column
    folder : str
        Output folder name
    color_change : "opacity" | "color", optional
        For a stacked graph, what to vary of iterations
    by : str, week by default
        Used for file naming
    incl_dd : boolean, False by default
        Whether to include hot-cold and cold-hot regions
    stack : boolean, False by default
        Whether to stack plots
    limit : optional
        Limit number of iterations
    """

    if not isinstance(cols, list):
        cols = [cols]

    count = 0
    for col in cols: 

        if stack:
            f, ax = plt.subplots(figsize=(9, 9))
            f.set_facecolor("white") # plot backgroundcolor, I like navajowhite

        num_itrs = min(limit, len(export[col])) if limit else len(export[col])
        for i, quadrants in tqdm(enumerate(export[col]), total=num_itrs):

            if limit and count - 1 == limit: 
                break
            count += 1

            if not stack: # reset plot
                f, ax = plt.subplots(figsize=(9, 9))
                f.set_facecolor("white")
            
            # the labels have to be sorted into the same order as the hcmap, you can leave the legends as the numbers
            # by changing cl=quadrants, but I don't know how to get the colors right without using numbers
            labels = ["0. Not Significant", "1. Hot Spot", "2. Cold Spot"]

            hcmap = get_heat_map_colors(stack, i, num_itrs, color_change)

            # We want all the labels in the legend:
            #  * Append blank rows to the map
            #  * Assign the correct labels to rows with information, And all labels to the blank rows
            map \
                .append([{k: None for k in map.columns} for _ in range(len(labels))], ignore_index=True) \
                .assign(cl = [*[labels[q] for q in quadrants], *labels]) \
                .plot(column='cl', categorical=True, k=2, cmap=hcmap, ax=ax, legend=True, edgecolor='black', linewidth=0.3)

            if not stack:
                date = export[date_col][i]

                if isinstance(date, list):
                    plot_title = f"{col.title().replace('_', ' ')} From {date[0]} to {date[1]}"
                else:
                    plot_title = f"{col.title().replace('_', ' ')}, {date}"

                path=f"{folder}/{by}{i}_{col}_heat.png"
                ax.set_axis_off()
                ax.set_title(plot_title)
                plt.savefig(path)

        if stack:
            date = [export[date_col][0], export[date_col][num_itrs - 1]]

            if isinstance(export[date_col][0], list):
                date = [date[0][0], date[1][1]]

            plot_title = f"{col.title().replace('_', ' ')} From {date[0]} to {date[1]}"
            path=f"{folder}/stacked_{col}_heat.png"
            ax.set_axis_off()
            ax.set_title(plot_title)
            plt.savefig(path)

def export_heat_vals(grouped, cols, date_col, group_col, map, map_group_col, limit=None, sig = 0.05):
    """
    Calculate map quadrants for each grouping

    Parameters
    ----------
    grouped : dataframe
        Result of group_data()
    cols : str | list[str]
        Column names for variables of interest
    date_col : str
        Name of grouped date column
    group_col : str
        Name of grouping (eg fips_code) column of data
    map : dataframe
        Geopandas dataframe with 'geometry' column
    map_group_col : str
        Coluumn of map dataframe with grouping (fips code) to join with grouped date
    limit : num, optional
        Limit number of iterations
    sig : num, 0.05 by default
        Significance level to perform local moran's I test at
    Return
    ------
    (dataframe, list)
        Tuple of: Dateframe containing quadrants, dates, and global moran test; list of fip codes in order
    """
    if not isinstance(cols, list):
        cols = [cols]

    output = pd.DataFrame(columns=[*cols, 'date', *[i + "_g" for i in cols]])

    for i, (date, group) in tqdm(enumerate(grouped), total=min(limit, len(grouped)) if limit else len(grouped)):

        if limit and i == limit:
            break

        # it's important we merge on map so grouping order is consistent
        ordered = pd.merge(map, group, left_on=map_group_col, right_on=group_col)
        
        row = {
            'date': [min(group['date_start']), max(group['date_end'])] if ('date_start' in ordered.columns) else min(group[date_col])
        }

        W = lps.weights.Queen(ordered['geometry'])
        W.transform = 'r' # pysal.org/libpysal/generated/libpysal.weights.W.html#libpysal.weights.W.set_transform

        for j, col in enumerate(cols):
            # Local Moran quadrants
            row[col] = moran_quadrants(ordered[col], W, sig)

            # Global Moran
            mi = Moran(ordered[col], W, transformation='r', two_tailed=False)
            row[col + "_g"] = (mi.I, mi.p_norm)

        output = output.append(row, ignore_index=True)

    return (output, map[map_group_col])
