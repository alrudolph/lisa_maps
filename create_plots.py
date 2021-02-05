import libpysal as lps
from esda import Moran_Local, Moran
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

def create_heat_map(grouped, cols, date_col, group_col, map, map_group_col, folder, color_change, date_format='%Y-%m-%d', by='week', incl_dd = False, stack = False, limit=None, sig = 0.05):
    export = export_heat_vals(grouped, cols, date_col, group_col, map, map_group_col, date_format, limit, sig)
    create_heat_map_from_export(grouped, export, cols, grouped[date_col], map, folder, color_change, date_format, by, incl_dd, stack, limit, sig)

#
#
#
#   QUANTILE MAPS:
#
#
#

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
            opac = get_color_scale(100, 180, stack_idx, stack_total)
        elif color_change == "color":
            scale_p = get_color_scale(90, 160, stack_idx, stack_total)
            scale_s1 = get_color_scale(160, 220, stack_idx, stack_total)
            scale_s2 = get_color_scale(160, 255, stack_idx, stack_total)
            opac = get_color_scale(60, 60, 0, 1) # always 60 

        # map colors: not significant, hot spot, doughnut, cold spot, diamond
        map_colors = [
            f'#ffffff00', # important that opacity is set to zero
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

def create_heat_map_from_export(export, cols, date_col, map, folder, color_change="color", by="week", incl_dd = False, stack = False, limit=None):
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

            if not incl_dd: 
                quadrants = quadrants * (quadrants % 2 != 0) # remove doughnut and diamond

            if not stack: # reset plot
                f, ax = plt.subplots(figsize=(9, 9))
                f.set_facecolor("white")
            
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
            local_moran = Moran_Local(ordered[col], W)
            row[col] = local_moran.q * (local_moran.p_sim < sig) # significant quadrant values

            # Global Moran
            mi = Moran(ordered[col], W, transformation='r', two_tailed=False)
            row[col + "_g"] = (mi.I, mi.p_norm)

        output = output.append(row, ignore_index=True)

    return (output, map[map_group_col])
