import matplotlib.pyplot as plt
from matplotlib import colors
import math
import numpy as np
from .plotting_helpers import *
import pdb

def convert_to_heatmap_matrix(horizontal, vertical, *vals):
    """Given n arrays of the same length, the first 2 representing the keys of the horizontal and the vertical axis 
    and the rest each represents the corresponding values, it returns the axis suitable for direct use in heatmap 
    generation and a 2D array representation for each of the vals"""

    # Extract Unique values in the 2 key arrays.
    uniq_h = list(np.unique(horizontal))
    uniq_v = list(dict.fromkeys(vertical))
    
    # Prepare the empty 2D Containers for each of the results.
    results = []
    for _ in range(len(vals)):
        results.append( [ [0]*len(uniq_h) for _ in range(len(uniq_v))] )
        # results.append(np.empty((len(uniq_v), len(uniq_h))))

    for i, (key_h, key_v) in enumerate(zip(horizontal, vertical)):
        ind_h = uniq_h.index(key_h)
        ind_v = uniq_v.index(key_v)
        for j, val in enumerate(vals):
            results[j][ind_v][ind_h] = val[i]
    
    return uniq_h, uniq_v, *results


def draw_heatmap(
    h_coords,
    v_coords,
    vals,
    labels = None,
    share_cbar = True, #TODO: Implement False functionality of that.
    middle_point = 0., #TODO: Implement functionality when that is None.
    nrows = 1,
    figsize = (8,10),
    cmap = 'RdBu_r',
    h_titles = [],
    v_titles = [],
    h_label = '',
    v_label = '',
    cbar_step = 1,
    fn_cbar_label_mapper = default_mapper,
):
    """Adds a heatmap that has many subfigures with nrows rows and the number of columns needed to fit all items in data.
    data should have an object per heatmap to draw. You need to create a new figure if you don't want the heatmap to be drawn
    on the current figure. It won't reset the figure for you. This assumes that all subplots share the coordinates."""

    ncols = math.ceil(len(vals) / nrows)
    fig, ax = plt.subplots(nrows=nrows,ncols=ncols, sharex=True, sharey=True, figsize=figsize)


    # Extract Minimum and Maximum values
    min_val = float('inf')
    max_val = float('-inf')
    for item in vals:
        min_val = min(min_val, min([x for y in item for x in y]))
        max_val = max(max_val, max([x for y in item for x in y]))

    # Round to the next int.
    min_val = math.floor(min_val)
    max_val = math.ceil(max_val)

    # Create the colorbar ranges
    divnorm=colors.TwoSlopeNorm(vmin=min_val, vcenter=middle_point, vmax=max_val)

    # Plot each set of data in one of the subplots.
    ax_cnt = 0
    for ind, result in enumerate(vals):
        # Get the correct x,y indices of the subplot to use.
        ax_x, ax_y = get_x_y_from_ax_cnt(ax_cnt, ncols)
        # Increment for next use.
        ax_cnt+= 1
        # Draw the actual heatmap.
        im = ax[ax_x, ax_y].imshow(result, cmap=cmap, norm=divnorm)
        # Set the ticks in the axis.
        ax[ax_x, ax_y].set_xticks(range(len(h_coords)))
        ax[ax_x, ax_y].set_yticks(range(len(v_coords)))
        # Set the labels in the axis.
        ax[ax_x, ax_y].set_xticklabels(h_coords)
        ax[ax_x, ax_y].set_yticklabels(v_coords)
        # If labels are provided 
        if labels:
            for h in range(len(h_coords)):
                for v in range(len(v_coords)):
                    ax[ax_x, ax_y].text(h, v, labels[ind][v][h], ha='center', va='center', color='black')
            
        # Rotate the x-axis ticks.
        plt.setp(ax[ax_x, ax_y].get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    # Set titles of the horizontal subplots. They're only set for the top drawings.
    for ax_i, col in zip(ax[0], h_titles):
        ax_i.set_title(f'{col}')
    
    # Set titles for the vertical subplots. They're only set for the rightmost drawings.
    for ax_i, row in zip(ax[:,-1], v_titles):
        ax_i.yaxis.set_label_position("right")
        ax_i.set_ylabel(f'{row}', size='large')
    
    # Set the global labels of the horizontal and vertical
    fig.supxlabel(h_label)
    fig.supylabel(v_label)

    plt.tight_layout()

    # Draw the unified colorbar according to the ranges specified.
    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), ticks=list(range(min_val, max_val+cbar_step, cbar_step)), cmap=cmap, norm=divnorm)
    cbar.ax.set_yticklabels(fn_cbar_label_mapper(list(range(min_val, max_val+cbar_step, cbar_step))))