def get_x_y_from_ax_cnt(ax_cnt, ncols):
    return ax_cnt // ncols, ax_cnt % ncols

def default_mapper(val):
    return val