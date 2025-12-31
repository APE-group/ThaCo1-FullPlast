import matplotlib as mpl


def set_ticks(size, scale_size, width, scale_width):

    mpl.rcParams['xtick.major.size'] = size
    mpl.rcParams['xtick.major.width'] = width
    mpl.rcParams['xtick.minor.size'] = size/scale_size
    mpl.rcParams['xtick.minor.width'] = width/scale_width
    mpl.rcParams['ytick.major.size'] = size
    mpl.rcParams['ytick.major.width'] = width
    mpl.rcParams['ytick.minor.size'] = size/scale_size
    mpl.rcParams['ytick.minor.width'] = width/scale_width
