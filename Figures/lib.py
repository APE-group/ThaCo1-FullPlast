import sys, psutil
import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from itertools import combinations
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from matplotlib import cm
from scipy import signal
import csv, yaml, pickle, bisect

def mm_to_inch(size_in_mm):
    size_in_inch = size_in_mm / 25.4
    return size_in_inch

def inch_to_mm(size_in_inch):
    size_in_mm = size_in_inch * 25.4
    return size_in_mm

def gauss_kde(v, sigma):
    n = 8
    xx = np.linspace(-n*sigma, n*sigma, 2*n+1)
    gaussian = np.exp(-(xx/(sigma))**2/2)
    gaussian /= np.sum(gaussian)
    return np.convolve(v, gaussian, mode="same")

# MNIST - First attempt

def get_mnist_digits(mnist_path):
    
    mnist_digits = [[] for i in range(10)]
    
    with open(mnist_path, 'r') as csv_file:
        
        for i,data in enumerate(csv.reader(csv_file)):
            
            # The first column is the label
            label = int(data[0])
            
            # The rest of columns are pixels
            pixels = data[1:]
            
            # Make those columns into a array of 8-bits pixels
            # This array will be of 1D with length 784
            # The pixel intensity values are integers from 0 to 255
            pixels = np.array(pixels, dtype='uint8')
            
            # If changing from negative to positive
            # pixels = 255 - pixels
            
            # Reshape the array into 28 x 28 array (2-dimensional array)
            pixels = pixels.reshape((28, 28))
            
            mnist_digits[label].append(pixels)
    
    return mnist_digits

# MNIST - Second attempt

def load_mnist_digits(img_fn, lbl_fn):
    SIZE = 28
    img_fp = open(img_fn, 'rb')
    img_fp.seek(16)
    digits_img = np.fromfile(img_fp, dtype=np.uint8)
    img_fp.close()
    digits = np.split(digits_img, len(digits_img) / (SIZE * SIZE))
    digits = [np.array(cell).reshape(SIZE, SIZE) for cell in digits]
    digits = np.array(digits)
    lbl_fp = open(lbl_fn, 'rb')
    lbl_fp.seek(8)
    labels = np.fromfile(lbl_fp, dtype=np.uint8)
    lbl_fp.close()

    return digits, labels

def load_mnist_array(stage, load_path):
    print('Loading MNIST digits ... \n')
    # Load data.

    if stage == 'Training':
        digits, labels = load_mnist_digits(load_path + 'train-images-idx3-ubyte',
                                           load_path + 'train-labels-idx1-ubyte')
    if stage == 'Test':
        digits, labels = load_mnist_digits(load_path + 't10k-images-idx3-ubyte',
                                           load_path + 't10k-labels-idx1-ubyte')

    # Shuffle data
    rand = np.random.RandomState(10)
    shuffle = rand.permutation(len(digits))
    digits, labels = digits[shuffle], labels[shuffle]

    #print('\nDeskew images ... \n')
    #digits_deskewed = list(map(deskew, digits))

    return digits, labels

def load_cifar_digits(img_lbl_fn):
    SIZE = 32
    with open(img_lbl_fn, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    # d has shape (10000, SIZE*SIZE*3)
    images = d[b'data']
    images = np.array([_.reshape(3,SIZE,SIZE).transpose(1,2,0) for _ in images])
    labels = d[b'labels']
    
    return images, labels

def load_cifar_array(stage, load_path):
    print('Loading CIFAR10 images ... \n')
    # Load data.

    if stage == 'Training':
        images, labels = load_cifar_digits(load_path + 'data_batch_1')
        for i in range(2,6):
            im, lb = load_cifar_digits(load_path + 'data_batch_' + str(i))
            images = np.append(images, im, axis=0)
            labels = np.append(labels, lb, axis=0)

    if stage == 'Test':
        images, labels = load_cifar_digits(load_path + 'test_batch')

    # Shuffle data
    #rand = np.random.RandomState(10)
    #shuffle = rand.permutation(len(digits))
    #digits, labels = digits[shuffle], labels[shuffle]

    #print('\nDeskew images ... \n')
    #digits_deskewed = list(map(deskew, digits))

    return images, labels

def deskew(img):
    SIZE = 28
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SIZE * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SIZE, SIZE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def SecondsConverter(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return hours, minutes, seconds

def WindowGauss(sigma):
    window_gauss = signal.windows.gaussian(int(2 * 5 * sigma), sigma)
    window_gauss /= np.sqrt(2 * np.pi * sigma ** 2)
    return window_gauss

def Mask(x, m=None, M=None):
    if isinstance(x, list):
        x = np.array(x)

    x = np.sort(x)

    if m is None and M is None:
        raise ValueError("ERROR! Set a at least one between m and M")

    start = 0
    end = len(x)

    if m is not None:
        m += 1e-20
        start = np.searchsorted(x, m, side='left')
    if M is not None:
        M -= 1e-20
        end = np.searchsorted(x, M, side='right')

    return x[start:end]

def MinLengthDict(d, current_path=()):
    min_path = None
    min_value = None
    min_length = float('inf')  # Partiamo da un valore grande

    for key, value in d.items():
        path = current_path + (key,)

        if isinstance(value, dict):
            sub_path, sub_length, sub_value = MinLengthDict(value, path)
            if sub_length < min_length:
                min_length = sub_length
                min_path = sub_path
                min_value = sub_value

        elif isinstance(value, list) and value:  # Evita liste vuote non utili
            if len(value) < min_length:
                min_length = len(value)
                min_path = path
                min_value = value

    # Se nessuna lista è stata trovata, restituisce None
    return (min_path, min_length, min_value) if min_path else (None, None, None)

def MemoryUsage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Converti in MB

def SearchFiles(dir_path, file_name_target):
    files_found = []
    for dir, subdir, files in os.walk(dir_path):
        for file_name in files:
            if file_name == file_name_target:
                files_found.append(os.path.join(dir, file_name))

    return files_found

def SearchDirectory(main_path):
    directory = []
    for nome in os.listdir(main_path):
        percorso_elemento = os.path.join(main_path, nome)
        if os.path.isdir(percorso_elemento):
            directory.append(percorso_elemento)
    return directory

def LoadArray(path, data_type, event_type=None,conn_type=None):

    data = []
    try:
        if data_type == 'spikes':
            data = np.asarray(np.load(path, allow_pickle=True).item()[event_type], dtype=object)
        elif data_type == 'weight' or data_type == 'delay':
            data = np.asarray(np.load(path, allow_pickle=True).item()[conn_type][data_type], dtype=object)
    except KeyError as e:
        print(f"Error: Key '{e.args[0]}' not found")

    return data

def FlattenList2List(input_list):

    flatten_list = []
    for item in input_list:
        if isinstance(item, list):
            flatten_list.extend(FlattenList2List(item))
        elif isinstance(item, np.ndarray):
            flatten_list.extend(FlattenList2List(list(item)))
        else:
            flatten_list.append(item)

    return flatten_list

def FlattenDict2List(input_dict, fl_list=True):

    flatten_list = []
    for key, value in input_dict.items():
        if isinstance(value, dict):
            flatten_list.extend(FlattenDict2List(value))
        elif isinstance(value, list) and fl_list==True:
            flatten_list.extend(FlattenList2List(value))
        else:
            flatten_list.append(value)

    return flatten_list

def KeyExists(dic, keys):

    if type(keys) == str: keys = (keys,)

    current_level = dic
    for key in keys:
        if isinstance(current_level, dict) and key in current_level:
            current_level = current_level[key]
        else:
            return False
    return True

def GetDictValue(dic, keys):
    for key in keys:
        if isinstance(dic, dict) and key in dic:
            dic = dic[key]
        else:
            return None
    return dic

def GaussConvolution(data, sigma):
    data_conv = signal.convolve(data, WindowGauss(sigma), 'same')
    return data_conv

def CountSpikes(spikes, t1, t2):
    i_t1, i_t2 = bisect.bisect_left(spikes, t1), bisect.bisect_right(spikes, t2)
    n_spikes = i_t2 - i_t1
    return n_spikes

def FiringRate(spikes, t_start, t_stop, dt_s=0.1, dt_ds=5, nu_t_high=500, nu_t_low=None, remove_zeros=True, conv=False):

    #prepare array
    if len(spikes) > 0:
        spikes = np.sort(np.hstack(spikes))

    #conv params
    sample_rate = 1000 / dt_s # (Hz)
    si_t_high = sample_rate / (2 * np.pi * nu_t_high) #nu_t (Hz)
    if nu_t_low != None: si_t_low = sample_rate / (2 * np.pi * nu_t_low)


    #sampling
    n_steps_s = round((t_stop - t_start) / dt_s)
    sampling = t_start + np.arange(n_steps_s) * dt_s
    bw_s = dt_s
    fr_bin_s = np.zeros(len(sampling))

    if dt_ds > 0:
        #downsampling
        n_steps_ds = round((t_stop - t_start) / dt_ds)
        downsampling = t_start + np.arange(n_steps_ds) * dt_ds
        fr_bin_ds = np.zeros(len(downsampling))

    #sampling fr
    for bin, t in enumerate(sampling):
        t1_s, t2_s = t - bw_s/2, t + bw_s/2
        n_spikes = CountSpikes(spikes, t1_s, t2_s)
        fr_bin_s[bin] = n_spikes

    fr = fr_bin_s

    #convolution
    if conv: fr = signal.convolve(fr, WindowGauss(si_t_high), 'same')

    if dt_ds > 0:
        for i_ds, t in enumerate(downsampling):
            i1_ds, i2_ds = i_ds, i_ds + 1
            i1_s, i2_s = int(i1_ds * (dt_ds / dt_s)), int(i2_ds * (dt_ds / dt_s))
            n_spikes = np.sum(fr[i1_s:i2_s])
            fr_bin_ds[i_ds] = n_spikes

        if conv: fr = fr_bin_ds

    if nu_t_low != None:
        fr = signal.convolve(fr, WindowGauss(si_t_low), 'same')

    if dt_ds > 0:
        fr = np.array(fr) * (1000 / dt_ds)
    else:
        fr = np.array(fr) * (1000 / dt_s)

    if remove_zeros==True: fr = fr[(fr > 0)]

    return fr

def MeanWeighted(mu, std, axis=0):

    mu = np.asarray(mu)
    std = np.asarray(std)

    weights = 1 / std**2
    sum_weights = np.sum(weights, axis=axis)
    mu_weighted = np.sum(mu * weights, axis=axis) / sum_weights
    std_weighted = np.sqrt(1 / sum_weights)

    return mu_weighted, std_weighted

def SumWeighted(mu, std):
    mu = np.array(mu)
    std = np.array(std)

    weights = 1 / std ** 2
    sum_weighted = np.array([np.sum(x * w) for x, w in zip(mu, weights)])
    combined_std = np.array([np.sqrt(np.sum(1 / w)) for w in weights])

    return sum_weighted, combined_std

def MeanTrials(data, mode=None):

    n_stages = len(data)

    if mode == 'ratio':
        n, d = data[:], data[0][:]
        data = np.divide(n, d, where=d != 0)
        data[np.broadcast_to((d == 0), data.shape)] = np.nan
    elif mode == 'relative':
        n, d = data[:] - data[0][:], data[0][:]
        data = np.divide(n, d, where=d != 0)
        data[np.broadcast_to((d == 0), data.shape)] = 0

    mu_trial, std_trial = (
        np.array([[np.nanmean(data_trial) for data_trial in data[nstage]] for nstage in range(n_stages)]),
        np.array([[np.nanstd(data_trial) / np.sqrt(np.sum((data_trial != np.nan))) for data_trial in data[nstage]] for nstage in range(n_stages)]) + 1e-20)

    mu, std = MeanWeighted(mu_trial, std_trial, axis=1)

    return mu, std

def PropagationErrors(mu_x, std_x, mu_y, std_y, mode="sum"):
    """
    Propagazione degli errori tra due variabili x e y.

    Parametri
    ----------
    mu_x : array-like
        Valori medi di x
    std_x : array-like
        Deviazioni standard di x
    mu_y : array-like
        Valori medi di y
    std_y : array-like
        Deviazioni standard di y
    mode : {"sum", "product", "division"}
        Modalità di propagazione:
        - "sum"      -> z = x + y
        - "product"  -> z = x * y
        - "division" -> z = x / y

    Ritorna
    -------
    mu_z : ndarray
        Valori medi di z
    std_z : ndarray
        Deviazioni standard propagate di z
    """
    if mu_x == 0:
        mu_x = np.zeros_like(mu_y)
        std_x = np.ones_like(std_y) * 1e-20
    elif mu_x == 1:
        mu_x = np.ones_like(mu_y)
        std_x = np.ones_like(std_y) * 1e-20
    else:
        mu_x = np.array(mu_x)
        std_x = np.array(std_x)

    if mu_y == 0:
        mu_y = np.zeros_like(mu_x)
        std_y = np.ones_like(std_x) * 1e-20
    elif mu_x == 1:
        mu_y = np.ones_like(mu_x)
        std_y = np.ones_like(std_x) * 1e-20
    else:
        mu_y = np.array(mu_x)
        std_y = np.array(std_x)

    if mode == "sum":
        mu_z = mu_x + mu_y
        std_z = np.sqrt(std_x**2 + std_y**2)

    elif mode == "product":
        mu_z = mu_x * mu_y
        rel_std = np.sqrt((std_x/mu_x)**2 + (std_y/mu_y)**2)
        std_z = np.abs(mu_z) * rel_std

    elif mode == "division":
        mu_z = mu_x / mu_y
        rel_std = np.sqrt((std_x/mu_x)**2 + (std_y/mu_y)**2)
        std_z = np.abs(mu_z) * rel_std

    else:
        raise ValueError("mode deve essere 'sum', 'product' o 'division'")

    return mu_z, std_z

def ShiftCMap(cmap, rgb):

    # 1. Parametri base
    n_colors_original = 256
    crop_size = 192  # quanti colori tenere
    # 2. Carica terrain e applica offset
    terrain = cm.get_cmap(cmap, n_colors_original)
    terrain_colors = terrain(np.linspace(0, 1, n_colors_original))  # (256, 4)
    offset = np.concatenate([rgb, [0]])  # RGB shift
    shifted_colors = terrain_colors + offset
    shifted_colors = np.clip(shifted_colors, 0, 1)
    # 3. Taglia la colormap a 192 colori (puoi cambiare range)
    cropped_colors = shifted_colors[:crop_size]  # oppure [start:stop]
    # 4. Interpola per tornare a 256 colori
    # Genera nuove posizioni su [0, 1]
    interp_indices = np.linspace(0, crop_size - 1, n_colors_original)
    resampled_colors = np.empty((n_colors_original, 4))
    for i in range(4):  # RGBA
        resampled_colors[:, i] = np.interp(
            np.linspace(0, crop_size - 1, n_colors_original),
            np.arange(crop_size),
            cropped_colors[:, i]
        )
    # 5. Crea la nuova colormap
    custom_cmap = ListedColormap(resampled_colors, name=f"{cmap}_shifted_cropped")
    return custom_cmap

def BrokenAx(axes, length, height, shift, width=1, label=None, axis='x'):
    """
    Applica le lineette di rottura (//) tra assi consecutivi e opzionalmente una label centrata.

    Parameters:
        axes (list): Lista di oggetti Axes (in ordine).
        length (float): Lunghezza delle lineette (in unità di axes coords).
        height (float): Altezza delle lineette (in unità di axes coords).
        width (float): Spessore delle lineette.
        shift (float): Spostamento orizzontale (x) o verticale (y) delle lineette.
        label (dict or None): Se specificato, un dizionario con chiavi 'fig', 'label', 'fontsize'.
        axis (str): 'x' (default) o 'y', indica la direzione della rottura.
    """
    n = len(axes)
    if n < 2:
        return  # niente da fare

    for i in range(n - 1):
        ax1 = axes[i]
        ax2 = axes[i + 1]

        # Disegna le lineette tra ax1 e ax2
        if axis == 'x':
            # Left plot (ax1)
            kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=width)
            x0 = 1 - length + shift
            x1 = 1 + length + shift
            y0 = -height
            y1 = +height
            ax1.plot([x0, x1], [y0, y1], **kwargs)

            # Right plot (ax2)
            kwargs['transform'] = ax2.transAxes
            x0 = -length - shift
            x1 = +length - shift
            ax2.plot([x0, x1], [y0, y1], **kwargs)

        elif axis == 'y':
            # Bottom plot (ax1)
            kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=width)
            y0 = 1 - length + shift
            y1 = 1 + length + shift
            x0 = -height
            x1 = +height
            ax1.plot([x0, x1], [y0, y1], **kwargs)

            # Top plot (ax2)
            kwargs['transform'] = ax2.transAxes
            y0 = -length - shift
            y1 = +length - shift
            ax2.plot([x0, x1], [y0, y1], **kwargs)

    # Aggiunge la label centrata
    if label is not None:
        fig, ax_label, fontsize = label['fig'], label['label'], label['fontsize']

        if axis == 'x':
            x_start = axes[0].get_position().x0
            x_end = axes[-1].get_position().x1
            y_pos = axes[0].get_position().y0 + label['yoffset']
            x_center = (x_start + x_end) / 2
            fig.text(x_center, y_pos, ax_label, ha='center', fontsize=fontsize)
        elif axis == 'y':
            x_pos = axes[0].get_position().x0 + label['xoffset']
            y_start = axes[0].get_position().y0
            y_end = axes[-1].get_position().y1
            y_center = (y_start + y_end) / 2
            fig.text(x_pos, y_center, ax_label, ha='center', va='center', rotation='vertical', fontsize=fontsize)

def AxLines(fig, axes, colors, labels, y_offset=0.01, text_offset=0.005, linewidth=2, fontsize=10):
    from matplotlib import pyplot as plt
    """
    Disegna linee orizzontali sopra ciascun asse e piazza una label centrata su ogni asse.

    Parameters:
    - fig: figura matplotlib
    - axes: lista di assi su cui disegnare le linee
    - colors: lista di colori, uno per ciascuna linea
    - labels: lista di etichette da mostrare sopra ogni linea
    - y_offset: distanza verticale (in coordinate figura) sopra l'asse per la linea
    - text_offset: ulteriore distanza per il testo sopra la linea
    - linewidth: spessore della linea
    """
    for ax, color, label in zip(axes, colors, labels):
        # Bounding box in coordinate figura
        bbox = ax.get_position()
        x0, x1 = bbox.x0 + 0.0025, bbox.x1
        y = bbox.y1 + y_offset  # y subito sopra l'asse

        # Disegna la linea
        fig.lines.append(plt.Line2D([x0, x1], [y, y], transform=fig.transFigure, color=color, linewidth=linewidth))

        # Etichetta centrata
        x_center = (x0 + x1) / 2
        fig.text(x_center, y + text_offset, label, ha='center', va='bottom', fontsize=fontsize, color=color)

def MeanRatio(mu, std):

    mu0, std0 = mu[0], std[0]
    ratio, err = mu / mu0, np.sqrt((std / mu0) ** 2 + (std0 * mu / mu0 ** 2) ** 2)

    return ratio, err

def KDE1D(x, bin_width, xrange=None, log=False):

    xmin, xmax = x.min(), x.max()

    if xrange == None:
        xbins = int((xmax - xmin) / bin_width)
        xbins = complex(0,xbins)
    #elif xmin > xrange[0] or xmax < xrange[1]:
    #    print('\nError: xrange insert not valid!')
    #    xbins = int((xmax - xmin) / bin_width)
    #    xbins = complex(0, xbins)
    else:
        xbins = int((xrange[1] - xrange[0]) / bin_width)
        xbins = complex(0, xbins)
        xmin, xmax = xrange

    #print('\nxmin: %lg , xmax: %lg , bin width: %lg' %(xmin,xmax,bin_width))

    bins = int((xmax - xmin) / bin_width)


    xx = np.linspace(xmin,xmax,bins) if log == False else np.logspace(np.log10(xmin),np.log10(xmax),bins)

    kde_uni = KDEUnivariate(x)
    dens = kde_uni.fit(kernel='gau',bw='normal_reference')
    #print('\nbandwith found: %s' % (dens.bw))
    p = dens.evaluate(xx)

    return xx, p / np.sum(p)

def Histogram(data, nbins=None, dx=None ,xmin=None, xmax=None, cumulative=False, logx=False, KDE=False, norm=True):

    if type(data) != np.array: data = np.array(data)

    if len(data) == 0:
        return [], []
    else:
        if xmin == None: xmin = np.min(data)
        if xmax == None: xmax = np.max(data)

        if nbins == None and dx == None:
            raise ValueError('\nPlease, provide either the number bins or the precision to be used')
        elif dx != None:
            nbins = int((xmax - xmin) / dx)

        data = Mask(data, xmin, xmax)

        if logx == False:
            bins = np.linspace(xmin, xmax, nbins + 1)
        elif logx == True:
            bins = np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)

        hist, bin = np.histogram(data, bins=bins)
        if norm == True: hist = hist / np.sum(hist)

        if cumulative == True:
            hist_plot = np.array([np.sum(hist[:l]) for l in np.arange(0,len(hist),dtype=int)])
        elif cumulative == False:
            hist_plot = hist
            if KDE == True:
                bin_width = (xmax - xmin) / nbins
                bin, hist_plot = KDE1D(x=data, bin_width=bin_width, xrange=(xmin,xmax))

        if len(bin) > len(hist_plot): bin = bin[:-1]
        return bin, hist_plot

def LoadWeights(syn_path, syn_type='exc_inh', reshape=None):

    data = None
    try:
        data = np.load(syn_path, allow_pickle=True).item()[syn_type]
    except FileNotFoundError:
        print(f"\nPath doesn't exists!\n{syn_path} - {syn_type}")

    if data != None:
        w = data['weight']
        if reshape != None:
            if len(w) < np.prod(reshape):
                n_neurons = reshape[0]
                w_new = w[:]
                for n_row in range(n_neurons): w_new = np.insert(w_new, n_row * n_neurons + n_row, np.nan)
                w = np.reshape(w_new, (n_neurons, n_neurons))
            else:
                w = np.reshape(w, reshape)
        else:
            w = np.array(w)
        return w

def GetDictValue(dic, keys):
    for key in keys:
        #print(key)
        if isinstance(dic, dict) and key in dic:
            dic = dic[key]
            #print(dic)
        else:
            print(f'\nKey not found {key}')
            dic = None
            break
    return dic

def SetDictValue(d, keys, val, create=False, append=False):

    for key in keys[:-1]:
        if key not in d:
            if create:
                d[key] = {}
            else:
                print(f"Error: Key not found {key}")
                return
        d = d[key]

    if append:
        if isinstance(d[keys[-1]], list):
            d[keys[-1]].append(val)
        else:
            print(f"Error: List not found {keys[-1]}:{d[keys[-1]]}")
            return
    else:
        d[keys[-1]] = val

def CreateDictfromLists(val_type, *keys_lists):
    # Caso base: se non ci sono chiavi, restituisce un dizionario vuoto o una lista vuota a seconda di val_type
    if not keys_lists:
        return [] if val_type == list else {}

    first_keys = keys_lists[0]
    nested_dict = {k: CreateDictfromLists(val_type, *keys_lists[1:]) for k in first_keys}

    # Se siamo nell'ultimo livello, riempiamo con liste vuote o dizionari vuoti a seconda di val_type
    if len(keys_lists) == 1:
        if val_type == list:
            nested_dict = {k: [] for k in first_keys}
        else:  # val_type == dict
            nested_dict = {k: {} for k in first_keys}

    return nested_dict


import sys, os


def kde2D_multivariate(x, y, xbin_width, ybin_width, xrange=None, yrange=None, bw='normal_reference'):
    print('\nKDE Multivariate')
    # create grid of sample locations (default: 100x100)

    if xrange != None:
        xmin, xmax = xrange
    else:
        xmin, xmax = x.min(), x.max()
    if yrange != None:
        ymin, ymax = yrange
    else:
        ymin, yxmax = y.min(), y.max()

    xbins = int((xmax - xmin) / xbin_width)
    xbins = complex(0, xbins)

    ybins = int((ymax - ymin) / ybin_width)
    ybins = complex(0, ybins)

    xx, yy = np.mgrid[xmin:xmax:xbins,
             ymin:ymax:ybins]

    xy_sample = np.vstack([xx.ravel(), yy.ravel()])
    xy_values = np.vstack([x, y])

    kde_multi = KDEMultivariate(data=xy_values, var_type='cc', bw=bw)

    print('\nbandwith found: %s' % (kde_multi.bw))

    # score_samples() returns the log-likelihood of the samples
    zz = np.reshape(kde_multi.pdf(xy_sample).T, xx.shape)
    return xx, yy, zz / np.sum(zz)


def GetMNISTFeatures(model, dataset_name):

    mnist_path = f'./../../../Dataset/PreprocessedData/{model}'
    mnist_path = os.path.abspath(mnist_path)

    features = np.load(os.path.join(mnist_path,f'{dataset_name}_features.npy'))
    labels = np.load(os.path.join(mnist_path,f'{dataset_name}_labels.npy'))

    return features, labels

def ThCxInput(index_path, features_train, features_test, th_cx_matrix, net_params):
    import time
    st = time.time()

    index_dict = np.load(index_path, allow_pickle=True).item()
    features_index_train = index_dict['training']['index mnist']
    features_index_test = index_dict['test']['index mnist']
    features_class_train = [[features_train[index] for index in index_list] for cl, index_list in features_index_train.items()]
    features_class_test = [[features_test[index] for index in index_list] for cl, index_list in features_index_test.items()]

    n_class, n_ranks, n_neur_th, n_neur_group, n_areas = (net_params['n_class'], net_params['n_ranks_training'],
                                                          len(features_class_train[0][0]), net_params['n_neur_group'], net_params['n_areas'])
    n_groups, n_neur_cx_area = n_class * n_ranks, n_class * n_ranks * n_neur_group
    w_input_cx_sum = {'group':[], 'class':[], 'non-specific':[]}

    for cl_example, features_class in enumerate(features_class_train):
        for rank_example, feat in enumerate(features_class):
            pattern_example = feat
            w_input_cx = np.dot(pattern_example, th_cx_matrix)
            group_cx = cl_example * n_ranks + rank_example
            for neur in range(group_cx * n_neur_group, (group_cx + 1) * n_neur_group):
                 w_input_cx_sum['group'].append( w_input_cx[neur] / n_areas)

    for cl_example, features_class in enumerate(features_class_test):
        for rank_example, feat in enumerate(features_class):
            pattern_example = feat
            w_input_cx = np.dot(pattern_example, th_cx_matrix)
            for neur in range(n_neur_cx_area):
                group = neur // n_neur_group
                cl_cx = group //n_ranks % n_class
                w_input_cx_neur = w_input_cx[neur] / n_areas
                if cl_cx == cl_example: w_input_cx_sum['class'].append(w_input_cx_neur)
                else: w_input_cx_sum['non-specific'].append(w_input_cx_neur)

    return w_input_cx_sum

def CxThInput(index_path, features_train, features_test, cx_th_matrix, net_params):
    import time
    st = time.time()

    index_dict = np.load(index_path, allow_pickle=True).item()
    features_index_train = index_dict['training']['index mnist']
    features_index_test = index_dict['test']['index mnist']
    features_class_train = [[features_train[index] for index in index_list] for cl, index_list in features_index_train.items()]
    features_class_test = [[features_test[index] for index in index_list] for cl, index_list in features_index_test.items()]

    n_class, n_ranks, n_neur_th, n_neur_group, n_areas = (net_params['n_class'], net_params['n_ranks_training'],
                                                          len(features_class_train[0][0]), net_params['n_neur_group'], net_params['n_areas'])
    n_neur_cx_area = n_class * n_ranks * n_neur_group
    n_groups = n_class * n_ranks * n_areas
    w_input_th = {'group':[], 'class':[], 'non-specific':[]}

    for ngroup in range(n_groups):
        cl_cx = ngroup //n_ranks % n_class
        rank_cx = ngroup % n_ranks
        w_cx_th_group = cx_th_matrix[ngroup * n_neur_group: (ngroup + 1) * n_neur_group]
        pattern_example = features_class_train[cl_cx][rank_cx].astype(bool)
        w_input_th_neur = np.sum([w_cx_th_group[neur][pattern_example] for neur in range(n_neur_group)], axis=0)
        w_input_th['group'].extend(w_input_th_neur)


    for ngroup in range(n_groups):
        cl_cx = ngroup //n_ranks % n_class
        w_cx_th_group = cx_th_matrix[ngroup * n_neur_group: (ngroup + 1) * n_neur_group]
        for cl_example, features_class in enumerate(features_class_test):
            for rank_example, feat in enumerate(features_class):
                pattern_example = feat.astype(bool)
                w_input_th_neur = np.sum([w_cx_th_group[neur][pattern_example] for neur in range(n_neur_group)],axis=0)
                if cl_cx == cl_example:
                    w_input_th['class'].extend(w_input_th_neur)
                else:
                    w_input_th['non-specific'].extend(w_input_th_neur)

    return w_input_th

def CxCxInput(cx_cx_index, cx_cx_matrix, net_params):

    n_class, n_ranks, n_areas, n_exc_ca = net_params['n_class'], net_params['n_ranks_training'], net_params['n_areas'], net_params['n_neur_group']
    n_neurons = n_areas * n_class * n_ranks * n_exc_ca
    flatten_matrix = np.hstack(cx_cx_matrix)
    w_cx_cx_sum = {'group': [], 'class': [], 'non-specific': []}

    for neur in range(n_neurons):
        w_cx_cx_sum['group'].append(np.sum(flatten_matrix[cx_cx_index[neur]['group']]))
        w_cx_cx_sum['class'].append(np.sum(flatten_matrix[cx_cx_index[neur]['class']]))
        w_cx_cx_sum['non-specific'].append(np.sum(flatten_matrix[cx_cx_index[neur]['non-specific']]))

    return w_cx_cx_sum

def get_total_size(obj, seen=None):
    if seen is None:
        seen = set()

    size = sys.getsizeof(obj)

    # Controlla se l'oggetto è già stato visto per evitare loop
    if id(obj) in seen:
        return 0

    # Aggiungi l'oggetto corrente alla lista di visti
    seen.add(id(obj))

    # Se l'oggetto è una collezione (lista, dizionario, set, ecc.)
    if isinstance(obj, (list, tuple, set)):
        size += sum(get_total_size(item, seen) for item in obj)
    elif isinstance(obj, dict):
        size += sum(get_total_size(k, seen) + get_total_size(v, seen) for k, v in obj.items())

    return size

def MaxLenghtDict(d, current_path=()):
    max_length = 0
    max_path = None
    max_value = None

    for key, value in d.items():
        path = current_path + (key,)
        if isinstance(value, dict):
            # Ricorsione nei sotto-dizionari
            sub_path, sub_length, sub_value = MaxLenghtDict(value, path)
            if sub_length > max_length:
                max_length = sub_length
                max_path = sub_path
                max_value = sub_value
        elif isinstance(value, list):
            # Controlla la lunghezza della lista
            if len(value) > max_length:
                max_length = len(value)
                max_path = path
                max_value = value

    return max_path, max_length, max_value

def MinLenghtDict(d, current_path=()):

    min_length = float('inf')
    min_path = None
    min_value = None

    for key, value in d.items():
        path = current_path + (key,)
        if isinstance(value, dict):
            sub_path, sub_length, sub_value = MinLenghtDict(value, path)
            if sub_length < min_length:
                min_length = sub_length
                min_path = sub_path
                min_value = sub_value
        elif isinstance(value, list):
            if len(value) < min_length:
                min_length = len(value)
                min_path = path
                min_value = value

    return min_path, min_length, min_value

def NestedEmptyList(dimensions):
    if len(dimensions) == 1:
        return [[] for _ in range(dimensions[0])]
    else:
        return [NestedEmptyList(dimensions[1:]) for _ in range(dimensions[0])]

def LoadDictFromYaml(file_path):
    file_dict = None
    try:
        with open(file_path, 'r') as f:
            file_dict = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"\nPath doesn't exists!\n{file_path}")

    return file_dict

def WeightsCategories(w_matrix, index_cat_dict,n_areas, n_class, n_ranks, cat='all'):

    n_groups = n_areas * n_class * n_ranks

    weight_cats_dict = {'group': [], 'class': [], 'non-specific': []}
    syn_group, syn_class, syn_diff = [], [], []

    for matrix in w_matrix:
        if cat == 'group' or cat == 'all':
            w_group = np.hstack(matrix)[index_cat_dict['group']]
            syn_group.append(w_group)
        if cat == 'class' or cat == 'all':
            w_class = np.hstack(matrix)[index_cat_dict['class']]
            syn_class.append(w_class)
        if cat == 'non-specific' or cat == 'all':
            w_diff = np.hstack(matrix)[index_cat_dict['non-specific']]
            syn_diff.append(w_diff)

    if cat == 'group' or cat == 'all':
        weight_cats_dict['group'] = np.hstack(syn_group)
    if cat == 'class' or cat == 'all':
        weight_cats_dict['class'] = np.hstack(syn_class)
    if cat == 'non-specific' or cat == 'all':
        weight_cats_dict['non-specific'] = np.hstack(syn_diff)

    return weight_cats_dict if cat == 'all' else weight_cats_dict[cat]

def GetCxCxIndex(n_areas, n_class, n_ranks, n_exc_ca):
    cycle = 1
    n_groups = n_areas * n_class * n_ranks
    n_neurons = n_groups * n_exc_ca * n_areas
    n_synapses = n_neurons ** 2
    neur_input_index_dict = {neur: {'group': np.zeros(n_synapses, dtype=bool), 'class': np.zeros(n_synapses, dtype=bool), 'non-specific': np.zeros(n_synapses, dtype=bool)} for neur in range(n_neurons)}
    #group_index_dict = {ngroup: np.zeros(n_synapses, dtype=int) for ngroup in range(n_groups)}


    for i in range(n_neurons):
        for j in range(n_neurons):
            group1, group2 = i // n_exc_ca, j // n_exc_ca
            class1, class2 = group1 // (n_ranks * cycle) % n_class, group2 // (n_ranks * cycle) % n_class
            area1, area2 = group1 // (n_class * n_ranks * cycle), group2 // (n_class * n_ranks * cycle)
            index_flatten = n_neurons * i + j
            if area1 == area2:
                if group1 == group2: #and i != j:
                    neur_input_index_dict[i]['group'][index_flatten] = True
                    #group_index_dict[group1][index_flatten] = 1
                elif class1 == class2 and group1 != group2:
                    neur_input_index_dict[i]['class'][index_flatten] = True
                elif class1 != class2:
                    neur_input_index_dict[i]['non-specific'][index_flatten] = True
            elif area1 != area2:
                delta = np.abs(group1 - group2)
                if delta == n_class * n_ranks * cycle:
                    neur_input_index_dict[i]['group'][index_flatten] = True
                elif class1 == class2 and delta != n_class * n_ranks * cycle:
                    neur_input_index_dict[i]['class'][index_flatten] = True
                elif class1 != class2:
                    neur_input_index_dict[i]['non-specific'][index_flatten] = True
    return neur_input_index_dict#, group_index_dict


def MaskInterval(x, start=0, intra_lenght=5, inter_lenght=10):
    mask = np.zeros_like(x, dtype=bool)
    while start < len(x):
        end = min(start + intra_lenght, len(x))
        mask[start:end] = True
        start += inter_lenght
    return mask

def EllipseDistribution(x, y, chi2=2.447):

    points = np.column_stack((x, y))
    pca = PCA(n_components=2)
    pca.fit(points)
    mean = pca.mean_
    components = pca.components_
    explained_var = pca.explained_variance_

    width, height = 2 * chi2 * np.sqrt(explained_var)
    angle = np.degrees(np.arctan2(*components[0][::-1]))

    return mean, width, height, angle

def GetCxThIndex(n_areas, n_class, n_ranks, n_exc_ca):
    cycle = 1
    n_groups = n_areas * n_class * n_ranks
    n_neurons = n_groups * n_exc_ca * n_areas
    n_synapses = n_neurons ** 2
    neur_input_index_dict = {neur: {'group': np.zeros(n_synapses, dtype=bool), 'class': np.zeros(n_synapses, dtype=bool), 'non-specific': np.zeros(n_synapses, dtype=bool)} for neur in range(n_neurons)}
    #group_index_dict = {ngroup: np.zeros(n_synapses, dtype=int) for ngroup in range(n_groups)}


    for i in range(n_neurons):
        for j in range(n_neurons):
            group1, group2 = i // n_exc_ca, j // n_exc_ca
            class1, class2 = group1 // (n_ranks * cycle) % n_class, group2 // (n_ranks * cycle) % n_class
            area1, area2 = group1 // (n_class * n_ranks * cycle), group2 // (n_class * n_ranks * cycle)
            index_flatten = n_neurons * i + j
            if area1 == area2:
                if group1 == group2: #and i != j:
                    neur_input_index_dict[i]['group'][index_flatten] = True
                    #group_index_dict[group1][index_flatten] = 1
                elif class1 == class2 and group1 != group2:
                    neur_input_index_dict[i]['class'][index_flatten] = True
                elif class1 != class2:
                    neur_input_index_dict[i]['non-specific'][index_flatten] = True
            elif area1 != area2:
                delta = np.abs(group1 - group2)
                if delta == n_class * n_ranks * cycle:
                    neur_input_index_dict[i]['group'][index_flatten] = True
                elif class1 == class2 and delta != n_class * n_ranks * cycle:
                    neur_input_index_dict[i]['class'][index_flatten] = True
                elif class1 != class2:
                    neur_input_index_dict[i]['non-specific'][index_flatten] = True
    return neur_input_index_dict#, group_index_dict

def GetCatIndex(n_areas, n_class, n_ranks, n_exc_ca):

    cycle = 1
    n_groups = n_areas * n_class * n_ranks
    n_neurons = n_groups * n_exc_ca
    n_synapses = n_neurons ** 2
    index_cat_dict = {'group': np.zeros(n_synapses,dtype=bool), 'class': np.zeros(n_synapses,dtype=bool), 'non-specific': np.zeros(n_synapses,dtype=bool)}

    for i in range(n_neurons):
        for j in range(n_neurons):
            group1, group2 = i // n_exc_ca, j // n_exc_ca
            class1, class2 = group1 // (n_ranks * cycle) % n_class, group2 // (n_ranks * cycle) % n_class
            area1, area2 = group1 // (n_class * n_ranks * cycle), group2 // (n_class * n_ranks * cycle)
            index_flatten = n_neurons * i + j
            if area1 == area2:
                if group1 == group2 and i != j:
                    index_cat_dict['group'][index_flatten] = True
                elif class1 == class2 and group1 != group2:
                    index_cat_dict['class'][index_flatten] = True
                elif class1 != class2:
                    index_cat_dict['non-specific'][index_flatten] = True
            elif area1 != area2:
                delta = np.abs(group1 - group2)
                if delta == n_class * n_ranks * cycle:
                    index_cat_dict['group'][index_flatten] = True
                elif class1 == class2 and delta != n_class * n_ranks * cycle:
                    index_cat_dict['class'][index_flatten] = True
                elif class1 != class2:
                    index_cat_dict['non-specific'][index_flatten] = True

    return index_cat_dict


def NetAccuracy(spikes_trial, trial_dict_path, stage,  params, prediction=['neuron'], nconf=None):

    if type(prediction) == str(): prediction = [prediction]

    net_params, times  = params['network'], params['times']

    n_areas, n_class, n_ranks_train, n_ranks_test, n_exc_ca, t_img_test, t_pause = (net_params['n_areas'], net_params['n_class'],
                                net_params['n_ranks_train'], net_params['n_ranks_test'], net_params['n_exc_ca'], net_params['t_img_test'],
                                net_params['t_pause'])

    n_groups, n_img_test, t_img = n_areas * n_class * n_ranks_train, n_ranks_test * n_class, t_img_test + t_pause
    n_groups_area = n_groups // n_areas
    n_neurons = n_groups * n_exc_ca

    time = times[stage]['classification']
    t_start, t_stop = time['start'], time['stop']

    if nconf != None:
        t_start, t_stop = time['start'][nconf], time['stop'][nconf]

    lp = SearchFiles(trial_dict_path, 'trial_dict.npy')[0]

    labels_trial = np.load(lp, allow_pickle=True, encoding='latin1').item()['test']['labels']

    spikes_neuron = np.array([spikes_neur[bisect.bisect_left(spikes_neur, t_start): bisect.bisect_right(spikes_neur, t_stop)] for spikes_neur in spikes_trial], dtype=object)
    spikes_group = np.array([[np.sort(np.hstack(spikes_neuron[group * n_exc_ca:(group + 1) * n_exc_ca])) if len(spikes_neuron[group * n_exc_ca:(group + 1) * n_exc_ca])>0 else np.array([]) for group in range(area*n_groups_area,(area+1)*n_groups_area)] for area in range(n_areas)],dtype=object)

    accuracy = {pred: 0 for pred in prediction}
    spikes_count = []

    for img in range(n_img_test):

        label = labels_trial[img]
        t0 = t_start + t_pause + img * t_img
        t1 = t0 + t_img_test

        spikes_neuron_img = np.array([CountSpikes(spikes_neur, t0, t1) for spikes_neur in spikes_neuron])
        spikes_group_img = np.array([[CountSpikes(spikes_group[area, group], t0, t1) for group in range(n_groups_area)] for area in range(n_areas)])
        spikes_class_img = np.array([[np.sum(spikes_group_img[area, classe * n_ranks_train:(classe + 1) * n_ranks_train]) for classe in range(n_class)] for area in range(n_areas)])

        winner_neur = np.argmax(spikes_neuron_img)
        winner_group = np.argmax(np.sum([spikes_group_img[area] for area in range(n_areas)], axis=0))
        winner_class = np.argmax(np.sum([spikes_class_img[area] for area in range(n_areas)], axis=0))

        if 'neuron' in prediction:
            pred_group = winner_neur // n_exc_ca
            pred_class = pred_group // n_ranks_train % n_class
            if pred_class == label: accuracy['neuron'] = accuracy['neuron'] + 1
        if 'group' in prediction:
            pred_group = winner_group
            pred_class = pred_group // n_ranks_train % n_class
            if pred_class == label: accuracy['group'] = accuracy['group'] + 1
        if 'class' in prediction:
            pred_class = winner_class
            if pred_class == label: accuracy['class'] = accuracy['class'] + 1

        spikes_count.append((1000 / t_img_test) * np.sum(spikes_neuron_img) / n_neurons )

    for pred in prediction: accuracy[pred] = 100 * accuracy[pred] / n_img_test

    return accuracy, spikes_count

def KNNMonoarea(trial_dict_path, params):

    network, dataset = params['network'], params['dataset']

    n_areas, n_class, n_ranks_train, n_ranks_test = network['n_areas'], network['n_class'], network['n_ranks_train'], network['n_ranks_test']

    n_neighbours = dataset['n_neighbours']
    features_train, labels_train = dataset['features_train'], dataset['labels_train']
    features_test, labels_test = dataset['features_test'], dataset['labels_test']
    balanced_train, balanced_test = dataset['balanced_train'], dataset['balanced_test']

    n_feat = len(features_train[0]) // 2
    confusion_matrix = np.zeros((n_class, n_class))

    trial_dict = np.load(os.path.join(trial_dict_path, 'trial_dict.npy'), allow_pickle=True, encoding='latin1').item()
    index_training_trial = trial_dict['training']['index mnist']
    index_test_trial = trial_dict['test']['index mnist']
    index_test_shuffle_trial = trial_dict['test']['index shuffled']

    if balanced_train == True:
        X_train = np.array([features_train[index] for nclass in range(n_class) for index in index_training_trial[nclass]])
        Y_train = np.array([cl for cl in range(n_class) for rank in range(n_ranks_train)])
    else:
        X_train = np.array([features_train[index] for index in index_training_trial])
        Y_train = np.array([labels_train[index] for index in index_training_trial])

    # TEST
    if balanced_test == True:
        X_test = np.array([features_test[index] for nclass in range(n_class) for index in index_test_trial[nclass]])
        Y_test = np.array([cl for cl in range(n_class) for rank in range(n_ranks_test)])
    else:
        X_test = np.array([features_test[index] for index in index_test_trial])
        Y_test = np.array([labels_test[index] for index in index_test_trial])

    X_test_shuffled, Y_test_shuffled = (np.array( [X_test[index] for index in index_test_shuffle_trial]),
                                        np.array( [Y_test[index] for index in index_test_shuffle_trial]))

    n_img_test_class = [np.sum((Y_test==nclass)) for nclass in range(n_class)]
    n_img_test = len(Y_test_shuffled)

    d = np.array([[np.dot(ex1, ex2) for ex2 in X_train] for ex1 in X_test_shuffled]) / (0.5 * n_feat)
    for example in range(n_img_test):

        lab_test = Y_test_shuffled[example]

        d_min, idx = np.transpose(sorted(zip(d[example], np.arange(len(d[example]))), key=lambda tup: tup[0], reverse=True))
        groups = idx[:n_neighbours].astype(int)
        weight = d_min[:n_neighbours] / np.sum(d_min[:n_neighbours])  # 1 / d_min_2D[:n_neighbours]
        voting = np.zeros(n_class)
        for n, idx in enumerate(groups): voting[Y_train[idx]] += weight[n]
        winner = np.argmax(voting)
        lab = winner

        confusion_matrix[lab, lab_test] += 1

    confusion_matrix = 100 * np.array([confusion_matrix[nclass] / n_img_test_class[nclass] for nclass in range(n_class)])

    return np.trace(confusion_matrix) / n_class
def KNNConfusionMatrix(dataset_path, output_path, params):

    n_areas, n_class, n_ranks_train, n_ranks_test= params['n_areas'], params['n_class'], params['n_ranks_train'], params['n_ranks_test']

    n_img_test  = n_ranks_test * n_class
    n_neighbours = 4


    #ARRAY LOADING
    features_raw_train = np.load(dataset_path + 'mnist_training_features.npy')
    labels_train = np.load(dataset_path + 'mnist_training_labels.npy')

    features_raw_test = np.load(dataset_path + 'mnist_test_features.npy')
    labels_test = np.load(dataset_path + 'mnist_test_labels.npy')

    features_class_train = [[] for n in range(n_class)]
    features_class_test = [[] for n in range(n_class)]

    n_feat = len(features_raw_train[0]) // 2

    #TRAINING FEATURES
    for n, lab in enumerate(labels_train[:]):
        if lab < n_class: features_class_train[lab].append(features_raw_train[n])
    #TEST FEATURES
    for n, lab in enumerate(labels_test[:]):
        if lab < n_class: features_class_test[lab].append(features_raw_test[n])

    trials_path = SearchDirectory(output_path)
    n_trials, trials_id = len(trials_path), [path[-10:] for path in trials_path]

    confusion_matrix = np.zeros((n_class, n_class))

    if n_areas == 1:

        for n_trial, trial_id in enumerate(trials_id):
            print(f'Trial: {n_trial + 1}/{n_trials}', end='\r')

            trial_path = trials_path[n_trial]
            lp = SearchFiles(trial_path, 'trial_dict.npy')[0]
            trial_dict = np.load(lp, allow_pickle=True, encoding='latin1').item()

            # TRAINING
            index_training_trial = trial_dict['training']['index mnist']
            feat_class_train_trial = [[features_class_train[cl][index] for index in index_list] for cl, index_list in index_training_trial.items()]

            X_train = np.array([feat_class_train_trial[cl][rank] for cl in range(n_class) for rank in range(n_ranks_train)])
            Y_train = np.array([cl for cl in range(n_class) for rank in range(n_ranks_train)])

            # TEST
            index_test_trial = trial_dict['test']['index mnist']
            index_test_shuffle_trial = trial_dict['test']['index shuffled']
            feat_class_test_trial = [[features_class_test[cl][index] for index in index_list] for cl, index_list in index_test_trial.items()]
            X_test = np.array([feat_class_test_trial[cl][rank] for cl in range(n_class) for rank in range(n_ranks_test)])
            Y_test = np.array([cl for cl in range(n_class) for rank in range(n_ranks_test)])
            X_test_shuffled, Y_test_shuffled = (np.array( [X_test[index] for index in index_test_shuffle_trial]),
                                                np.array( [Y_test[index] for index in index_test_shuffle_trial]))

            d = np.array([[np.dot(ex1, ex2) for ex2 in X_train] for ex1 in X_test_shuffled]) / (0.5 * n_feat)

            acc_2D = 0

            for example in range(n_img_test):

                lab_test = Y_test_shuffled[example]

                d_2D = d[example] ** 2
                d_min_2D, idx_2D = np.transpose(sorted(zip(d_2D, np.arange(len(d_2D))), key=lambda tup: tup[0], reverse=True))
                groups_2D = idx_2D[:n_neighbours].astype(int)
                weight_2D = d_min_2D[:n_neighbours] / np.sum(d_min_2D[:n_neighbours])  # 1 / d_min_2D[:n_neighbours]
                voting_2D = np.zeros(n_class)
                for n, idx in enumerate(groups_2D): voting_2D[Y_train[idx]] += weight_2D[n]
                winner_2D = np.argmax(voting_2D)
                lab_2D = winner_2D

                confusion_matrix[lab_2D, lab_test] += 1

    elif n_areas == 2:

        for n_trial, trial_id in enumerate(trials_id):
            print(f'Trial: {n_trial+1}/{n_trials}', end='\r')

            trial_path = trials_path[n_trial]
            lp = SearchFiles(trial_path, 'trial_dict.npy')[0]
            trial_dict = np.load(lp, allow_pickle=True, encoding='latin1').item()

            #TRAINING
            index_training_trial = trial_dict['training']['index mnist']
            feat_class_train_trial = [[features_class_train[cl][index] for index in index_list] for cl, index_list in index_training_trial.items()]

            X_train_L = np.array([feat_class_train_trial[cl][rank][:n_feat] for cl in range(n_class) for rank in range(n_ranks_train)])
            X_train_R = np.array([feat_class_train_trial[cl][rank][n_feat:] for cl in range(n_class) for rank in range(n_ranks_train)])
            Y_train = np.array([cl for cl in range(n_class) for rank in range(n_ranks_train)])

            # TEST
            index_test_trial = trial_dict['test']['index mnist']
            index_test_shuffle_trial = trial_dict['test']['index shuffled']
            feat_class_test_trial = [[features_class_test[cl][index] for index in index_list] for cl, index_list in index_test_trial.items()]
            X_test_L = np.array([feat_class_test_trial[cl][rank][:n_feat] for cl in range(n_class) for rank in range(n_ranks_test)])
            X_test_R = np.array([feat_class_test_trial[cl][rank][n_feat:] for cl in range(n_class) for rank in range(n_ranks_test)])
            Y_test = np.array([cl for cl in range(n_class) for rank in range(n_ranks_test)])
            X_test_L_shuffled, X_test_R_shuffled, Y_test_shuffled = np.array([X_test_L[index] for index in index_test_shuffle_trial]), np.array([X_test_R[index] for index in index_test_shuffle_trial]), np.array([Y_test[index] for index in index_test_shuffle_trial])

            d_L = np.array([[np.dot(ex1, ex2) for ex2 in X_train_L] for ex1 in X_test_L_shuffled]) / (0.5 * n_feat)
            d_R = np.array([[np.dot(ex1, ex2) for ex2 in X_train_R] for ex1 in X_test_R_shuffled]) / (0.5 * n_feat)

            acc_1D, acc_2D = 0, 0

            for example in range(n_img_test):

                lab_test = Y_test_shuffled[example]

                #d_1D = np.array([np.min([ex_L,ex_R]) for ex_L, ex_R in zip(d_L[example],d_R[example])])
                #d_min_1D, idx_1D = np.transpose(sorted(zip(d_1D,np.arange(len(d_1D))), key=lambda tup: tup[0],reverse=True))
                #groups_1D = idx_1D[:n_neighbours].astype(int)
                #weight_1D = d_min_1D[:n_neighbours] / np.sum(d_min_1D[:n_neighbours])#1 / d_min_1D[:n_neighbours]
                #voting_1D = np.zeros(n_class)
                #for n, idx in enumerate(groups_1D): voting_1D[Y_train[idx]] += weight_1D[n]
                #winner_1D = np.argmax(voting_1D)
                #lab_1D = winner_1D

                d_2D = d_L[example]**2 + d_R[example]**2
                d_min_2D, idx_2D = np.transpose(sorted(zip(d_2D, np.arange(len(d_2D))), key=lambda tup: tup[0],reverse=True))
                groups_2D = idx_2D[:n_neighbours].astype(int)
                weight_2D = d_min_2D[:n_neighbours] / np.sum(d_min_2D[:n_neighbours])#1 / d_min_2D[:n_neighbours]
                voting_2D = np.zeros(n_class)
                for n, idx in enumerate(groups_2D): voting_2D[Y_train[idx]] += weight_2D[n]
                winner_2D = np.argmax(voting_2D)
                lab_2D = winner_2D


                confusion_matrix[lab_2D, lab_test] += 1

    return confusion_matrix / (n_trials * n_img_test)


def FrCorrelation(spikes, params, stage, dt_ds, config=None):
    from matplotlib import pyplot as plt
    params_net, params_times = params['network'], params['times']
    n_areas, n_class, n_ranks, n_exc_ca, n_img_test, t_img_test, t_pause = (params_net['n_areas'], params_net['n_class'],
                                params_net['n_ranks_train'], params_net['n_exc_ca'],
                                params_net['n_img_test'], params_net['t_img_test'],params_net['t_pause'])
    n_groups = n_areas * n_class * n_ranks
    t_test = n_img_test * t_img_test

    tstart, tstop = params_times[stage]['classification']['start'], params_times[stage]['classification']['stop']
    spikes_groups = [Mask(np.sort(np.hstack(spikes[n_exc_ca*ngroup:n_exc_ca*(ngroup+1)])),tstart,tstop) - tstart for ngroup in range(n_groups)]
    spikes_groups_cut = [np.hstack([
        Mask(spikes_groups[ngroup], nimg * (t_img_test + t_pause),  nimg * (t_img_test + t_pause) + t_img_test) - nimg * t_pause
        for nimg in range(n_img_test)]) for ngroup in range(n_groups)]
    fr_groups = np.array([FiringRate(spikes_group, 0, t_test, dt_s=0.1, dt_ds=dt_ds, remove_zeros=False) for spikes_group in spikes_groups_cut])
    fr_group_mu, fr_group_std = np.mean(fr_groups, axis=1), np.std(fr_groups, axis=1)
    corr_matrix = [[[] for j in range(n_groups)] for i in range(n_groups)]
    nsteps = len(fr_groups[0])

    for gr1 in range(n_groups):
        for gr2 in range(n_groups):
            fr1, fr2 = fr_groups[gr1], fr_groups[gr2]
            mu1, std1 = fr_group_mu[gr1], fr_group_std[gr1]
            mu2, std2 = fr_group_mu[gr2], fr_group_std[gr2]
            corr = np.dot((fr1 - mu1), (fr2 - mu2)) / (std1 * std2 * nsteps)
            if corr > 0:
                corr_matrix[gr1][gr2] = corr
            else:
                corr_matrix[gr1][gr2] = 1e-6

    return np.asarray(corr_matrix)

def KNN(dataset_path, n_neighbours, n_class, n_ranks_train, n_img_test, balanced_train, balanced_test):

    features_train = np.load(os.path.join(dataset_path, 'training_features.npy'))
    labels_train = np.load(os.path.join(dataset_path, 'training_labels.npy'))
    features_test = np.load(os.path.join(dataset_path, 'test_features.npy'))
    labels_test = np.load(os.path.join(dataset_path, 'test_labels.npy'))

    n_feat = len(features_train[0]) // 2
    n_ranks_test = n_img_test // n_class
    confusion_matrix = np.zeros((n_class, n_class))

    train_indices = np.arange(len(labels_train))
    test_indices = np.arange(len(labels_test))

    if balanced_train == True:
        train_indices_trial = [np.random.choice(train_indices[(train_indices==nclass)], size=n_ranks_train) for nclass in range(n_class)]
        X_train = np.array([features_train[index] for nclass in range(n_class) for index in train_indices_trial[nclass]])
        Y_train = np.array([cl for cl in range(n_class) for rank in range(n_ranks_train)])
    else:
        train_indices_trial = np.random.choice(train_indices, size=n_ranks_train*n_class)
        X_train = np.array([features_train[index] for index in train_indices_trial])
        Y_train = np.array([labels_train[index] for index in train_indices_trial])

    # TEST
    if balanced_test == True:
        test_indices_trial = [np.random.choice(test_indices[(test_indices==nclass)], size=n_ranks_test) for nclass in range(n_class)]
        X_test = np.array([features_test[index] for nclass in range(n_class) for index in test_indices_trial[nclass]])
        Y_test = np.array([cl for cl in range(n_class) for rank in range(n_ranks_test)])

    else:
        test_indices_trial = np.random.choice(test_indices, size=n_img_test)
        X_test = np.array([features_test[index] for index in test_indices_trial])
        Y_test = np.array([labels_test[index] for index in test_indices_trial])

    index_test_shuffle_trial = np.random.permutation(np.arange(n_img_test))

    X_test_shuffled, Y_test_shuffled = (np.array( [X_test[index] for index in index_test_shuffle_trial]),
                                        np.array( [Y_test[index] for index in index_test_shuffle_trial]))

    n_img_test_class = [np.sum((Y_test==nclass)) for nclass in range(n_class)]
    n_img_test = len(Y_test_shuffled)

    d = np.array([[np.dot(ex1, ex2) for ex2 in X_train] for ex1 in X_test_shuffled]) / (0.5 * n_feat)

    for example in range(n_img_test):

        lab_test = Y_test_shuffled[example]

        d_min, idx = np.transpose(sorted(zip(d[example], np.arange(len(d[example]))), key=lambda tup: tup[0], reverse=True))
        groups = idx[:n_neighbours].astype(int)
        weight = d_min[:n_neighbours] / np.sum(d_min[:n_neighbours])  # 1 / d_min_2D[:n_neighbours]
        voting = np.zeros(n_class)
        for n, idx in enumerate(groups): voting[Y_train[idx]] += weight[n]
        winner = np.argmax(voting)
        lab = winner

        confusion_matrix[lab, lab_test] += 1

    confusion_matrix = 100 * np.array([confusion_matrix[nclass] / n_img_test_class[nclass] for nclass in range(n_class)])

    return np.trace(confusion_matrix) / n_class

def TransitionMatrix(spikes_neurons, stage, substage, params):

    net_params, times = params['network'], params['times']

    n_areas, n_class, n_ranks_train, n_exc_ca = net_params['n_areas'], net_params['n_class'], net_params['n_ranks_train'], net_params['n_exc_ca']
    dt, fr_thresh = net_params['dt_fr'], net_params['thresh_fr']
    n_groups = n_areas * n_class * n_ranks_train

    time = times[stage][substage]
    t_start, t_stop = time['start'], time['stop']
    n_steps = round((t_stop - t_start) / dt)

    #print(t_start, t_stop)

    fr_neurons = np.transpose(np.array([FiringRate(spikes, t_start, t_stop, dt_s=dt, dt_ds=0, remove_zeros=False, conv=False) for spikes in spikes_neurons]))#np.array([FiringRate(spikes, t_start, t_stop, dt_s=dt, dt_ds=0, remove_zeros=False, conv=False) for spikes in spikes_neurons])#

    #import matplotlib.pyplot as plt
    #plt.figure()
    #fr_groups = np.mean(np.reshape(np.transpose(fr_neurons), (n_steps, n_groups, n_exc_ca)),axis=2)
    #print(np.transpose(fr_neurons)[21][:20].tolist())
    #print(fr_groups[21].tolist())
    #plt.imshow(fr_neurons, aspect='auto', cmap='Blues')

    groups_activation = []

    for nbin, t_bin in enumerate(fr_neurons):
        groups_fr = np.mean(np.reshape(t_bin, (n_groups, n_exc_ca)), axis=1)
        #print(nbin * dt, groups_fr)
        groups_active = (groups_fr > fr_thresh).astype(int)
        #print(nbin * dt, groups_fr, groups_active)
        groups_activation.append(groups_active)

    return np.array(groups_activation)

def stars_pvalue(p):
    if p > 0.05:
        return 'n.s.'
    elif p <= 0.05 and p > 0.01:
        return '*'
    elif p <= 0.01 and p > 0.001:
        return '**'
    elif p <= 0.001:
        return '***'
