import sys, psutil, os
import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate
from itertools import combinations
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from scipy import signal
import yaml, bisect

def SecondsConverter(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return hours, minutes, seconds

def LogBins(f, base=10):

    fmin = max(f[0], 1e-3)  # evita log(0)
    fmax = f[-1]

    # Trova gli esponenti interi che racchiudono fmin e fmax
    exp_min = int(np.floor(np.log10(fmin)))
    exp_max = int(np.ceil(np.log10(fmax)))

    bins = []
    for exp in range(exp_min, exp_max):
        decade = np.arange(1, 10) * base**exp
        bins.extend(decade[decade >= fmin])
    bins.append(base**exp_max)
    bins = [b for b in bins if b <= fmax]

    return np.array(bins)
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


def FindSequences(arr):

    arr = np.asarray(arr)
    mask = arr > 0
    edges = np.diff(mask.astype(int))

    start_indices = np.where(edges == 1)[0] + 1
    end_indices = np.where(edges == -1)[0] + 1

    if mask[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if mask[-1]:
        end_indices = np.append(end_indices, len(arr))

    return [arr[start:end] for start, end in zip(start_indices, end_indices)]
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

def CountSpikes(spikes, t1, t2):
    i_t1, i_t2 = bisect.bisect_left(spikes, t1), bisect.bisect_right(spikes, t2)
    n_spikes = i_t2 - i_t1
    return n_spikes

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

def ThaCoSimTime(trials_dir, stage_search):

    print('\nMean Execution stage %s' %(stage_search))

    #trials_dir = '/Users/mac/Documents/Projects/ThaCo3/Output/SimulationOutput/rand_mnist/train_classes_10/train_example_5/MainOutput/rand_mnist_training_test_statistic/'

    if stage_search=='training': file_name = 'run_awake_training.log'
    if stage_search=='nrem': file_name = 'run_nrem.log'
    if stage_search=='rem': file_name = 'run_rem.log'

    files_path = SearchFiles(trials_dir, file_name)

    string_search = {
        'training': {'pre': 'Pre-Training', 'post': 'Post-Training'},
        'nrem': {'pre': 'Pre-NREM Thermalization', 'post': 'Post-NREM'},
        'rem': {'pre': 'Pre-REM Thermalization', 'post': 'Post-REM'},
    }

    string_to_be_searched = [string_search[stage_search]['pre'], string_search[stage_search]['post'], 'Pre-Test', 'Post-Test', 'Total']
    file_lines = []

    stages_time  = [[] for stage in string_to_be_searched]

    for trial, path in enumerate(files_path):
        print('trial: %d/%d'%(trial+1,len(files_path)),end='\r')
        abs_file_path = os.path.abspath(path)
        with open(abs_file_path, 'r') as file:
            for line in file.readlines():
                file_lines.append(line)

        for n, line in enumerate(file_lines):
            words = line.split(' ')
            word1, word2 = None, None
            if len(words)==5:
                word1, word2 = words[0:2]
            elif len(words)==6:
                word1, word2 = ' '.join(words[0:2]), words[2]
            if word1 != None and word2!=None:
                for n, stage in enumerate(string_to_be_searched):
                    if word1 == stage and word2 == 'machine':
                        if len(words) == 5: time = float(words[3])
                        if len(words)==6: time = float(words[-2])
                        stages_time[n].append(time)

    mean_time = np.mean(stages_time, axis=1,dtype=int)
    print(list(zip(string_to_be_searched,mean_time)))

def PercentileIndices(data, p1, p2):
    data = np.asarray(data)
    p_low, p_high = sorted((p1, p2))

    v1, v2 = np.percentile(data, [p_low, p_high])
    return np.where((data >= v1) & (data <= v2))[0]

def Raster2Binary(spike_times, total_time, dt):
    T = int(np.ceil(total_time / dt))  # numero di bin temporali
    binary_array = np.zeros(T, dtype=int)

    bin_indices = (np.array(spike_times) / dt).astype(int)

    binary_array[bin_indices] = 1

    return binary_array

def FiringRate(spikes, t_start, t_stop, dt_s, dt_ds=0, nu_t_high=None, nu_t_low=None, remove_zeros=False, remove_pause=False):

    #prepare array
    if len(spikes) > 0:
        if isinstance(spikes, np.ndarray) and spikes.dtype != object:
            spikes = np.sort(np.ravel(spikes))
        elif isinstance(spikes, np.ndarray) and spikes.dtype == object:
            spikes = np.sort(np.hstack(spikes))
        elif isinstance(spikes, (list, tuple)):
            spikes = np.sort(np.hstack(spikes))
        else:
            spikes = np.sort(np.ravel(np.asarray(spikes)))
    else:
        spikes = np.array([])

    #conv params
    sample_rate = 1000 / dt_s # (Hz)
    if dt_ds > 0: downsample_rate = 1000 / dt_ds
    if nu_t_high != None: si_t_high = sample_rate / (2 * np.pi * nu_t_high) #nu_t (Hz)
    if nu_t_low != None: si_t_low = downsample_rate / (2 * np.pi * nu_t_low)

    if isinstance(remove_pause, dict):
        n_imgs, t_img, t_pause = remove_pause['n_imgs'], remove_pause['t_img'], remove_pause['t_pause']
        spikes_cut = np.concatenate([spikes[np.searchsorted(spikes, t_start + t_pause + nimg * (t_img + t_pause), side='left'):
                                          np.searchsorted(spikes, t_start + t_pause + nimg * (t_img + t_pause) + t_img, side='right')] - (nimg + 1) * t_pause
                                     for nimg in range(n_imgs)]) if len(spikes) > 0 else np.array([])
        t_stop = t_stop - n_imgs * t_pause - t_pause
        spikes = spikes_cut[:]

    #data structure
    n_steps_s = round((t_stop - t_start) / dt_s)
    sampling = t_start + np.arange(n_steps_s) * dt_s
    t1_s, t2_s = sampling - dt_s / 2, sampling + dt_s / 2
    fr_bin_s = np.searchsorted(spikes, t2_s, side='right') - np.searchsorted(spikes, t1_s, side='left')

    if dt_ds > 0:
        n_steps_ds = round((t_stop - t_start) / dt_ds)
        i_ds = np.clip((np.arange(n_steps_ds + 1) * (dt_ds / dt_s)).astype(int), 0, len(fr_bin_s))

    fr = np.array(fr_bin_s, dtype=float)

    # convolution
    if nu_t_high != None: fr = signal.convolve(fr, WindowGauss(si_t_high), 'same')

    #downsampling fr
    if dt_ds > 0:
        fr_cumsum = np.concatenate(([0], np.cumsum(fr)))
        fr = fr_cumsum[i_ds[1:]] - fr_cumsum[i_ds[:-1]]

    if nu_t_low != None: fr = signal.convolve(fr, WindowGauss(si_t_low), 'same')

    # remove zeros
    if remove_zeros == True: fr = fr[(fr > 0)]

    if dt_ds > 0:
        fr = np.array(fr) * (1000 / dt_ds)
    else:
        fr = np.array(fr) * (1000 / dt_s)

    return fr

def FRCachePath(fr_cache_path, area, stage_id, substage, dt_s_fr, dt_ds_fr):
    fname = f'fr_{area}_{stage_id}_{substage}_dt{int(dt_s_fr)}_ds{int(dt_ds_fr)}.npy'
    return os.path.join(fr_cache_path, fname)

def PopulationFRCachePath(fr_cache_path, area, stage_id, substage, dt_s_pop, dt_ds_pop):
    fname = f'fr_pop_{area}_{stage_id}_{substage}_dt{int(dt_s_pop)}_ds{int(dt_ds_pop)}.npy'
    return os.path.join(fr_cache_path, fname)

def LoadOrComputeFR(spikes, area, stage_id, substage, tstart, tstop, fr_cache_path,
                    dt_s_fr, dt_ds_fr, nu_t_high_fr, nu_t_low_fr):
    filepath = FRCachePath(fr_cache_path, area, stage_id, substage, dt_s_fr, dt_ds_fr)
    meta = {
        'tstart': tstart,
        'tstop': tstop,
        'dt_s_fr': dt_s_fr,
        'dt_ds_fr': dt_ds_fr,
        'nu_t_high_fr': nu_t_high_fr,
        'nu_t_low_fr': nu_t_low_fr,
        'remove_zeros': False,
        'remove_pause': False
    }

    if os.path.exists(filepath):
        cache = np.load(filepath, allow_pickle=True).item()
        if cache.get('meta', {}) == meta:
            return cache['fr']

    fr = np.array([
        FiringRate(
            spikes_neur,
            tstart,
            tstop,
            dt_s_fr,
            dt_ds_fr,
            nu_t_high_fr,
            nu_t_low_fr,
            remove_zeros=False,
            remove_pause=False
        )
        for spikes_neur in spikes
    ])
    np.save(filepath, {'fr': fr, 'meta': meta}, allow_pickle=True)
    return fr

def LoadOrComputePopulationFR(spikes, area, stage_id, substage, tstart, tstop, fr_cache_path,
                              dt_s_pop, dt_ds_pop, nu_t_high_pop, nu_t_low_pop):
    filepath = PopulationFRCachePath(fr_cache_path, area, stage_id, substage, dt_s_pop, dt_ds_pop)
    meta = {
        'tstart': tstart,
        'tstop': tstop,
        'dt_s_pop': dt_s_pop,
        'dt_ds_pop': dt_ds_pop,
        'nu_t_high_pop': nu_t_high_pop,
        'nu_t_low_pop': nu_t_low_pop,
        'remove_zeros': False,
        'remove_pause': False,
        'population_mode': 'all_spikes'
    }

    if os.path.exists(filepath):
        cache = np.load(filepath, allow_pickle=True).item()
        if cache.get('meta', {}) == meta:
            return cache['fr']

    fr = FiringRate(
        spikes,
        tstart,
        tstop,
        dt_s_pop,
        dt_ds_pop,
        nu_t_high_pop,
        nu_t_low_pop,
        remove_zeros=False,
        remove_pause=False
    )
    np.save(filepath, {'fr': fr, 'meta': meta}, allow_pickle=True)
    return fr

def MovingAverage(x, n_bins):
    if n_bins <= 1:
        return x.copy()
    kernel = np.ones(n_bins) / n_bins
    return np.convolve(x, kernel, mode='same')

def MapStatesToReferenceBins(states, ref_tstart, ref_dt, n_ref_bins, prefix='react', eps=1e-12):
    for state in states:
        istart_ref = int(np.floor((state['tstart'] - ref_tstart) / ref_dt + eps))
        istop_ref = int(np.ceil((state['tstop'] - ref_tstart) / ref_dt - eps))
        ipeak_ref = int(np.round((state['tpeak'] - ref_tstart) / ref_dt))

        istart_ref = int(np.clip(istart_ref, 0, n_ref_bins))
        istop_ref = int(np.clip(istop_ref, 0, n_ref_bins))
        if istop_ref <= istart_ref and istart_ref < n_ref_bins:
            istop_ref = min(istart_ref + 1, n_ref_bins)
        ipeak_ref = int(np.clip(ipeak_ref, istart_ref, max(istop_ref - 1, istart_ref)))

        state[f'istart_{prefix}'] = istart_ref
        state[f'istop_{prefix}'] = istop_ref
        state[f'ipeak_{prefix}'] = ipeak_ref

    return states

def DetectUpStates(pop_signal, tstart, dt, up_state_cfg, n_neurons):
    smooth_bins = max(int(round(up_state_cfg['smooth_ms'] / dt)), 1)
    min_up_bins = max(int(round(up_state_cfg['min_up_duration_ms'] / dt)), 1)
    threshold_hz = up_state_cfg['threshold_hz']

    pop_signal = np.asarray(pop_signal, dtype=float)
    pop_rate_raw = pop_signal.ravel() / n_neurons
    pop_rate_smooth = MovingAverage(pop_rate_raw, smooth_bins)
    is_up = pop_rate_smooth >= threshold_hz

    i = 0
    while i < len(is_up):
        if not is_up[i]:
            i = i + 1
            continue
        istart = i
        while i < len(is_up) and is_up[i]:
            i = i + 1
        istop = i
        if istop - istart < min_up_bins:
            is_up[istart:istop] = False

    states = []
    i = 0
    up_state_id = 0

    while i < len(is_up):
        if not is_up[i]:
            i = i + 1
            continue
        istart = i
        while i < len(is_up) and is_up[i]:
            i = i + 1
        istop = i

        ipeak = istart + np.argmax(pop_rate_smooth[istart:istop])
        states.append({
            'id': up_state_id,
            'istart': istart,
            'istop': istop,
            'ipeak': ipeak,
            'tstart': tstart + istart * dt,
            'tstop': tstart + istop * dt,
            'tpeak': tstart + ipeak * dt,
            'duration': (istop - istart) * dt
        })
        up_state_id = up_state_id + 1

    return {
        'states': states,
        'is_up': is_up,
    }

def CosineSimilarity(x, y, eps=1e-12):
    x = np.asarray(x)
    y = np.asarray(y)
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)
    x = x / (x_norm + eps)
    y = y / (y_norm + eps)
    return x @ y.T

def KDE1D(x,bin_width,xrange=None,log=False):

    print('\nKDE Univariate')

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


    print('\nxmin: %lg , xmax: %lg , bin width: %lg' %(xmin,xmax,bin_width))

    bins = int((xmax - xmin) / bin_width)


    xx = np.linspace(xmin,xmax,bins) if log == False else np.logspace(np.log10(xmin),np.log10(xmax),bins)

    kde_uni = KDEUnivariate(x)
    dens = kde_uni.fit(kernel='gau',bw='normal_reference')
    print('\nbandwith found: %s' % (dens.bw))
    p = dens.evaluate(xx)

    return xx, p / np.sum(p)

def Histogram(data, nbins=None, dx=None ,xmin=None, xmax=None, cumulative=False, logx=False, KDE=False, norm=True):

    data = np.asarray(data).ravel()

    if data.size == 0:
        return [], []

    data = data[np.isfinite(data)]
    if data.size == 0:
        return [], []

    if logx == True:
        data = data[(data > 0)]
        if data.size == 0:
            return [], []

    if xmin is None: xmin = np.min(data)
    if xmax is None: xmax = np.max(data)

    if logx == True and (xmin <= 0 or xmax <= 0):
        raise ValueError('\nFor log histogram xmin and xmax must be > 0')
    if xmin > xmax:
        raise ValueError('\nxmin must be <= xmax')

    if nbins is None and dx is None:
        raise ValueError('\nPlease, provide either the number of bins or the precision to be used')
    if nbins is None:
        dx = float(dx)
        if dx <= 0:
            raise ValueError('\ndx must be > 0')
        nbins = max(1, int(np.ceil((xmax - xmin) / dx)))
    else:
        nbins = int(nbins)
        if nbins <= 0:
            raise ValueError('\nnbins must be > 0')

    data = data[(data >= xmin) & (data <= xmax)]

    if xmin == xmax:
        bin = np.array([xmin])
        hist_plot = np.array([1.0 if norm == True else len(data)])
        return bin, hist_plot

    if logx == False:
        bins = np.linspace(xmin, xmax, nbins + 1)
    elif logx == True:
        bins = np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)

    bin = bins[:-1]
    hist = np.histogram(data, bins=bins, density=norm)[0]

    if cumulative == True:
        hist = np.cumsum(hist * np.diff(bins)) if norm == True else np.cumsum(hist)
        hist_plot = np.concatenate(([0], hist[:-1]))
    elif cumulative == False:
        hist_plot = hist
        if KDE == True:
            bin_width = (xmax - xmin) / nbins
            bin, hist_plot = KDE1D(x=data, bin_width=bin_width, xrange=[xmin, xmax], log=logx)

    return bin, hist_plot

def LoadWeights(syn_path, syn_type='exc_inh', reshape=None):

    data = None
    if isinstance(syn_path, dict):
        data = syn_path[syn_type]
    else:
        try:
            data = np.load(syn_path, allow_pickle=True).item()[syn_type]
        except FileNotFoundError:
            print(f"\nPath doesn't exists!\n{syn_path} - {syn_type}")

    if data != None:
        w = data['weight']
        if reshape != None:
            if len(w) < np.prod(reshape):
                n_neurons = reshape[0]
                w = np.asarray(w)
                w_new = np.full(n_neurons * n_neurons, np.nan, dtype=np.result_type(w.dtype, float))
                w_new[np.arange(n_neurons * n_neurons) % (n_neurons + 1) != 0] = w
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

def kde2D_multivariate(x,y,xbin_width,ybin_width,xrange=None, yrange=None,bw='normal_reference'):
    print('\nKDE Multivariate')
    # create grid of sample locations (default: 100x100)

    if xrange != None:
        xmin, xmax = xrange
    else:
        xmin, xmax = x.min(), x.max()
    if yrange != None:
        ymin, ymax = yrange
    else:
        ymin,yxmax = y.min(), y.max()

    xbins = int((xmax - xmin) / xbin_width)
    xbins = complex(0, xbins)


    ybins = int((ymax - ymin) / ybin_width)
    ybins = complex(0, ybins)

    xx, yy = np.mgrid[xmin:xmax:xbins,
                      ymin:ymax:ybins]

    xy_sample = np.vstack([xx.ravel(), yy.ravel()])
    xy_values  = np.vstack([x, y])

    kde_multi = KDEMultivariate(data=xy_values,var_type='cc', bw=bw)

    print('\nbandwith found: %s' %(kde_multi.bw))

    # score_samples() returns the log-likelihood of the samples
    zz = np.reshape(kde_multi.pdf(xy_sample).T, xx.shape)
    return xx, yy, zz / np.sum(zz)


def GetDatasetFeatures(model, dataset_name, root_input_path):


    dataset_path = os.path.join(root_input_path, model)

    features = np.load(os.path.join(dataset_path,f'{dataset_name}_features.npy'))
    labels = np.load(os.path.join(dataset_path,f'{dataset_name}_labels.npy'))

    return features, labels

def DatasetClassSample(features, labels, max_samples_per_class, seed=42):
    rng = np.random.default_rng(seed)
    features, labels = np.asarray(features), np.asarray(labels)
    classes = np.unique(labels)
    indices = []

    for nclass in classes:
        class_indices = np.where(labels == nclass)[0]
        if max_samples_per_class is not None and len(class_indices) > max_samples_per_class:
            class_indices = rng.choice(class_indices, size=max_samples_per_class, replace=False)
        indices.append(class_indices)

    indices = np.concatenate(indices)
    rng.shuffle(indices)
    return features[indices].astype(np.float32), labels[indices]

def DatasetSimilarityHistogram(features, labels, norm_factor=81, mode='intra'):
    features, labels = np.asarray(features, dtype=np.float32), np.asarray(labels)
    classes = np.unique(labels)
    bins = np.arange(norm_factor, dtype=float) / norm_factor
    dx = 1 / norm_factor
    densities = []

    for nclass, label_1 in enumerate(classes):
        features_1 = features[labels == label_1]
        if mode == 'intra':
            if len(features_1) < 2:
                continue
            sim = features_1 @ features_1.T
            sim = sim[np.triu_indices(len(features_1), k=1)]
            idx = np.clip(np.rint(sim).astype(int), 0, norm_factor - 1)
            density = np.bincount(idx, minlength=norm_factor)[:norm_factor].astype(float)
            densities.append(density / (np.sum(density) * dx))
        elif mode == 'inter':
            for label_2 in classes[nclass + 1:]:
                sim = features_1 @ features[labels == label_2].T
                idx = np.clip(np.rint(sim).astype(int), 0, norm_factor - 1)
                density = np.bincount(idx.ravel(), minlength=norm_factor)[:norm_factor].astype(float)
                densities.append(density / (np.sum(density) * dx))
        else:
            raise ValueError("mode must be 'intra' or 'inter'")

    densities = np.asarray(densities)
    if len(densities) == 0:
        return {'bins': bins, 'density_mean': np.zeros(norm_factor), 'density_sem': np.zeros(norm_factor)}
    return {'bins': bins, 'density_mean': np.mean(densities, axis=0), 'density_sem': np.std(densities, axis=0) / np.sqrt(len(densities))}

def DatasetSimilarityMatrix(features, labels, norm_factor=81):
    features, labels = np.asarray(features, dtype=np.float32), np.asarray(labels)
    classes = np.unique(labels)
    class_features = [features[labels == nclass] for nclass in classes]
    sim_matrix = np.zeros((len(classes), len(classes)), dtype=np.float32)
    counts_matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)

    for i, features_1 in enumerate(class_features):
        for j, features_2 in enumerate(class_features):
            if i == j:
                n_samples = len(features_1)
                counts_matrix[i, j] = n_samples * (n_samples - 1) // 2
                if counts_matrix[i, j] > 0:
                    sim = features_1 @ features_1.T
                    sim_matrix[i, j] = (np.sum(sim) - np.trace(sim)) / (2 * counts_matrix[i, j] * norm_factor)
            else:
                counts_matrix[i, j] = len(features_1) * len(features_2)
                if counts_matrix[i, j] > 0:
                    sim_matrix[i, j] = np.mean(features_1 @ features_2.T) / norm_factor

    return {'sim_matrix': sim_matrix, 'counts_matrix': counts_matrix}

def DatasetUmap(features, labels, n_components=2, max_samples=6000, seed=42, n_neighbors=15, min_dist=0.1):
    try:
        import umap
    except ImportError as exc:
        raise ImportError("Please install umap-learn to generate dataset_umap_2d.npy") from exc

    rng = np.random.default_rng(seed)
    features, labels = np.asarray(features, dtype=np.float32), np.asarray(labels)
    if len(labels) > max_samples:
        indices = rng.choice(np.arange(len(labels)), size=max_samples, replace=False)
        features, labels = features[indices], labels[indices]

    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric='euclidean',
                        random_state=seed)
    return {'X_umap': reducer.fit_transform(features), 'classes': np.unique(labels), 'y_plot': labels}

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
        w_cx_cx_sum['non-specific'].append(np.sum(flatten_matrix[cx_cx_index[neur]['non-specific']]) / n_class)

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

def SynCategoriesMask(conn_type,th_features=None, labels=None, net_params=None, n_cycles=None):

    n_areas = net_params['n_areas']
    n_class = net_params['n_class']
    if n_cycles is None:
        n_cycles = net_params['n_cycles'] if 'n_cycles' in net_params else 1
    autapses = net_params['autapses'] if 'autapses' in net_params else True
    n_ranks = net_params['n_ranks_training'] if 'n_ranks_training' in net_params else net_params['n_ranks']
    n_exc_ca = net_params['n_neur_group'] if 'n_neur_group' in net_params else net_params['n_exc_ca']

    if conn_type == 'cx_cx':

        group_step = n_class * n_ranks * n_cycles
        n_groups = n_areas * group_step
        n_neurons = n_groups * n_exc_ca
        if isinstance(labels, str) and labels == 'same':
            labels = np.array([group % (n_class * n_ranks) // n_ranks for group in range(n_groups)], dtype=int)
        else:
            labels = np.asarray(labels, dtype=int)[:n_groups]

        neurons = np.arange(n_neurons, dtype=int)
        neuron_groups = neurons // n_exc_ca
        neuron_classes = labels[neuron_groups]
        neuron_areas = neuron_groups // group_step

        same_group = neuron_groups[:, None] == neuron_groups[None, :]
        same_class = neuron_classes[:, None] == neuron_classes[None, :]
        same_area = neuron_areas[:, None] == neuron_areas[None, :]
        paired_groups = np.abs(neuron_groups[:, None] - neuron_groups[None, :]) == group_step
        if autapses:
            mask_group = (same_area & same_group) | (~same_area & paired_groups)
        else:
            no_self = neurons[:, None] != neurons[None, :]
            mask_group = (same_area & same_group & no_self) | (~same_area & paired_groups)
        mask_class = (same_area & same_class & ~same_group) | (~same_area & same_class & ~paired_groups)
        mask_non_specific = ~same_class

        return {'group': mask_group, 'class': mask_class, 'non-specific': mask_non_specific}

    elif conn_type == 'cx_th':

        n_groups_cx = n_areas * n_class * n_ranks * n_cycles
        n_neur_cx = n_groups_cx * n_exc_ca
        n_neur_th = net_params['n_neur_th'] if 'n_neur_th' in net_params else np.shape(th_features)[-1]
        labels = np.asarray(labels, dtype=int)[:n_groups_cx]
        th_features = np.asarray(th_features)[:n_groups_cx]

        mask_th_group = th_features.astype(bool)
        mask_th_classes = np.zeros((n_class, n_neur_th), dtype=bool)
        np.logical_or.at(mask_th_classes, labels, mask_th_group)

        mask_th_class = mask_th_classes[labels] & ~mask_th_group
        mask_th_non_specific = ~mask_th_group & ~mask_th_class

        cx_groups = np.arange(n_neur_cx, dtype=int) // n_exc_ca
        mask_group = mask_th_group[cx_groups]
        mask_class = mask_th_class[cx_groups]
        mask_non_specific = mask_th_non_specific[cx_groups]

        return {'group': mask_group, 'class': mask_class, 'non-specific': mask_non_specific}

    elif conn_type == 'th_cx':

        n_groups_cx = n_areas * n_class * n_ranks * n_cycles
        n_neur_cx = n_groups_cx * n_exc_ca
        n_neur_th = net_params['n_neur_th'] if 'n_neur_th' in net_params else np.shape(th_features)[-1]
        labels = np.asarray(labels, dtype=int)[:n_groups_cx]
        th_features = np.asarray(th_features)[:n_groups_cx]

        mask_th_group = th_features.astype(bool)
        mask_th_classes = np.zeros((n_class, n_neur_th), dtype=bool)
        np.logical_or.at(mask_th_classes, labels, mask_th_group)

        mask_th_class = mask_th_classes[labels] & ~mask_th_group
        mask_th_non_specific = ~mask_th_group & ~mask_th_class

        cx_groups = np.arange(n_neur_cx, dtype=int) // n_exc_ca
        mask_group = np.transpose(mask_th_group[cx_groups])
        mask_class = np.transpose(mask_th_class[cx_groups])
        mask_non_specific = np.transpose(mask_th_non_specific[cx_groups])

        return {'group': mask_group, 'class': mask_class, 'non-specific': mask_non_specific}
    else:
        raise ValueError(f"Unknown conn_type '{conn_type}'")

def SynCategoriesMatrix(w_matrix, mask_cat_dict, cat='all'):

    syn_cat_dict = {'group':[], 'class': [], 'non-specific': []}

    for ntrial, matrix in enumerate(w_matrix):
        indices = mask_cat_dict[ntrial]
        if np.shape(indices['group']) != np.shape(matrix):
            indices = {name: np.reshape(mask, np.shape(matrix)) for name, mask in indices.items()}

        if cat == 'group' or cat == 'all':
            syn_cat_dict['group'].append(matrix[indices['group']])
        if cat == 'class' or cat == 'all':
            syn_cat_dict['class'].append(matrix[indices['class']])
        if cat == 'non-specific' or cat == 'all':
            syn_cat_dict['non-specific'].append(matrix[indices['non-specific']])

    return syn_cat_dict if cat == 'all' else syn_cat_dict[cat]

def SpikesCount(spikes_trial, stage,  params, substage='classification', nconf=None):

    net_params, times  = params['network'], params['times']
    n_class, n_ranks_test = net_params['n_class'], net_params['n_ranks_test'],
    n_img_test, t_img_test = n_ranks_test * n_class, net_params['t_img_test']


    time = times[stage][substage]
    t_start, t_stop = time['start'], time['stop']
    if nconf != None:
        t_start, t_stop = time['start'][nconf], time['stop'][nconf]

    if substage == 'classification':
        scale = 1000 / (n_img_test*t_img_test)
    else:
        scale = 1000 / (t_stop - t_start)

    spikes_neuron = np.array([spikes_neur[bisect.bisect_left(spikes_neur, t_start): bisect.bisect_right(spikes_neur, t_stop)] for spikes_neur in spikes_trial], dtype=object)
    spikes_count_neuron = scale * np.array([len(spikes) for spikes in spikes_neuron])

    return spikes_count_neuron

def SpikesCountClassification(spikes_trial, stage, params, nconf=None):

    net_params, times = params['network'], params['times']
    n_class, n_ranks_test = net_params['n_class'], net_params['n_ranks_test']
    t_img_test, t_pause = net_params['t_img_test'], net_params['t_pause']
    n_img_test, t_img = n_ranks_test * n_class, t_img_test + t_pause

    time = times[stage]['classification']
    t_start = time['start']
    if nconf != None:
        t_start = time['start'][nconf]

    t_stim_start = t_start + t_pause + np.arange(n_img_test) * t_img
    t_stim_stop = t_stim_start + t_img_test

    spikes_count_stim = np.array([np.searchsorted(np.asarray(spikes_neur), t_stim_stop, side='right') -
                                  np.searchsorted(np.asarray(spikes_neur), t_stim_start, side='left')
                                  for spikes_neur in spikes_trial])

    return spikes_count_stim * 1000 / t_img_test

def NetAccuracyOld(spikes_trial, trial_dict_path, stage,  params, n_cycles, prediction='neuron', nconf=None, trial_dict=None):

    if type(prediction) == str(): prediction = [prediction]

    net_params, times  = params['network'], params['times']

    n_areas, n_class,n_cycles_tot, n_ranks_train, n_ranks_test, n_exc_ca, t_img_test, t_pause = (net_params['n_areas'], net_params['n_class'], net_params['n_cycles'],
                                net_params['n_ranks_train'], net_params['n_ranks_test'], net_params['n_exc_ca'], net_params['t_img_test'],
                                net_params['t_pause'])

    n_groups, n_img_test, t_img = n_areas * n_class * n_ranks_train, n_ranks_test * n_class, t_img_test + t_pause
    n_groups_area, n_groups_cycle = n_groups // n_areas, n_groups * n_cycles
    n_neurons = n_groups * n_exc_ca

    time = times[stage]['classification']
    t_start, t_stop = time['start'], time['stop']

    if nconf != None:
        t_start, t_stop = time['start'][nconf], time['stop'][nconf]

    if trial_dict is None:
        lp = SearchFiles(trial_dict_path, 'trial_dict.npy')[0]
        trial_dict = np.load(lp, allow_pickle=True, encoding='latin1').item()

    labels_train = trial_dict['training']['labels']
    labels_train_res = np.reshape(labels_train, (-1,n_groups_area))
    labels_test = trial_dict['test']['labels']

    spikes_neuron = np.array([spikes_neur[bisect.bisect_left(spikes_neur, t_start): bisect.bisect_right(spikes_neur, t_stop)] for spikes_neur in spikes_trial], dtype=object)
    #spikes_group = np.array([[np.sort(np.hstack(spikes_neuron[group * n_exc_ca:(group + 1) * n_exc_ca])) if len(spikes_neuron[group * n_exc_ca:(group + 1) * n_exc_ca])>0 else np.array([]) for group in range(area*n_groups_area,(area+1)*n_groups_area)] for area in range(n_areas)],dtype=object)
    spikes_group = np.array([np.sort(np.hstack(spikes_neuron[ngroup * n_exc_ca:(ngroup + 1) * n_exc_ca])) if len(spikes_neuron[ngroup * n_exc_ca:(ngroup + 1) * n_exc_ca])>0
                             else np.array([]) for ngroup in range(n_groups_cycle)],dtype=object)
    spikes_group = np.reshape(spikes_group, (n_cycles, n_areas, n_groups_area)) if len(spikes_group.flatten())>0 else np.array([[[[] for ngroup in range(n_groups_area)] for narea in range(n_areas)] for ncycle in range(n_cycles)])


    accuracy = {pred: 0 for pred in prediction}

    for img in range(n_img_test):

        label = labels_test[img]
        t0 = t_start + t_pause + img * t_img
        t1 = t0 + t_img_test

        spikes_neuron_img = np.array([CountSpikes(spikes_neur, t0, t1) for spikes_neur in spikes_neuron])
        spikes_group_img = np.array([[[CountSpikes(spikes_group[ncycle, narea, ngroup], t0, t1) for ngroup in range(n_groups_area)] for narea in range(n_areas)] for ncycle in range(n_cycles)])
        spikes_class_img = np.array([[[np.sum(spikes_group_img[ncycle, narea][(labels_train_res[ncycle] == nclass)]) for nclass in range(n_class)] for narea in range(n_areas)] for ncycle in range(n_cycles)])

        winner_neur = np.argmax(spikes_neuron_img)
        winner_group = np.argmax(np.sum(spikes_group_img, axis=1).flatten())
        winner_class = np.argmax(np.sum(np.sum(spikes_class_img,axis=1),axis=0))

        if 'neuron' in prediction:
            pred_group = winner_neur // n_exc_ca
            pred_class = labels_train[pred_group]
            if pred_class == label: accuracy['neuron'] = accuracy['neuron'] + 1
        if 'group' in prediction:
            pred_group = winner_group
            pred_class = labels_train[pred_group]
            if pred_class == label: accuracy['group'] = accuracy['group'] + 1
        if 'class' in prediction:
            pred_class = winner_class
            if pred_class == label: accuracy['class'] = accuracy['class'] + 1

    for pred in prediction: accuracy[pred] = 100 * accuracy[pred] / n_img_test

    return accuracy

def NetAccuracy(spikes_trial, trial_dict_path, stage, params, n_cycles, prediction='neuron', nconf=None, trial_dict=None):

    if type(prediction) == str: prediction = [prediction]

    net_params, times = params['network'], params['times']

    n_areas, n_class, n_ranks_train, n_ranks_test, n_exc_ca, t_img_test, t_pause = (net_params['n_areas'], net_params['n_class'],
                                                                                      net_params['n_ranks_train'], net_params['n_ranks_test'],
                                                                                      net_params['n_exc_ca'], net_params['t_img_test'],
                                                                                      net_params['t_pause'])

    n_groups, n_img_test, t_img = n_areas * n_class * n_ranks_train, n_ranks_test * n_class, t_img_test + t_pause
    n_groups_area = n_groups // n_areas

    time = times[stage]['classification']
    t_start, t_stop = time['start'], time['stop']

    if nconf != None:
        t_start, t_stop = time['start'][nconf], time['stop'][nconf]

    if trial_dict is None:
        lp = SearchFiles(trial_dict_path, 'trial_dict.npy')[0]
        trial_dict = np.load(lp, allow_pickle=True, encoding='latin1').item()

    labels_train = np.asarray(trial_dict['training']['labels'])[:n_cycles * n_groups_area]
    labels_train_res = np.reshape(labels_train, (n_cycles, n_groups_area))
    labels_test = np.asarray(trial_dict['test']['labels'])

    spikes_neuron = np.array([spikes_neur[bisect.bisect_left(spikes_neur, t_start): bisect.bisect_right(spikes_neur, t_stop)] for spikes_neur in spikes_trial], dtype=object)
    t_stim_start = t_start + t_pause + np.arange(n_img_test) * t_img
    t_stim_stop = t_stim_start + t_img_test
    spikes_neuron_img = np.array([np.searchsorted(np.asarray(spikes_neur), t_stim_stop, side='right') -
                                  np.searchsorted(np.asarray(spikes_neur), t_stim_start, side='left')
                                  for spikes_neur in spikes_neuron])

    accuracy = {pred: 0 for pred in prediction}

    if 'neuron' in prediction:
        winner_neur = np.argmax(spikes_neuron_img, axis=0)
        pred_group = winner_neur // n_exc_ca
        accuracy['neuron'] = 100 * np.sum(labels_train[pred_group] == labels_test) / n_img_test

    if 'group' in prediction or 'class' in prediction:
        spikes_group_img = np.sum(np.reshape(spikes_neuron_img, (n_cycles, n_areas, n_groups_area, n_exc_ca, n_img_test)), axis=3)
        spikes_group_img = np.sum(spikes_group_img, axis=1)

        if 'group' in prediction:
            winner_group = np.argmax(np.reshape(spikes_group_img, (n_cycles * n_groups_area, n_img_test)), axis=0)
            accuracy['group'] = 100 * np.sum(labels_train[winner_group] == labels_test) / n_img_test

        if 'class' in prediction:
            spikes_class_img = np.einsum('cgi,cgk->cki', spikes_group_img, labels_train_res[:, :, None] == np.arange(n_class)[None, None, :])
            winner_class = np.argmax(np.sum(spikes_class_img, axis=0), axis=0)
            accuracy['class'] = 100 * np.sum(winner_class == labels_test) / n_img_test

    return accuracy

def KNNMonoarea(trial_dict_path, params, params_dataset, n_neighbours=1):

    network, dataset = params['network'], params_dataset

    n_areas, n_class, n_ranks_train, n_ranks_test = network['n_areas'], network['n_class'], network['n_ranks_train'], network['n_ranks_test']

    features_train, labels_train = dataset['train_features'], dataset['train_labels']
    features_test, labels_test = dataset['test_features'], dataset['test_labels']
    balanced_train, balanced_test = dataset['balanced_train'], dataset['balanced_test']

    n_feat = len(features_train[0]) // 2
    confusion_matrix = np.zeros((n_class, n_class))

    trial_dict = np.load(os.path.join(trial_dict_path, 'trial_dict.npy'), allow_pickle=True, encoding='latin1').item()
    index_training_trial = trial_dict['training']['index mnist']
    index_test_trial = trial_dict['test']['index mnist']
    index_test_shuffle_trial = trial_dict['test']['index shuffling']

    X_train = np.array([features_train[index] for index in index_training_trial])
    Y_train = np.array([labels_train[index] for index in index_training_trial])

    # TEST
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


def FrCorrelation(spikes, params, stage, dt_s, dt_ds, nu_t_high, nu_t_low, nconf=None, substage='classification', spikes_2=None):
    params_net, params_times = params['network'], params['times']
    rem_pause = params_net['rem_pause'][substage]

    tstart, tstop = params_times[stage][substage]['start'], params_times[stage][substage]['stop']
    if nconf != None:
        tstart = tstart[nconf]
        tstop = tstop[nconf]

    t_corr = rem_pause['n_imgs'] * rem_pause['t_img'] if rem_pause else tstop - tstart
    n_steps_s = round(t_corr / dt_s)

    activities = []
    for spikes_pop in [spikes] if spikes_2 is None else [spikes, spikes_2]:
        n_units = len(spikes_pop)
        activity = np.zeros((n_units, n_steps_s), dtype=np.float32)
        spikes_units = []
        for nunit in range(n_units):
            spikes_unit = np.asarray(spikes_pop[nunit])
            spikes_unit = Mask(spikes_unit, tstart, tstop) - tstart if len(spikes_unit) > 0 else np.array([])
            if rem_pause and len(spikes_unit) > 0:
                img_id = np.floor_divide(spikes_unit - rem_pause['t_pause'], rem_pause['t_img'] + rem_pause['t_pause']).astype(int)
                spikes_rel = spikes_unit - rem_pause['t_pause'] - img_id * (rem_pause['t_img'] + rem_pause['t_pause'])
                mask = (img_id >= 0) & (img_id < rem_pause['n_imgs']) & (spikes_rel >= 0) & (spikes_rel <= rem_pause['t_img'])
                spikes_unit = img_id[mask] * rem_pause['t_img'] + spikes_rel[mask]
            spikes_units.append(spikes_unit)

        spikes_len = np.array([len(spikes_unit) for spikes_unit in spikes_units])
        if n_steps_s > 0 and np.sum(spikes_len) > 0:
            spikes_all = np.concatenate(spikes_units)
            units_all = np.repeat(np.arange(n_units), spikes_len)
            bins_all = np.floor((spikes_all + dt_s / 2) / dt_s).astype(int)
            mask = (bins_all >= 0) & (bins_all < n_steps_s)
            np.add.at(activity, (units_all[mask], bins_all[mask]), 1)

        sample_rate = 1000 / dt_s
        if nu_t_high != None:
            activity = signal.convolve(activity, WindowGauss(sample_rate / (2 * np.pi * nu_t_high))[None, :], 'same')
        if dt_ds > 0:
            n_steps_ds = round(t_corr / dt_ds)
            i_ds = np.clip((np.arange(n_steps_ds + 1) * (dt_ds / dt_s)).astype(int), 0, activity.shape[1])
            activity_cumsum = np.concatenate((np.zeros((n_units, 1), dtype=activity.dtype), np.cumsum(activity, axis=1)), axis=1)
            activity = activity_cumsum[:, i_ds[1:]] - activity_cumsum[:, i_ds[:-1]]
        if nu_t_low != None:
            activity = signal.convolve(activity, WindowGauss((1000 / (dt_ds if dt_ds > 0 else dt_s)) / (2 * np.pi * nu_t_low))[None, :], 'same')
        activity = activity * (1000 / dt_ds) if dt_ds > 0 else activity * (1000 / dt_s)
        activities.append(activity)

    activity_1, activity_2 = activities[0], activities[-1]
    activity_1_mu, activity_2_mu = np.mean(activity_1, axis=1, keepdims=True), np.mean(activity_2, axis=1, keepdims=True)
    activity_1_std, activity_2_std = np.std(activity_1, axis=1), np.std(activity_2, axis=1)
    activity_1_centered, activity_2_centered = activity_1 - activity_1_mu, activity_2 - activity_2_mu
    corr_matrix = np.zeros((activity_1.shape[0], activity_2.shape[0]), dtype=np.float32)
    denom = np.outer(activity_1_std, activity_2_std) * activity_1.shape[1]
    np.divide(activity_1_centered @ np.transpose(activity_2_centered), denom, out=corr_matrix, where=(denom != 0))

    return corr_matrix

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

def Joint2D(x, y, parameters, syn_out_neur_dict=None):

    print('\nCalculating joint probability distribution (neuron fr vs neuron synaptic output')

    par = parameters

    stages = ['pre-sleep', 'post-sleep']
    areas, n_neurons, n_groups, n_exc_ca = par['n_areas'], par['n_neurons'], par['n_groups'], par['n_exc_ca']
    xrange, yrange = par['xrange_jpdf'], par['yrange_jpdf']
    xbin_width, ybin_width = par['bin_width_jpdf']

    set_min, set_max = par['sets_syn']
    sets = np.arange(set_min, set_max+1, 1)
    n_sets = len(sets)

    n_syn_intra, n_syn_inter = n_exc_ca - 1, n_exc_ca#n_exc_ca * (n_exc_ca - 1), n_exc_ca**2

    joint_dict = {'pre-sleep': np.array([]), 'post-sleep': np.array([])}
    if syn_out_neur_dict==None: syn_out_neur_dict = {'pre-sleep': {}, 'post-sleep': {}}

    for stage in stages:

        print('\nStage: %s', stage)

        w_stage = weights_cat_dict[stage]

        for n_set in sets:
            set, id_set = n_set, n_set - set_min
            set_exist = set in syn_out_neur_dict[stage].keys()

            if not set_exist:
                print('Set: %d (%d/%d)' % (set, id_set + 1, n_sets), end='\r')

                w_intra, w_inter = w_stage[set]['group'][0].copy(), w_stage[set]['group'][1].copy()

                w_intra, w_inter = np.reshape(w_intra, (n_neurons, n_syn_intra)), np.reshape(w_inter, (n_neurons, n_syn_inter))
                w_mean = np.mean(np.concatenate([w_intra, w_inter],axis=1),axis=1)

                syn_out_neur_dict[stage][set] = w_mean

        syn_out_neurons = np.hstack([np.hstack(syn_out_neur_dict[stage][set]) for set in sets])
        fr_neurons = np.hstack([np.hstack(fr_dict[stage][set]) for set in sets])

        datax ,datay = fr_neurons, syn_out_neurons

        print(len(datax), len(datay))
        print(np.min(datax), np.max(datax))
        print(np.min(datay), np.max(datay))

        joint = kde2D_multivariate(x=datax,y=datay,xbin_width=xbin_width,ybin_width=ybin_width,xrange=xrange,yrange=yrange,
                                   bw='cv_ml')

        joint_dict[stage] = joint

    return syn_out_neur_dict, joint_dict

def GroupsActivation(spikes_neurons, stage, substage, params, nconf, oscillations=False):

    net_params, times = params['network'], params['times']

    n_areas, n_class, n_ranks_train, n_exc_ca = net_params['n_areas'], net_params['n_class'], net_params['n_ranks_train'], net_params['n_exc_ca']
    dt, fr_thresh = net_params['dt_fr'], net_params['thresh_fr']
    n_groups = n_areas * n_class * n_ranks_train
    remove_pause = net_params['rem_pause']

    time = times[stage][substage]
    t_start, t_stop = time['start'][nconf], time['stop'][nconf]
    thresh_osc = net_params['thresh_osc']
    sigma_osc = net_params['sigma_t_osc']

    spikes_groups = [np.sort(np.hstack(spikes_neurons[n_exc_ca * ngroup:n_exc_ca * (ngroup + 1)])) for ngroup in range(n_groups)]
    fr_groups = np.array([FiringRate(spikes_group, t_start, t_stop, dt_s=dt, dt_ds=0, nu_t_high=None, nu_t_low=None, remove_pause=remove_pause[substage])
                          for spikes_group in spikes_groups]) / n_exc_ca
    groups_activation = np.transpose(np.array(fr_groups > fr_thresh, dtype=int))

    #from matplotlib import pyplot as plt
    #plt.figure()
    #plt.imshow(np.transpose(groups_activation), origin='lower', aspect='auto', cmap='Greys')
    #plt.xticks(np.arange(n_steps)[::20], np.arange(n_steps, dtype=int)[::20] * dt // 1000)
    #plt.savefig(f'/Users/mac/PycharmProjects/ConfPlots/Plots/NatureThaco1Revenge/osc_{substage}.png')

    if oscillations:
        #dt_osc = time['dt_osc']
        #n_sum = dt_osc // dt
        group_act_t = np.sum(groups_activation, axis=1)
        #group_act_sum_t = np.array([np.sum([np.max(seq) for seq in FindSequences(act_bin)]) for act_bin in  np.reshape(group_act_t, (-1, n_sum))])# * 1000 / dt_osc
        #group_act_sum_t = np.array([[np.max(seq) for seq in FindSequences(act_bin)] for act_bin in np.reshape(group_act_t, (-1, n_sum))])
        group_act_t_conv = signal.convolve(group_act_t, WindowGauss(sigma_osc), 'same')
        #plt.figure()
        #plt.plot(group_act_t_conv, color='black')
        #plt.xticks(np.arange(n_steps)[::20], np.arange(n_steps, dtype=int)[::20] *dt // 1000)
        #plt.savefig(f'/Users/mac/PycharmProjects/ConfPlots/Plots/NatureThaco1Revenge/osc_conv_{substage}.png')

        mask_down_states = np.array((group_act_t_conv < thresh_osc), dtype=int)
        #print(mask_down_states)
        time_interval_down_states = np.array([np.sum(seq) for seq in FindSequences(mask_down_states)]) * dt / 1000
        #print(down_states*dt/1000)
        return time_interval_down_states
    else:
        return groups_activation
