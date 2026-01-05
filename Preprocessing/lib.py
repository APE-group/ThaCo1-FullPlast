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

def CountSpikes(spikes, t1, t2):
    i_t1, i_t2 = bisect.bisect_left(spikes, t1), bisect.bisect_right(spikes, t2)
    n_spikes = i_t2 - i_t1
    return n_spikes

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

def FiringRate(spikes, t_start, t_stop, dt_s, dt_ds=0, nu_t_high=None, nu_t_low=None, remove_zeros=False, remove_pause=False):

    #prepare array
    if len(spikes) > 0:
        spikes = np.sort(np.hstack(spikes))

    #conv params
    sample_rate = 1000 / dt_s # (Hz)
    if dt_ds > 0: downsample_rate = 1000 / dt_ds
    if nu_t_high != None: si_t_high = sample_rate / (2 * np.pi * nu_t_high) #nu_t (Hz)
    if nu_t_low != None: si_t_low = downsample_rate / (2 * np.pi * nu_t_low)

    if isinstance(remove_pause, dict):
        n_imgs, t_img, t_pause = remove_pause['n_imgs'], remove_pause['t_img'], remove_pause['t_pause']
        spikes_cut = np.concatenate([Mask(spikes, t_start + t_pause+nimg*(t_img+t_pause),  t_start + t_pause+nimg*(t_img+t_pause)+t_img)-(nimg+1)*t_pause for nimg in range(n_imgs)])
        t_stop = t_stop - n_imgs * t_pause
        spikes = spikes_cut[:]

    #data structure
    n_steps_s = round((t_stop - t_start) / dt_s)
    sampling = t_start + np.arange(n_steps_s) * dt_s
    bw_s = dt_s
    fr_bin_s = np.zeros(len(sampling))

    if dt_ds > 0:
        n_steps_ds = round((t_stop - t_start) / dt_ds)
        downsampling = t_start + np.arange(n_steps_ds) * dt_ds
        fr_bin_ds = np.zeros(len(downsampling))

    #sampling fr
    for bin, t in enumerate(sampling):
        t1_s, t2_s = t - bw_s/2, t + bw_s/2
        n_spikes = CountSpikes(spikes, t1_s, t2_s)
        fr_bin_s[bin] = n_spikes

    fr = fr_bin_s

    # convolution
    if nu_t_high != None: fr = signal.convolve(fr, WindowGauss(si_t_high), 'same')

    #downsampling fr
    if dt_ds > 0:
        for i_ds, t in enumerate(downsampling):
            i1_ds, i2_ds = i_ds, i_ds + 1
            i1_s, i2_s = int(i1_ds * (dt_ds / dt_s)), int(i2_ds * (dt_ds / dt_s))
            n_spikes = np.sum(fr[i1_s:i2_s])
            fr_bin_ds[i_ds] = n_spikes

        fr = fr_bin_ds

    if nu_t_low != None: fr = signal.convolve(fr, WindowGauss(si_t_low), 'same')

    # remove zeros
    if remove_zeros == True: fr = fr[(fr > 0)]

    if dt_ds > 0:
        fr = np.array(fr) * (1000 / dt_ds)
    else:
        fr = np.array(fr) * (1000 / dt_s)

    return fr

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

    if type(data) != np.array: data = np.array(data)

    if len(data) == 0:
        return [], []
    else:
        if xmin == None: xmin = np.min(data)
        if xmax == None: xmax = np.max(data)

        if nbins == None and dx == None:
            raise ValueError('\nPlease, provide either the number bins or the preciosion to be used')
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
                bin, hist_plot = KDE1D(x=data, bin_width=bin_width, xrange=range)

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

def GetMNISTFeatures(input_data_path, dataset_name):
    input_data_path = os.path.abspath(input_data_path)
    features = np.load(os.path.join(input_data_path,f'{dataset_name}_features.npy'))
    labels = np.load(os.path.join(input_data_path,f'{dataset_name}_labels.npy'))
    return features, labels

def WCxCxCategories(w_matrix, index_cat_dict, cat='all'):

    weight_cats_dict = {'group': [], 'class': [], 'non-specific': []}
    syn_group, syn_class, syn_diff = [], [], []

    for ntrial, matrix in enumerate(w_matrix):
        indices = index_cat_dict[ntrial]
        if cat == 'group' or cat == 'all':
            w_group = np.hstack(matrix)[indices['group']]
            syn_group.append(w_group)
        if cat == 'class' or cat == 'all':
            w_class = np.hstack(matrix)[indices['class']]
            syn_class.append(w_class)
        if cat == 'non-specific' or cat == 'all':
            w_diff = np.hstack(matrix)[indices['non-specific']]
            syn_diff.append(w_diff)

    if cat == 'group' or cat == 'all':
        weight_cats_dict['group'] = syn_group
    if cat == 'class' or cat == 'all':
        weight_cats_dict['class'] = syn_class
    if cat == 'non-specific' or cat == 'all':
        weight_cats_dict['non-specific'] = syn_diff

    return weight_cats_dict if cat == 'all' else weight_cats_dict[cat]

def WCxThCategories(matrices, cx_labels, th_feat_train, th_indices_train, cx_shuffling, n_neur_cx, n_exc_ca):
    n_groups_cx = n_neur_cx // n_exc_ca
    neurons_cx = np.arange(n_neur_cx, dtype=int)
    groups_cx = np.arange(n_groups_cx, dtype=int)
    syn_cat_dict = {'group':[], 'class': [], 'non-specific': []}


    for ntrial, w_cx_th in enumerate(matrices):
        trial_indices = th_indices_train[ntrial]
        if isinstance(trial_indices, dict):
            trial_indices = np.hstack(FlattenDict2List(trial_indices))
        labels = cx_labels[ntrial]
        th_features = np.array([th_feat_train[index] for index in trial_indices])
        th_features_class = np.array([th_features[(labels == cl)] for cl in np.unique(labels)])
        syn_group, syn_class, syn_aspecific = [], [], []

        if cx_shuffling == None:
            shuffling = groups_cx
        else:
            shuffling = cx_shuffling[ntrial]

        mask_th_classes = np.array([(feat_class > 0) for feat_class in np.sum(th_features_class, axis=1)])

        for ngroup in range(n_groups_cx):
            group_cx = shuffling[ngroup]
            class_cx = labels[group_cx]
            mask_cx = (neurons_cx // n_exc_ca == group_cx)

            group_th = th_features[group_cx]
            mask_th_group = group_th.astype(bool)
            mask_th_class = mask_th_classes[class_cx] & ~mask_th_group
            mask_th_aspecific = np.logical_not(mask_th_group) & ~mask_th_class

            syn_cx_th_group = w_cx_th[mask_cx, :][:, mask_th_group]
            syn_cx_th_class = w_cx_th[mask_cx, :][:, mask_th_class]
            syn_cx_th_aspecific = w_cx_th[mask_cx, :][:, mask_th_aspecific]

            syn_group.append(syn_cx_th_group)
            syn_class.append(syn_cx_th_class)
            syn_aspecific.append(syn_cx_th_aspecific)

        syn_cat_dict['group'].append(FlattenList2List(syn_group))
        syn_cat_dict['class'].append(FlattenList2List(syn_class))
        syn_cat_dict['non-specific'].append(FlattenList2List(syn_aspecific))

    return syn_cat_dict

def WThCxCategories(matrices, cx_labels, th_feat_train, th_indices_train, cx_shuffling, n_neur_cx, n_exc_ca):


    n_groups_cx = n_neur_cx // n_exc_ca
    neurons_cx = np.arange(n_neur_cx, dtype=int)
    groups_cx = np.arange(n_groups_cx, dtype=int)
    syn_cat_dict = {'group':[], 'class': [], 'non-specific': []}

    for ntrial, w_th_cx in enumerate(matrices):
        trial_indices = th_indices_train[ntrial]
        if isinstance(trial_indices, dict):
            trial_indices = np.hstack(FlattenDict2List(trial_indices))
        labels = cx_labels[ntrial]
        th_features = np.array([th_feat_train[index] for index in trial_indices])
        th_features_class = np.array([th_features[(labels == cl)] for cl in np.unique(labels)])
        syn_group, syn_class, syn_aspecific = [], [], []

        if cx_shuffling == None:
            shuffling = groups_cx
        else:
            shuffling = cx_shuffling[ntrial]

        mask_th_classes = np.array([(feat_class > 0) for feat_class in np.sum(th_features_class, axis=1)])

        for ngroup in range(n_groups_cx):
            group_cx = shuffling[ngroup]
            class_cx = labels[group_cx]
            mask_cx = (neurons_cx // n_exc_ca == group_cx)

            group_th = th_features[group_cx]
            mask_th_group = group_th.astype(bool)
            mask_th_class = mask_th_classes[class_cx] & ~mask_th_group
            mask_th_aspecific = np.logical_not(mask_th_group) & ~mask_th_class

            syn_cx_th_group = w_th_cx[mask_th_group, :][:, mask_cx]
            syn_cx_th_class = w_th_cx[mask_th_class, :][:, mask_cx]
            syn_cx_th_aspecific = w_th_cx[mask_th_aspecific, :][:, mask_cx]

            syn_group.append(syn_cx_th_group)
            syn_class.append(syn_cx_th_class)
            syn_aspecific.append(syn_cx_th_aspecific)

        syn_cat_dict['group'].append(FlattenList2List(syn_group))
        syn_cat_dict['class'].append(FlattenList2List(syn_class))
        syn_cat_dict['non-specific'].append(FlattenList2List(syn_aspecific))

    return syn_cat_dict

def GetCxCxIndex(labels, n_areas, n_class, n_ranks, n_exc_ca, n_cycles):

    n_groups = n_areas * n_class * n_ranks
    n_neurons = n_groups * n_exc_ca
    n_synapses = n_neurons ** 2
    index_cat_dict = {'group': np.zeros(n_synapses,dtype=bool), 'class': np.zeros(n_synapses,dtype=bool), 'non-specific': np.zeros(n_synapses,dtype=bool)}
    if labels == 'same': labels = [group // (n_ranks * n_cycles) % n_class for group in range(n_groups)]

    for i in range(n_neurons):
        for j in range(n_neurons):
            group1, group2 = i // n_exc_ca, j // n_exc_ca
            class1, class2 = labels[group1], labels[group2]
            area1, area2 = group1 // (n_class * n_ranks * n_cycles), group2 // (n_class * n_ranks * n_cycles)
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
                if delta == n_class * n_ranks * n_cycles:
                    index_cat_dict['group'][index_flatten] = True
                elif class1 == class2 and delta != n_class * n_ranks * n_cycles:
                    index_cat_dict['class'][index_flatten] = True
                elif class1 != class2:
                    index_cat_dict['non-specific'][index_flatten] = True

    return index_cat_dict

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

def NetAccuracy(spikes_trial, trial_dict_path, stage,  params, n_cycles, prediction='neuron', nconf=None):

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

def KNN(trial_dict_path, params, n_neighbours=1):

    network, dataset = params['network'], params['dataset']

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

def DownStatesInterval(spikes_neurons, stage, substage, params, nconf, oscillations=False):

    net_params, times = params['network'], params['times']

    n_areas, n_class, n_ranks_train, n_exc_ca = net_params['n_areas'], net_params['n_class'], net_params['n_ranks_train'], net_params['n_exc_ca']
    dt, fr_thresh = net_params['dt_fr'], net_params['thresh_fr']
    n_groups = n_areas * n_class * n_ranks_train
    remove_pause = net_params['rem_pause']

    time = times[stage][substage]
    t_start, t_stop = time['start'][nconf], time['stop'][nconf]
    thresh_osc = net_params['thresh_osc']
    sigma_osc = net_params['sigma_t_osc']

    fr_neurons = np.transpose(np.array([FiringRate(spikes, t_start, t_stop, dt_s=dt, dt_ds=0, nu_t_high=None, nu_t_low=None, remove_pause=remove_pause[substage]) for spikes in spikes_neurons]))
    groups_activation = []

    for nbin, t_bin in enumerate(fr_neurons):
        groups_fr = np.mean(np.reshape(t_bin, (n_groups, n_exc_ca)), axis=1)
        groups_active = (groups_fr > fr_thresh).astype(int)
        groups_activation.append(groups_active)

    if oscillations:
        group_act_t = np.sum(np.transpose(groups_activation), axis=0)
        group_act_t_conv = signal.convolve(group_act_t, WindowGauss(sigma_osc), 'same')
        mask_down_states = np.array((group_act_t_conv < thresh_osc), dtype=int)
        time_interval_down_states = np.array([np.sum(seq) for seq in FindSequences(mask_down_states)]) * dt / 1000
        return time_interval_down_states
    else:
        return np.array(groups_activation)