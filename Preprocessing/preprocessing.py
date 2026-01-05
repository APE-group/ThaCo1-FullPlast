import numpy as np
import pandas as pd
import os.path, sys
import gc, time
from lib import (NetAccuracy, MemoryUsage, SecondsConverter, FiringRate, SpikesCount, DownStatesInterval, KNN,
                 GetMNISTFeatures, LoadWeights, GetCxCxIndex, WCxCxCategories, WCxThCategories, WThCxCategories)

#KEYS
debug = True
accuracy_key = True
spikes_count_key = True
firing_rate_key = True
oscillations_key = True
syn_matrix_key = True
syn_cats_key = True


#CONFIGURATIONS - to be chosed among ["mnist_fullplast", "mnist_cxcxplast", 'cifar10_fullplast', "cifar10_cxcxplast"]
configs = 'mnist_fullplast'
#INPUT TYPE
if "cifar" in configs:
    input_model = 'cifar10_resnet324'
elif "mnist" in configs:
    input_model = 'mnist_hog324'
else:
    print("ERROR!! Unknown model")
    sys.exit()

#-------------------------------- PATHS
root_save_path = f'/Users/mac/Documents/Projects/ThaCo3/GitHub/Output/thaco_scirep_10x3_sleep_cycles_2019/Analysis/'
root_loadpath = f'/Users/mac/Documents/Projects/ThaCo3/GitHub/Output/thaco_scirep_10x3_sleep_cycles_2019/SimulationOutput/train_classes_10/train_example_3/'
input_path = f"/Users/mac/PycharmProjects/ThaCo1-FullPlast/Dataset/PreprocessedData/{input_model}"

#----------------------------- NETWORK PARAMETERS
n_areas = 1
n_class = 10
n_ranks_train = 3
n_ranks_test = 25
n_img_test = int(n_ranks_test * n_class)
n_exc_ca = 20
nfov = 9
coding = 4
n_neur_cx = n_areas * n_ranks_train * n_class * n_exc_ca
n_neur_th = nfov * coding * 9

#simulated stages
n_training_cycles = 1
n_sleep_epochs = 20
stages = ['awake_training'] + ['nrem'] * n_sleep_epochs
if debug: stages = ['awake_training', 'nrem']
stages_id = [f"{n_stage:02}_{stage}" for n_stage, stage in enumerate(stages)]
spikes_count_substages = {'awake_training': ['classification'], 'nrem': ['classification', 'sleep']}
substages_dict = {'awake_training': ['learning' ,'classification'], 'nrem': ['thermalization', 'sleep', 'classification']}
syn_cats = ['group', 'class', 'non-specific']
configs = [configs]
n_configs = len(configs)
n_stages = len(stages)
n_cycles = stages.count('awake_training')

#TRIALS LIST
save_label = '_'.join(configs)
save_paths = {conf: os.path.join(root_save_path, conf) for conf in configs}
trials_id = [os.listdir(os.path.join(root_loadpath, f'MainOutput/{conf}')) for conf in configs]
for nconf in range(n_configs):
    if '.DS_Store' in trials_id[nconf]: trials_id[nconf].remove('.DS_Store')
for conf in configs:
    if not os.path.exists(save_paths[conf]): os.makedirs(save_paths[conf])
if debug:
    trials_id = [trials_id[nconf][:2] for nconf in range(n_configs)]
    stages = stages[:2]
    stages_id = stages_id[:2]
n_trials = {conf: len(trials_id[nconf]) for nconf, conf in enumerate(configs)}
n_trials_tot = np.sum([n_trials[conf] for conf in configs])

#----------------------------- TIMING PARAMETERS
t_img_train = 400
t_img_test = 200
t_pause = 400
t_relaxation_test = 8000
t_training = n_ranks_train * n_class * (t_img_train + t_pause)
t_test = n_ranks_test * n_class * (t_img_test + t_pause)
t_nrem_therm = 10e3
t_nrem = 100e3


t_stage_tot = {'awake_training': t_training + t_relaxation_test + t_test,
           'nrem': t_nrem_therm + t_nrem + t_relaxation_test + t_test}
t_tot = t_stage_tot['awake_training'] + (n_stages - 1) * t_stage_tot['nrem']

t_start_train = 0.1 * np.ones(n_configs)
t_stop_train = t_start_train + t_training

t_start_relaxation_pre = t_stop_train
t_stop_relaxation_pre = t_start_relaxation_pre + t_relaxation_test

t_start_test_pre = t_stop_relaxation_pre
t_stop_test_pre = t_start_test_pre + t_test

t_start_nrem_therm = 0.1 * np.ones(n_configs)
t_stop_nrem_therm = t_start_nrem_therm + t_nrem_therm

t_start_nrem = t_stop_nrem_therm
t_stop_nrem = t_start_nrem + t_nrem

t_start_relaxation_post = t_stop_nrem
t_stop_relaxation_post = t_start_relaxation_post + t_relaxation_test

t_start_nrem_test = t_stop_relaxation_post
t_stop_nrem_test = t_start_nrem_test + t_test

#FR PARAMETERS
dt_s_fr = {'low': 0.1, 'high': 0.1}
dt_ds_fr = {'low': 5, 'high': 1}
nu_t_high_fr = {'low': 132, 'high': 500}
nu_t_low_fr = {'low': None, 'high': None}

#DOWNSTATE TIME INTERVALS PARAMETERS
dt_s_osc = 50
n_active = 15
thresh_fr = (1000 / dt_s_osc) * (n_active / n_exc_ca)
thresh_osc = 0.25
sigma_t_osc_nrem = 1
sigma_t_osc_awake = 1

#PARAMETERS DICT
params = {
    'network':{'n_fov':nfov, 'coding':coding, 'n_areas': n_areas, 'n_class': n_class,'n_cycles':n_cycles, 'n_ranks_train': n_ranks_train,
               'n_ranks_test': n_ranks_test, 'n_exc_ca': n_exc_ca, 'trials_id': trials_id, 'dt_fr': dt_s_osc, 'thresh_fr': thresh_fr,
               't_img_test': t_img_test, 't_pause': t_pause, 'n_img_test':n_img_test,
                'thresh_osc': thresh_osc, 'sigma_t_osc': sigma_t_osc_nrem,

    },
    'times':{
        'awake_training': {
        'learning': {'start': t_start_train, 'stop':  t_stop_train},
        'classification': {'start': t_start_test_pre, 'stop':  t_stop_test_pre}
    },
    'nrem': {
        'thermalization': {'start': t_start_nrem_therm, 'stop': t_stop_nrem_therm},
        'sleep': {'start': t_start_nrem, 'stop':  t_stop_nrem},
        'classification': {'start': t_start_nrem_test, 'stop':  t_stop_nrem_test}
    },
    }
}

#DATASET FEATURES PARAMETERS
train_features, train_labels = GetMNISTFeatures(input_path, 'training')
prediction = ['neuron'] #, 'group', 'class']
if isinstance(prediction, str): prediction = [prediction]
accuracy_knn_key = False
if accuracy_knn_key:
    test_features, test_labels = GetMNISTFeatures(input_path, 'test')
    balanced_train, balanced_test = True, True
    neighbours_values = [1, 2, 3, 4, 5]
    params_dataset = {'train_features': train_features, 'train_labels': train_labels, 'balanced_train': balanced_train,
                    'test_features': test_features, 'test_labels': test_labels, 'balanced_test': balanced_test,}
    params['dataset'] = params_dataset

#----------------------------- DATA STRUCTURE
accuracy = pd.DataFrame([], columns=['trial', 'configuration', 'stage', 'prediction', 'value'])
accuracy_knn = pd.DataFrame([], columns=['trial', 'value'])
spikes_count_neuron = pd.DataFrame([], columns=['configuration','pop', 'stage', 'substage', 'value'])
spikes_cx, spikes_th = [], []
fr_cx_single, fr_th_single = [], []
groups_oscillations = pd.DataFrame([], columns=['configuration', 'stage',  'value'])
w_cx_cx_matrix = pd.DataFrame([], columns=['trial', 'configuration', 'stage', 'matrix'])
w_cx_th_matrix = pd.DataFrame([], columns=['trial', 'configuration', 'stage', 'matrix'])
w_th_cx_matrix = pd.DataFrame([], columns=['trial', 'configuration', 'stage', 'matrix'])
w_cx_cx_input = pd.DataFrame([], columns=['trial', 'configuration', 'stage', 'cat', 'value'])
w_cx_th_input = pd.DataFrame([], columns=['trial', 'configuration', 'stage', 'cat', 'value'])
w_th_cx_input = pd.DataFrame([], columns=['trial', 'configuration', 'stage', 'cat', 'value'])


rngs_dict = {conf: [] for nconf, conf in enumerate(configs)}
dt_trial = []
time_left = -1, -1, -1
trials_processed = 0
remove_pause = {'learning': {'n_imgs':n_ranks_train*n_class, 't_img':t_img_train, 't_pause': t_pause},
             'classification': {'n_imgs':n_img_test, 't_img':t_img_test, 't_pause': t_pause},
             'thermalization': False, 'sleep': False}
params['network']['rem_pause'] = remove_pause
labels_train, trial_indices_train = [], []
trial_shuffling = None
trial_dictionaries = {conf: [] for conf in configs}
time_total_start = time.time()
#----------------------------- LOAD DATA
for nconf, conf in enumerate(configs):
    for n_trial, trial_id in enumerate(trials_id[nconf][:n_trials[conf]]):
        #paths
        trial_mo_path = os.path.join(root_loadpath, 'MainOutput', conf, trial_id)
        trial_syn_path = os.path.join(root_loadpath, 'Synapses', conf, trial_id)
        trial_dict_path = os.path.join(trial_mo_path, '00_awake_training')
        if any(not os.path.exists(os.path.join(trial_mo_path, stage_id, 'Events', f'cx_{stage_id}.npy')) for stage_id in stages_id) or any(not os.path.exists(os.path.join(trial_syn_path, stage_id, f'conn_cx_{stage_id}.npy')) for stage_id in stages_id):
            print(f'Trial {trial_id} incomplete')
            trials_processed += 1
            continue
        #trial indices data
        trial_dict = np.load(os.path.join(trial_dict_path, 'trial_dict.npy'), allow_pickle=True, encoding='latin1').item()
        rng_trial = np.load(os.path.join(trial_mo_path, '00_awake_training', 'trial_dict.npy'), allow_pickle=True).item()['nest_seed']
        trial_dictionaries[conf].append(trial_dict)
        labels_train.append(trial_dict['training']['labels'])
        trial_indices_train.append(trial_dict['training']['index mnist'])
        time_start_trial = time.time()
        rngs_dict[conf].append((trial_id, rng_trial))
        ncycle = 0
        if accuracy_knn_key:
            accuracy_knn_trial = {n_neighbours: KNN(trial_dict_path, params, n_neighbours) for n_neighbours in neighbours_values}
            accuracy_knn.loc[len(accuracy_knn)] = [trial_id, accuracy_knn_trial]
        for nstage, stage_id in enumerate(stages_id):
            stage = stages[nstage]
            if stage == 'awake_training': ncycle += 1
            print(f'Configuration: {nconf + 1}/{n_configs} - Trial: {n_trial + 1}/{n_trials[conf]} - Stage: {stage} {nstage + 1}/{len(stages)} - Memory: {MemoryUsage():.2f} MB - Time left: {time_left[0]}h {time_left[1]}m {int(time_left[2])}s                                       '
                  , end='\r', flush=True)

            #----------- paths
            stage_mo_path  = os.path.join(trial_mo_path, stage_id)
            stage_syn_path = os.path.join(trial_syn_path, stage_id)
            spikes_cx_path = os.path.join(stage_mo_path, 'Events', f'cx_{stage_id}.npy')
            spikes_th_path = os.path.join(stage_mo_path, 'Events', f'th_{stage_id}.npy')
            cx_syn_path = os.path.join(stage_syn_path, f'conn_cx_{stage_id}.npy')
            th_syn_path = os.path.join(stage_syn_path, f'conn_th_{stage_id}.npy')

            #----------- raw data
            #cx spikes
            spikes_cx_trial = np.load(spikes_cx_path, allow_pickle=True).item()['evt_exc']
            #syn matrices
            if syn_matrix_key:
                #cx-cx
                w_cx_cx_matrix_trial = LoadWeights(cx_syn_path, syn_type='exc_exc', reshape=[n_neur_cx, n_neur_cx])
                w_cx_cx_matrix.loc[len(w_cx_cx_matrix)] = [trial_id, conf, stage_id, w_cx_cx_matrix_trial]
                #cx-th
                w_cx_th_matrix_trial = LoadWeights(cx_syn_path, syn_type='bwd', reshape=[n_neur_cx, n_neur_th])
                w_cx_th_matrix.loc[len(w_cx_th_matrix)] = [trial_id, conf, stage_id, w_cx_th_matrix_trial]
                #th-cx
                w_th_cx_matrix_trial = LoadWeights(th_syn_path, syn_type='fwd', reshape=[n_neur_th, n_neur_cx])
                w_th_cx_matrix.loc[len(w_th_cx_matrix)] = [trial_id, conf, stage_id, w_th_cx_matrix_trial]

            #----------- processed data
            if accuracy_key:
                accuracy_trial = NetAccuracy(spikes_cx_trial, trial_dict_path, stage,  params, ncycle, prediction, nconf=nconf)
                for pred in prediction: accuracy.loc[len(accuracy)] = [trial_id, conf, stage_id, pred, accuracy_trial[pred]]
            #single neuron fr
            if spikes_count_key:
                spikes_th_trial = np.load(spikes_th_path, allow_pickle=True).item()['evt_exc']
                spikes_count_cx_trial = {substage: SpikesCount(spikes_cx_trial, stage, params, substage, nconf=nconf) for substage in  spikes_count_substages[stage]}
                spikes_count_th_trial = {substage: SpikesCount(spikes_th_trial, stage,  params, substage, nconf=nconf) for substage in spikes_count_substages[stage]}
                for substage in spikes_count_substages[stage]:
                    spikes_count_neuron.loc[len(spikes_count_neuron)] = [conf, 'cx', stage_id, substage, spikes_count_cx_trial[substage]]
                    spikes_count_neuron.loc[len(spikes_count_neuron)] = [conf, 'th', stage_id, substage, spikes_count_th_trial[substage]]

            #convoluted fr
            if firing_rate_key and n_trial==0:
                if not spikes_count_key:
                    spikes_th_trial = np.load(spikes_th_path, allow_pickle=True).item()['evt_exc']
                if nstage == 0:
                    t_stage = 0
                elif nstage ==1:
                    t_stage = t_stage_tot['awake_training']
                else:
                    t_stage = t_stage_tot['awake_training'] + (nstage-1) * t_stage_tot['nrem']
                spikes_cx.append(np.array(spikes_cx_trial, dtype=object) + t_stage)
                spikes_th.append(np.array(spikes_th_trial, dtype=object) + t_stage)

            #cx down states time intervals
            if oscillations_key:
                if stage == 'nrem':
                    group_osc_stage = {substage: DownStatesInterval(spikes_cx_trial, stage, substage, params, nconf, oscillations=True) for substage in substages_dict[stage]}
                    groups_oscillations.loc[len(groups_oscillations)] = [conf, stage_id, group_osc_stage]
            gc.collect()

        trials_processed += 1
        time_end_trial = time.time()
        dt_trial.append(time_end_trial - time_start_trial)
        time_left = SecondsConverter((n_trials_tot - trials_processed) * np.mean(dt_trial))

#----------------------------- OBSERVABLES
#RNGS
for conf in configs:
    trials_id_conf, rngs_conf = np.transpose(sorted(rngs_dict[conf], key=lambda x: x[1]))
    n_trials_conf = len(trials_id_conf)

    if accuracy_knn_key:
        arrays = accuracy_knn['value'].to_numpy()
        accuracy_knn_dict = {n_neighbours: np.array([trial_arr[n_neighbours] for trial_arr in arrays]) for n_neighbours in neighbours_values}
        accuracy_knn_mean = {n_neighbours: np.mean(accuracy_knn_dict[n_neighbours]) for n_neighbours in neighbours_values}
        accuracy_knn_std = {n_neighbours: np.std(accuracy_knn_dict[n_neighbours]) / np.sqrt(n_trials_conf) for n_neighbours in neighbours_values}
        data_path = os.path.join(save_paths[conf], 'accuracy_knn.npy')
        np.save(data_path, {'data': accuracy_knn_dict, 'mean': accuracy_knn_mean, 'std': accuracy_knn_std, 'trials_list': trials_id_conf, 'rngs': rngs_conf})

    if accuracy_key:
        #Accuracy
        accuracy_mean = {pred: {stage: np.mean(accuracy[(accuracy['configuration']==conf) & (accuracy['stage']==stage) & (accuracy['prediction']==pred)]['value'])
                                       for stage in stages_id} for pred in prediction}
        accuracy_std = {pred: {stage: np.std(accuracy[(accuracy['configuration']==conf) & (accuracy['stage']==stage) & (accuracy['prediction']==pred)]['value']) / np.sqrt(n_trials_conf)
                                      for stage in stages_id} for pred in prediction}
        accuracy_full = {pred: {stage: np.hstack([accuracy[(accuracy['configuration']==conf) & (accuracy['stage']==stage) & (accuracy['prediction']==pred) & (accuracy['trial']==trial_id)]['value'].to_numpy() for trial_id in trials_id_conf])
                                   for stage in stages_id} for pred in prediction}

        data_path = os.path.join(save_paths[conf],'accuracy.npy')
        np.save(data_path,{'data': accuracy_full, 'mean':accuracy_mean, 'std': accuracy_std,'trials_list': trials_id_conf, 'rngs': rngs_conf})

    #Number of spikes per neuron in stages
    if spikes_count_key:
        spikes_count_dict = {pop: {stage: {substage: spikes_count_neuron[
        (spikes_count_neuron['configuration']==conf) & (spikes_count_neuron['pop']==pop)
        & (spikes_count_neuron['stage']==stage) & (spikes_count_neuron['substage']==substage)]['value'].to_numpy()
        for substage in spikes_count_substages[stage[3:]]} for stage in stages_id} for pop in ['th', 'cx']}
        data_path = os.path.join(save_paths[conf], 'spikes_count.npy')
        np.save(data_path, {'data': spikes_count_dict, 'n_trials': n_trials_conf, 't_img_presentation': t_img_test, 'trials_list': trials_id_conf, 'rngs': rngs_conf})

    #convoluted firing rate
    if firing_rate_key:
        conv_types = ['low', 'high']
        for conv_type in conv_types:
            dt_s, dt_ds, nu_t_high, nu_t_low = dt_s_fr[conv_type], dt_ds_fr[conv_type], nu_t_high_fr[conv_type], nu_t_low_fr[conv_type]
            raster_cx = [np.concatenate([spikes_neur_stage[nneur] for spikes_neur_stage in spikes_cx]) for nneur in range(n_neur_cx)]
            raster_th = [np.concatenate([spikes_neur_stage[nneur] for spikes_neur_stage in spikes_th]) for nneur in range(n_neur_th)]
            fr_cx = FiringRate(raster_cx[:180], t_start_train[nconf], t_start_train[nconf]+t_tot, dt_s=dt_s, dt_ds=dt_ds, nu_t_high=nu_t_high, nu_t_low=nu_t_low, remove_pause=None)
            fr_th = FiringRate(raster_th, t_start_train[nconf], t_start_train[nconf]+t_tot, dt_s=dt_s, dt_ds=dt_ds, nu_t_high=nu_t_high, nu_t_low=nu_t_low, remove_pause=None)
            data_path = os.path.join(save_paths[conf], f'lfp_single_track_{conv_type}.npy')
            np.save(data_path, {'data': {'cx': {'fr': fr_cx, 'raster': raster_cx}, 'th': {'fr': fr_th, 'raster': raster_th}}, 't_img_presentation': t_img_test, 'dt_s': dt_s, 'dt_ds': dt_ds, 'nu_t_high': nu_t_high, 'nu_t_low': nu_t_low, 'trials_list': trials_id_conf[0]})

    #downstates time intervals
    if oscillations_key:
        oscillations = {stage: groups_oscillations[(groups_oscillations['configuration']==conf) & (groups_oscillations['stage']==stage)]['value'].to_numpy() for stage in stages_id}
        osc_dict = {stage: {substage: [osc_trial[substage] for osc_trial in oscillations[stage]] for substage in substages_dict[stage[3:]]} for stage in stages_id}
        osc_mean = {stage: {substage: [np.mean(arr) for arr in osc_dict[stage][substage]] for substage in substages_dict[stage[3:]]} for stage in stages_id}
        osc_std = {stage: {substage: [np.std(arr) / np.sqrt(len(arr)) for arr in osc_dict[stage][substage]] for substage in substages_dict[stage[3:]]} for stage in stages_id}

        data_path = os.path.join(save_paths[conf], 'cx_groups_oscillations.npy')
        np.save(data_path, {'data': osc_dict, 'mean': osc_mean, 'std': osc_std, 'n_trials': n_trials_conf, 'dt_fr': dt_s_osc,
                            'thresh_fr': thresh_fr ,'thresh_osc':thresh_osc, 'sigma_t_osc_nrem':sigma_t_osc_nrem, 'sigma_t_osc_awake':sigma_t_osc_awake, 'trials_list': trials_id_conf})

    if syn_matrix_key:

        # cx <--> cx syn matrix
        matrices_cx_cx = {stage: w_cx_cx_matrix[(w_cx_cx_matrix['configuration'] == conf) & (w_cx_cx_matrix['stage'] == stage)]['matrix'].to_numpy() for stage in stages_id}
        matrix_mean = {stage: np.mean(matrices_cx_cx[stage], axis=0) for stage in stages_id}
        matrix_std = {stage: np.std(matrices_cx_cx[stage], axis=0) / np.sqrt(n_trials_conf) for stage in stages_id}
        matrix_ratio = {stage: np.mean(matrices_cx_cx[stage] / matrices_cx_cx[stages_id[0]], axis=0) for stage in stages_id}
        data_path = os.path.join(save_paths[conf], 'w_cx_cx_matrix.npy')
        np.save(data_path, {'mean': matrix_mean, 'std': matrix_std, 'ratio': matrix_ratio, 'trials_list': trials_id_conf})

        # cx --> th syn matrix
        matrices_cx_th = {stage: w_cx_th_matrix[(w_cx_th_matrix['configuration'] == conf) & (w_cx_th_matrix['stage'] == stage)]['matrix'].to_numpy() for stage in stages_id}
        matrix_mean = {stage: np.mean(matrices_cx_th[stage], axis=0) for stage in stages_id}
        matrix_std = {stage: np.std(matrices_cx_th[stage], axis=0) / np.sqrt(n_trials_conf) for stage in stages_id}
        matrix_ratio = {stage: np.mean(matrices_cx_th[stage] / matrices_cx_th[stages_id[0]], axis=0) for stage in stages_id}
        data_path = os.path.join(save_paths[conf], 'w_cx_th_matrix.npy')
        np.save(data_path, {'data': w_cx_th_matrix_trial, 'mean': matrix_mean, 'std': matrix_std, 'ratio': matrix_ratio, 'trials_list': trials_id_conf})

        # th --> cx syn matrix
        matrices_th_cx = {stage: w_th_cx_matrix[(w_th_cx_matrix['configuration'] == conf) & (w_th_cx_matrix['stage'] == stage)]['matrix'].to_numpy() for stage in stages_id}
        matrix_mean = {stage: np.mean(matrices_th_cx[stage], axis=0) for stage in stages_id}
        matrix_std = {stage: np.std(matrices_th_cx[stage], axis=0) / np.sqrt(n_trials_conf) for stage in stages_id}
        matrix_ratio = {stage: np.mean(matrices_th_cx[stage] / matrices_th_cx[stages_id[0]], axis=0) for stage in stages_id}
        data_path = os.path.join(save_paths[conf], 'w_th_cx_matrix.npy')
        np.save(data_path, {'data': w_th_cx_matrix_trial, 'mean': matrix_mean, 'std': matrix_std, 'ratio': matrix_ratio, 'trials_list': trials_id_conf})

        #neuron input/output
        if spikes_count_key:
            fr_cx = {stage_id: np.transpose(list(map(list, zip(*spikes_count_dict['cx'][stage_id]['classification'])))) for
                     stage_id in stages_id}
            fr_th = {stage_id: np.transpose(list(map(list, zip(*spikes_count_dict['th'][stage_id]['classification'])))) for
                     stage_id in stages_id}
        else:
            data_path = os.path.join(save_paths[conf], 'spikes_count.npy')
            spikes_count = np.load(data_path, allow_pickle=True).item()
            fr_cx = {stage_id: np.transpose(list(map(list, zip(*spikes_count['data']['cx'][stage_id]['classification'])))) for
                     stage_id in stages_id}
            fr_th = {stage_id: np.transpose(list(map(list, zip(*spikes_count['data']['th'][stage_id]['classification'])))) for
                     stage_id in stages_id}

        # -- CORTEX

        #cx input
        w_input_cx = {'cx': {stage_id: [np.transpose(matrices_cx_cx[stage_id][ntrial]).sum(axis=1) for ntrial in range(n_trials_conf)] for stage_id in stages_id},
                      'th': {stage_id: [np.transpose(matrices_th_cx[stage_id][ntrial]).sum(axis=1) for ntrial in range(n_trials_conf)] for stage_id in stages_id}}
        # cx output
        w_output_cx = {'cx': {stage_id: [matrices_cx_cx[stage_id][ntrial].sum(axis=1) for ntrial in range(n_trials_conf)] for stage_id in stages_id},
                      'th': {stage_id: [matrices_cx_th[stage_id][ntrial].sum(axis=1) for ntrial in range(n_trials_conf)] for stage_id in stages_id}}
        # cx synaptic activity
        syn_activity_cx = {'cx': {stage_id: [fr_cx[stage_id][ntrial] @ matrices_cx_cx[stage_id][ntrial] for ntrial in range(n_trials_conf)] for stage_id in stages_id},
                           'th': {stage_id: [fr_th[stage_id][ntrial] @ matrices_th_cx[stage_id][ntrial] for ntrial in range(n_trials_conf)] for stage_id in stages_id}}
        data_path = os.path.join(save_paths[conf], 'w_cx_sum.npy')
        np.save(data_path, {'input': w_input_cx, 'output': w_output_cx, 'synaptic_activity':syn_activity_cx, 'trials_list': trials_id_conf})

        # -- THALAMUS

        # th input
        w_input_th = {'cx': {stage_id: [np.transpose(matrices_cx_th[stage_id][ntrial]).sum(axis=1) for ntrial in range(n_trials_conf)] for stage_id in stages_id},
                      'th': {stage_id: [None for ntrial in range(n_trials_conf)] for stage_id in stages_id}}

        # th output
        w_output_th = {'cx': {stage_id: [matrices_th_cx[stage_id][ntrial].sum(axis=1) for ntrial in range(n_trials_conf)] for stage_id in stages_id},
                      'th': {stage_id: [None for ntrial in range(n_trials_conf)] for stage_id in stages_id}}

        # th synaptic activity
        syn_activity_th = {'cx': {stage_id: [fr_cx[stage_id][ntrial] @ matrices_cx_th[stage_id][ntrial] for ntrial in range(n_trials_conf)] for stage_id in stages_id},
                           'th': {stage_id: [None for ntrial in range(n_trials_conf)] for stage_id in stages_id}}

        data_path = os.path.join(save_paths[conf], 'w_th_sum.npy')
        np.save(data_path, {'input': w_input_th, 'output': w_output_th, 'synaptic_activity':syn_activity_th, 'trials_list': trials_id_conf})

        if syn_cats_key:
            #cx neuron (group-, class-, non-specific) indices
            indices_cat_dict = [GetCxCxIndex('same', n_areas, n_class, n_ranks_train, n_exc_ca, n_training_cycles)] * n_trials_conf

            #CX-CX categorized syn. weights (group-, class-, non-specific)
            w_cats_trials = {stage: WCxCxCategories(matrices_cx_cx[stage], indices_cat_dict) for stage in stages_id}
            w_cats_trials = {cat: {stage: w_cats_trials[stage][cat] for stage in stages_id} for cat in syn_cats}

            # mean & sem
            w_cats_mu = {cat: {stage: np.array([np.mean(w_cat) for w_cat in w_cats_trials[cat][stage]]) for stage in stages_id} for cat in syn_cats}
            w_cats_std = {cat: {stage: np.array([np.std(w_cat) / np.sqrt(len(w_cat)) for w_cat in w_cats_trials[cat][stage]]) for stage in stages_id} for cat in syn_cats}
            data_path = os.path.join(save_paths[conf],'w_cx_cx_categories.npy')
            np.save(data_path,{'mean':w_cats_mu, 'std': w_cats_std, 'trials_list': trials_id_conf})

            #syn matrices in 40th to 60th percentile class-specific post-sleep potentiation
            w_class_mean_trials = np.array([w_cats_mu['class'][stage] for stage in stages_id])
            matrices = matrices_cx_cx
            p1, p2, p3 = np.percentile(w_class_mean_trials[-1], 40), np.percentile(w_class_mean_trials[-1],50), np.percentile(w_class_mean_trials[-1], 60)
            p13_cx_ids = [ntrial for ntrial, w_trial in enumerate(w_class_mean_trials[-1]) if w_trial > p1 and w_trial < p3]
            if not p13_cx_ids: p13_cx_ids = np.arange(n_trials_conf, dtype=int)
            w_matrix_trials = {stage: [matrices[stage][idx] for idx in p13_cx_ids] for stage in [stages_id[0], stages_id[-1]]}
            indices_trials = [trial_dictionaries[conf][ntrial] for ntrial in p13_cx_ids]
            trials_id_p13 = [trials_id_conf[ntrial] for ntrial in p13_cx_ids]
            data_path = os.path.join(save_paths[conf], 'w_cx_cx_matrix.npy')
            matrix_data = np.load(data_path, allow_pickle=True).item()
            matrix_data['data'] = {'matrices': w_matrix_trials, 'indices': indices_trials, 'trials_id': trials_id_p13}
            np.save(data_path, matrix_data)


            #CX-TH categorized syn. weights (group-, class-, non-specific)
            w_cats_trials = {stage: WCxThCategories(matrices_cx_th[stage], labels_train, train_features, trial_indices_train, trial_shuffling, n_neur_cx, n_exc_ca) for stage in stages_id}
            w_cats_trials = {cat: {stage: w_cats_trials[stage][cat] for stage in stages_id} for cat in syn_cats}

            # mean & sem
            w_cats_mu = {cat: {stage: np.array([np.mean(w_cat) for w_cat in w_cats_trials[cat][stage]]) for stage in stages_id} for cat in syn_cats}
            w_cats_std = {cat: {stage: np.array([np.std(w_cat) / np.sqrt(len(w_cat)) for w_cat in w_cats_trials[cat][stage]])  for stage in stages_id} for cat in syn_cats}
            data_path = os.path.join(save_paths[conf], 'w_cx_th_categories.npy')
            np.save(data_path, {'mean': w_cats_mu, 'std': w_cats_std, 'trials_list': trials_id_conf})

            # syn matrices in 40th to 60th percentile class-specific post-sleep potentiation
            w_class_mean_trials = np.array([w_cats_mu['class'][stage] for stage in stages_id])
            matrices = matrices_cx_th
            w_matrix_trials = {stage: [matrices[stage][idx] for idx in p13_cx_ids] for stage in [stages_id[0], stages_id[-1]]}
            data_path = os.path.join(save_paths[conf], 'w_cx_th_matrix.npy')
            matrix_data = np.load(data_path, allow_pickle=True).item()
            matrix_data['data'] = {'matrices': w_matrix_trials, 'indices': indices_trials, 'trials_id': trials_id_p13}
            np.save(data_path, matrix_data)

            #TH-CX categorized syn. weights (group-, class-, non-specific)
            w_cats_trials = {stage: WThCxCategories(matrices_th_cx[stage], labels_train, train_features, trial_indices_train, trial_shuffling, n_neur_cx, n_exc_ca) for stage in stages_id}
            w_cats_trials = {cat: {stage: w_cats_trials[stage][cat] for stage in stages_id} for cat in syn_cats}

            # mean & sem
            w_cats_mu = {cat: {stage: np.array([np.mean(w_cat) for w_cat in w_cats_trials[cat][stage]]) for stage in stages_id} for cat in syn_cats}
            w_cats_std = {cat: {stage: np.array([np.std(w_cat) / np.sqrt(len(w_cat)) for w_cat in w_cats_trials[cat][stage]]) for stage in stages_id} for cat in syn_cats}
            data_path = os.path.join(save_paths[conf], 'w_th_cx_categories.npy')
            np.save(data_path, {'mean': w_cats_mu, 'std': w_cats_std, 'trials_list': trials_id_conf})

            # syn matrices in 40th to 60th percentile for class-specific post-sleep potentiation
            w_class_mean_trials = np.array([w_cats_mu['class'][stage] for stage in stages_id])
            matrices = matrices_th_cx
            w_matrix_trials = {stage: [matrices[stage][idx] for idx in p13_cx_ids] for stage in [stages_id[0], stages_id[-1]]}
            data_path = os.path.join(save_paths[conf], 'w_th_cx_matrix.npy')
            matrix_data = np.load(data_path, allow_pickle=True).item()
            matrix_data['data'] = {'matrices': w_matrix_trials, 'indices': indices_trials, 'trials_id': trials_id_p13}
            np.save(data_path, matrix_data)


time_total_stop = time.time()
time_execution =  SecondsConverter(time_total_stop - time_total_start)
print(f'\nExecution time: {time_execution[0]}h {time_execution[1]}m {int(time_execution[2])}s')