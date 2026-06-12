import numpy as np
import os.path
import gc, time
from lib import (LoadWeights, NetAccuracy, MemoryUsage, SecondsConverter, FiringRate, SpikesCount,
                 GroupsActivation, GetDatasetFeatures, SynCategoriesMask,
                 DatasetClassSample, DatasetSimilarityHistogram, DatasetSimilarityMatrix, DatasetUmap,
                 LoadDictFromYaml)

config_path = os.path.join(os.path.dirname(__file__), 'config_1.yaml')
config_dict = LoadDictFromYaml(config_path)

execution_config = config_dict['execution']
analysis_keys_config = config_dict['analysis_keys']
model_config = config_dict['model']
path_config = config_dict['paths']
network_config = config_dict['network']
timing_config = config_dict['timing']
fr_single_config = config_dict['firing_rate_single']
oscillations_config = config_dict['oscillations']
synapses_config = config_dict['synapses']
dataset_similarity_config = config_dict['dataset_similarity']

debug = execution_config['debug']

accuracy_key = analysis_keys_config['accuracy_key']
spikes_count_key = analysis_keys_config['spikes_count_key']
firing_rate_key = analysis_keys_config['firing_rate_single_key']
oscillations_key = analysis_keys_config['oscillations_key']
syn_matrix_trial_key = analysis_keys_config['syn_matrix_trial_key']
syn_cats_key = analysis_keys_config['syn_cats_key']
syn_activity_key = analysis_keys_config['syn_activity_key']
dataset_similarity_key = analysis_keys_config['dataset_similarity_key']

spikes_count_pops = ['exc', 'inh'] if spikes_count_key == True else [spikes_count_key] if spikes_count_key else []

#----------------------------- MODEL CONFIG
n_areas = network_config['n_areas']
n_class = network_config['n_class']
n_ranks_train = network_config['n_ranks_train']
n_ranks_test = network_config['n_ranks_test']
n_exc_ca = network_config['n_exc_ca']
autapses = network_config['autapses']
t_img_train, t_img_test, t_pause, t_relaxation_test = timing_config['t_img_train'], timing_config['t_img_test'], timing_config['t_pause'], timing_config['t_relaxation_test']
t_nrem_therm, t_nrem = timing_config['t_nrem_therm'], timing_config['t_nrem']
n_fov, coding = network_config['n_fov'], network_config['coding']
th_inh_exc, cx_inh_exc = synapses_config['th_inh_exc'], synapses_config['cx_inh_exc']

configs = [conf_dict['config'] for conf_dict in model_config['experiments']]
input_models = [conf_dict['input_model'] for conf_dict in model_config['experiments']]
prediction = model_config['prediction']
train_example_shuffling = model_config['train_example_shuffling']
remove_trials = model_config['remove_trials']
select_trials = model_config['select_trials']

if syn_matrix_trial_key == True:
    syn_matrix_trial_key = ['cx_cx', 'cx_th', 'th_cx']
elif syn_matrix_trial_key == False:
    syn_matrix_trial_key = []
if isinstance(syn_matrix_trial_key, str):
    syn_matrix_trial_key = [syn_matrix_trial_key]

#----------------------------- PATHS
root_save_path = path_config['root_save_path']
root_loadpath = path_config['root_loadpath']
root_input_path = path_config['root_input_path']
if not os.path.isabs(root_save_path):
    root_save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), root_save_path))
if not os.path.isabs(root_loadpath):
    root_loadpath = os.path.abspath(os.path.join(os.path.dirname(__file__), root_loadpath))
if not os.path.isabs(root_input_path):
    root_input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), root_input_path))

stages_id = config_dict['stages_id']
stages = [stage_id[3:] for stage_id in stages_id]
n_training_cycles = stages.count('awake_training')
spikes_count_substages = {'awake_training': ['classification'], 'nrem': ['classification', 'sleep']}
substages_dict = {'awake_training': ['learning', 'classification'], 'nrem': ['thermalization', 'sleep', 'classification']}
syn_cats = ['group', 'class', 'non-specific']

n_configs = len(configs)
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

ncycle = 0
ncycle_stage = {}
for nstage, stage_id in enumerate(stages_id):
    if stages[nstage] == 'awake_training':
        ncycle = ncycle + 1
    ncycle_stage[stage_id] = ncycle

#----------------------------- PARAMETERS
n_trials = {conf: len(trials_id[nconf]) for nconf, conf in enumerate(configs)}
n_trials_tot = np.sum([n_trials[conf] for conf in configs])
n_stages = len(stages)
n_cycles = n_training_cycles
n_img_test = int(n_ranks_test * n_class)

t_training = n_ranks_train * n_class * (t_img_train + t_pause) + t_pause
t_test = n_ranks_test * n_class * (t_img_test + t_pause) + t_pause

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

#FIRING RATE
conv_types = ['low', 'high'] if type(firing_rate_key) == bool else firing_rate_key
dt_s_fr = fr_single_config['dt_s_single']
dt_ds_fr = fr_single_config['dt_ds_single']
nu_t_high_fr = fr_single_config['nu_t_high_single']
nu_t_low_fr = fr_single_config['nu_t_low_single']

#DATASET SIMILARITY
max_features_per_class = dataset_similarity_config['max_features_per_class']
max_umap_samples = dataset_similarity_config['max_umap_samples']
dataset_similarity_seed = dataset_similarity_config['seed']
dataset_similarity_norm = dataset_similarity_config['norm_factor']
umap_n_components = dataset_similarity_config['n_components']
umap_n_neighbors = dataset_similarity_config['n_neighbors']
umap_min_dist = dataset_similarity_config['min_dist']

#OSCILLATIONS
dt_fr = oscillations_config['dt_fr']
n_active = oscillations_config['n_active']
thresh_fr = (1000 / dt_fr) * (n_active / n_exc_ca)
thresh_osc = oscillations_config['thresh_osc']
sigma_t_osc_nrem = oscillations_config['sigma_t_osc_nrem']
sigma_t_osc_awake = oscillations_config['sigma_t_osc_awake']

params = {
    'network': {'n_fov': n_fov, 'coding': coding, 'n_areas': n_areas, 'n_class': n_class, 'n_cycles': n_cycles, 'n_ranks_train': n_ranks_train,
                'n_ranks_test': n_ranks_test, 'n_exc_ca': n_exc_ca, 'trials_id': trials_id, 'dt_fr': dt_fr, 'thresh_fr': thresh_fr,
                't_img_test': t_img_test, 't_pause': t_pause, 'n_img_test': n_img_test, 'thresh_osc': thresh_osc, 'sigma_t_osc': sigma_t_osc_nrem},
    'times': {
        'awake_training': {
            'learning': {'start': t_start_train, 'stop': t_stop_train},
            'classification': {'start': t_start_test_pre, 'stop': t_stop_test_pre}
        },
        'nrem': {
            'thermalization': {'start': t_start_nrem_therm, 'stop': t_stop_nrem_therm},
            'sleep': {'start': t_start_nrem, 'stop': t_stop_nrem},
            'classification': {'start': t_start_nrem_test, 'stop': t_stop_nrem_test}
        },
    }
}

rem_pause = {'learning': {'n_imgs': n_ranks_train * n_class, 't_img': t_img_train, 't_pause': t_pause},
             'classification': {'n_imgs': n_img_test, 't_img': t_img_test, 't_pause': t_pause},
             'thermalization': False, 'sleep': False}
params['network']['rem_pause'] = rem_pause
params['network']['ncycle_stage'] = ncycle_stage

n_neurons = {'cx': {'exc': n_areas * n_ranks_train * n_class * n_exc_ca},
             'th': {'exc': n_fov * coding * 9}}
n_neur_cx = n_neurons['cx']['exc']
n_neur_th = n_neurons['th']['exc']

net_params = {'n_class': n_class, 'n_ranks_training': n_ranks_train, 'n_neur_th': n_neur_th, 'n_neur_group': n_exc_ca,
              'n_areas': n_areas, 'n_cycles': 1, 'autapses': autapses}

#----------------------------- DATA STRUCTURE
accuracy = {conf: {pred: {stage: {} for stage in stages_id} for pred in prediction} for conf in configs}
spikes_count_neuron = {conf: {pop: {neur_type: {stage: {substage: {} for substage in spikes_count_substages[stage[3:]]} for stage in stages_id}
                                    for neur_type in ['exc', 'inh']} for pop in ['th', 'cx']} for conf in configs}
groups_oscillations = {conf: {stage: {} for stage in stages_id} for conf in configs}
w_cx_cx_matrix = {conf: {stage: {} for stage in stages_id} for conf in configs}
w_cx_th_matrix = {conf: {stage: {} for stage in stages_id} for conf in configs}
w_th_cx_matrix = {conf: {stage: {} for stage in stages_id} for conf in configs}
matrix_data_cx_cx = {conf: {} for conf in configs}
matrix_data_cx_th = {conf: {} for conf in configs}
matrix_data_th_cx = {conf: {} for conf in configs}
w_cx_sum = {conf: {} for conf in configs}
w_th_sum = {conf: {} for conf in configs}

features_train_dict = {}
labels_train = {conf: {} for conf in configs}
th_features_train = {conf: {} for conf in configs}
trial_shuffling = {conf: {} for conf in configs} if train_example_shuffling else None

rngs_dict = {conf: [] for conf in configs}
trial_dictionaries = {conf: {} for conf in configs}
trials_id_loaded = {conf: [] for conf in configs}
trials_incomplete = {conf: {} for conf in configs}
trials_removed = {conf: [] for conf in configs}
trial_diff = {conf: 0 for conf in configs}
spikes_cx, spikes_th = {conf: [] for conf in configs}, {conf: [] for conf in configs}
dt_trial = []
time_left = -1, -1, -1
trials_processed = 0

time_total_start = time.time()

#----------------------------- DATASET FEATURES
syn_cat_mask_cxcx_key = syn_cats_key or bool(syn_matrix_trial_key)
syn_cat_mask_cx_th_th_cx_key = syn_cats_key
synapses_key = syn_activity_key or syn_cat_mask_cxcx_key or syn_cat_mask_cx_th_th_cx_key
spikes_count_neuron_measure_pops = spikes_count_pops + [pop for pop in ['exc', 'inh'] if syn_activity_key and pop not in spikes_count_pops]
spikes_cx_exc_key = accuracy_key or firing_rate_key or oscillations_key or 'exc' in spikes_count_neuron_measure_pops
spikes_cx_inh_key = 'inh' in spikes_count_neuron_measure_pops
spikes_th_exc_key = firing_rate_key or 'exc' in spikes_count_neuron_measure_pops
spikes_th_inh_key = 'inh' in spikes_count_neuron_measure_pops
spikes_cx_key = spikes_cx_exc_key or spikes_cx_inh_key
spikes_th_key = spikes_th_exc_key or spikes_th_inh_key
spikes_count_neuron_measure_key = len(spikes_count_neuron_measure_pops) > 0
labels_train_key = syn_cat_mask_cx_th_th_cx_key or (syn_cat_mask_cxcx_key and train_example_shuffling)
th_features_key = syn_cat_mask_cx_th_th_cx_key

for nconf, conf in enumerate(configs):
    if th_features_key:
        train_features_conf = GetDatasetFeatures(input_models[nconf], 'training', root_input_path)[0]
        features_train_dict[conf] = np.asarray(train_features_conf)

#----------------------------- TRIAL DISCOVERY
for nconf, conf in enumerate(configs):
    for n_trial, trial_id in enumerate(trials_id[nconf]):
        trial_path = os.path.join(root_loadpath, 'MainOutput', conf, trial_id)
        trial_syn_path = os.path.join(root_loadpath, 'Synapses', conf, trial_id)
        trial_dict_path = os.path.join(trial_path, '00_awake_training')
        trial_dict_full_path = os.path.join(trial_dict_path, 'trial_dict.npy')

        if not os.path.exists(trial_dict_full_path):
            rng_trial = {'nest_seed': None, 'numpy_seed': None}
            print(f'{conf} - Trial {trial_id} incomplete - rng {rng_trial}')
            trials_incomplete[conf][trial_id] = rng_trial
            trial_diff[conf] = trial_diff[conf] + 1
            continue

        trial_dict = np.load(trial_dict_full_path, allow_pickle=True, encoding='latin1').item()
        nest_seed = trial_dict['nest_seed'] if 'nest_seed' in trial_dict else None
        numpy_seed = trial_dict['numpy_seed'] if 'numpy_seed' in trial_dict else None
        rng_trial = {'nest_seed': nest_seed, 'numpy_seed': numpy_seed}

        if spikes_cx_key:
            if any(not os.path.exists(os.path.join(trial_path, stage_id, 'Events', f'cx_{stage_id}.npy')) for stage_id in stages_id):
                print(f'{conf} - Trial {trial_id} incomplete - rng {rng_trial}')
                trials_incomplete[conf][trial_id] = rng_trial
                trial_diff[conf] = trial_diff[conf] + 1
                continue

        if spikes_th_key:
            if any(not os.path.exists(os.path.join(trial_path, stage_id, 'Events', f'th_{stage_id}.npy')) for stage_id in stages_id):
                print(f'{conf} - Trial {trial_id} incomplete - rng {rng_trial}')
                trials_incomplete[conf][trial_id] = rng_trial
                trial_diff[conf] = trial_diff[conf] + 1
                continue

        if synapses_key:
            if any(not os.path.exists(os.path.join(trial_syn_path, stage_id, f'conn_cx_{stage_id}.npy')) for stage_id in stages_id):
                print(f'{conf} - Trial {trial_id} incomplete - rng {rng_trial}')
                trials_incomplete[conf][trial_id] = rng_trial
                trial_diff[conf] = trial_diff[conf] + 1
                continue
            if any(not os.path.exists(os.path.join(trial_syn_path, stage_id, f'conn_th_{stage_id}.npy')) for stage_id in stages_id):
                print(f'{conf} - Trial {trial_id} incomplete - rng {rng_trial}')
                trials_incomplete[conf][trial_id] = rng_trial
                trial_diff[conf] = trial_diff[conf] + 1
                continue

        if nest_seed in remove_trials and nest_seed not in select_trials:
            print('\nTrial to be removed')
            trials_removed[conf].append(trial_id)
            continue

        rngs_dict[conf].append((trial_id, nest_seed, numpy_seed))
        trial_dictionaries[conf][trial_id] = trial_dict
        trials_id_loaded[conf].append(trial_id)

        if labels_train_key:
            labels_train[conf][trial_id] = trial_dict['training']['labels']
            if train_example_shuffling:
                trial_shuffling[conf][trial_id] = trial_dict['training']['index shuffling']
        if th_features_key:
            th_features_trial = features_train_dict[conf][trial_dict['training']['index mnist']]
            if train_example_shuffling:
                th_features_trial = th_features_trial[trial_dict['training']['index shuffling']]
            th_features_train[conf][trial_id] = th_features_trial

n_trials = {conf: len(trials_id_loaded[conf]) for conf in configs}
n_trials_tot = np.sum([n_trials[conf] for conf in configs])
trials_id_sorted = {conf: [trial_id for trial_id, nest_seed, numpy_seed in sorted(rngs_dict[conf], key=lambda x: x[1] if x[1] is not None else -1)] for conf in configs}
rngs_sorted = {conf: {'nest_seed': np.array([nest_seed for trial_id, nest_seed, numpy_seed in sorted(rngs_dict[conf], key=lambda x: x[1] if x[1] is not None else -1)], dtype=object),
                      'numpy_seed': np.array([numpy_seed for trial_id, nest_seed, numpy_seed in sorted(rngs_dict[conf], key=lambda x: x[1] if x[1] is not None else -1)], dtype=object)} for conf in configs}
params['network']['trials_id'] = [trials_id_sorted[conf] for conf in configs]

#----------------------------- LOAD DATA
for nconf, conf in enumerate(configs):
    for n_trial, trial_id in enumerate(trials_id_loaded[conf][:n_trials[conf]]):
        time_start_trial = time.time()
        trial_path = os.path.join(root_loadpath, 'MainOutput', conf, trial_id)
        trial_syn_path = os.path.join(root_loadpath, 'Synapses', conf, trial_id)
        trial_dict_path = os.path.join(trial_path, '00_awake_training')
        trial_dict = trial_dictionaries[conf][trial_id]

        for nstage, stage_id in enumerate(stages_id):
            stage = stages[nstage]
            ncycle = ncycle_stage[stage_id]
            print(f'Configuration: {nconf + 1}/{n_configs} - Trial: {n_trial + 1}/{n_trials[conf]} - Stage: {stage} {nstage + 1}/{len(stages)} - Memory: {MemoryUsage():.2f} MB - Time left: {time_left[0]}h {time_left[1]}m {int(time_left[2])}s                                       '
                  , end='\r', flush=True)

            stage_path = os.path.join(trial_path, stage_id)
            stage_syn_path = os.path.join(trial_syn_path, stage_id)
            spikes_cx_trial, spikes_th_trial = None, None
            spikes_cx_inh_trial, spikes_th_inh_trial = None, None

            if spikes_cx_key:
                spikes_cx_path = os.path.join(stage_path, 'Events', f'cx_{stage_id}.npy')
                spikes_cx_data = np.load(spikes_cx_path, allow_pickle=True).item()
                if spikes_cx_exc_key:
                    spikes_cx_trial = spikes_cx_data['evt_exc']
                if spikes_cx_inh_key:
                    spikes_cx_inh_trial = spikes_cx_data['evt_inh']

            if spikes_th_key:
                spikes_th_path = os.path.join(stage_path, 'Events', f'th_{stage_id}.npy')
                spikes_th_data = np.load(spikes_th_path, allow_pickle=True).item()
                if spikes_th_exc_key:
                    spikes_th_trial = spikes_th_data['evt_exc']
                if spikes_th_inh_key:
                    spikes_th_inh_trial = spikes_th_data['evt_inh']

            if synapses_key:
                cx_syn_path = os.path.join(stage_syn_path, f'conn_cx_{stage_id}.npy')
                th_syn_path = os.path.join(stage_syn_path, f'conn_th_{stage_id}.npy')
                cx_syn_data = np.load(cx_syn_path, allow_pickle=True).item()
                th_syn_data = np.load(th_syn_path, allow_pickle=True).item()

                w_cx_cx_matrix_trial = LoadWeights(cx_syn_data, syn_type='exc_exc', reshape=[n_neur_cx*ncycle, n_neur_cx*ncycle])
                w_th_cx_matrix_trial = LoadWeights(th_syn_data, syn_type='fwd', reshape=[n_neur_th, n_neur_cx*ncycle])
                w_cx_th_matrix_trial = LoadWeights(cx_syn_data, syn_type='bwd', reshape=[n_neur_cx*ncycle, n_neur_th])
                w_cx_cx_matrix[conf][stage_id][trial_id] = w_cx_cx_matrix_trial
                w_th_cx_matrix[conf][stage_id][trial_id] = w_th_cx_matrix_trial
                w_cx_th_matrix[conf][stage_id][trial_id] = w_cx_th_matrix_trial

            if accuracy_key:
                accuracy_trial = NetAccuracy(spikes_cx_trial, trial_dict_path, stage, params, ncycle, prediction, nconf=nconf, trial_dict=trial_dict)
                for pred in prediction: accuracy[conf][pred][stage_id][trial_id] = accuracy_trial[pred]

            if spikes_count_neuron_measure_key:
                for neur_type, spikes_cx_pop_trial, spikes_th_pop_trial in [('exc', spikes_cx_trial, spikes_th_trial), ('inh', spikes_cx_inh_trial, spikes_th_inh_trial)]:
                    if neur_type not in spikes_count_neuron_measure_pops:
                        continue
                    spikes_count_cx_trial = {substage: SpikesCount(spikes_cx_pop_trial, stage, params, substage, nconf=nconf) for substage in spikes_count_substages[stage]}
                    spikes_count_th_trial = {substage: SpikesCount(spikes_th_pop_trial, stage, params, substage, nconf=nconf) for substage in spikes_count_substages[stage]}
                    for substage in spikes_count_substages[stage]:
                        spikes_count_neuron[conf]['cx'][neur_type][stage_id][substage][trial_id] = spikes_count_cx_trial[substage]
                        spikes_count_neuron[conf]['th'][neur_type][stage_id][substage][trial_id] = spikes_count_th_trial[substage]

            if firing_rate_key and n_trial == 0:
                if nstage == 0:
                    t_stage = 0
                elif nstage == 1:
                    t_stage = t_stage_tot['awake_training']
                else:
                    t_stage = t_stage_tot['awake_training'] + (nstage - 1) * t_stage_tot['nrem']
                spikes_cx[conf].append(np.array(spikes_cx_trial, dtype=object) + t_stage)
                spikes_th[conf].append(np.array(spikes_th_trial, dtype=object) + t_stage)

            if oscillations_key:
                if stage == 'nrem':
                    group_osc_stage = {substage: GroupsActivation(spikes_cx_trial, stage, substage, params, nconf, oscillations=True) for substage in substages_dict[stage]}
                    groups_oscillations[conf][stage_id][trial_id] = group_osc_stage

            gc.collect()

        trials_processed += 1
        time_end_trial = time.time()
        dt_trial.append(time_end_trial - time_start_trial)
        time_left = SecondsConverter((n_trials_tot - trials_processed) * np.mean(dt_trial))
        gc.collect()

#----------------------------- SYNAPTIC OBSERVABLES
if synapses_key:
    for nconf, conf in enumerate(configs):
        trials_id_conf, rngs_conf = trials_id_sorted[conf], rngs_sorted[conf]
        trials_failed_conf = trials_incomplete[conf]
        n_trials_conf = len(trials_id_conf)

        conn_specs = {'cx_cx': {'matrix': w_cx_cx_matrix[conf], 'data': matrix_data_cx_cx, 'mask_key': syn_cat_mask_cxcx_key,
                                'mask_store': {}, 'cats_file': 'w_cx_cx_categories.npy'},
                      'cx_th': {'matrix': w_cx_th_matrix[conf], 'data': matrix_data_cx_th, 'mask_key': syn_cat_mask_cx_th_th_cx_key,
                                'mask_store': {}, 'cats_file': 'w_cx_th_categories.npy'},
                      'th_cx': {'matrix': w_th_cx_matrix[conf], 'data': matrix_data_th_cx, 'mask_key': syn_cat_mask_cx_th_th_cx_key,
                                'mask_store': {}, 'cats_file': 'w_th_cx_categories.npy'}}
        matrices_dict = {}

        for conn_type in ['cx_cx', 'cx_th', 'th_cx']:
            matrices_dict[conn_type] = {stage: [conn_specs[conn_type]['matrix'][stage][trial_id] for trial_id in trials_id_conf] for stage in stages_id}
            matrices = matrices_dict[conn_type]
            matrices_0 = matrices[stages_id[0]]
            matrix_mean = {stage: np.mean(matrices[stage], axis=0) for stage in stages_id}
            matrix_std = {stage: np.std(matrices[stage], axis=0) / np.sqrt(n_trials_conf) for stage in stages_id}
            if conn_type == 'cx_cx':
                matrix_ratio = {stage: np.mean([matrices[stage][ntrial][:min(np.shape(matrices[stage][ntrial])[0], np.shape(matrices_0[ntrial])[0]),
                                                                        :min(np.shape(matrices[stage][ntrial])[1], np.shape(matrices_0[ntrial])[1])] /
                                                matrices_0[ntrial][:min(np.shape(matrices[stage][ntrial])[0], np.shape(matrices_0[ntrial])[0]),
                                                                   :min(np.shape(matrices[stage][ntrial])[1], np.shape(matrices_0[ntrial])[1])]
                                                for ntrial in range(n_trials_conf)], axis=0) for stage in stages_id}
            elif conn_type == 'cx_th':
                matrix_ratio = {stage: np.mean([matrices[stage][ntrial][:min(np.shape(matrices[stage][ntrial])[0], np.shape(matrices_0[ntrial])[0]), :] /
                                                matrices_0[ntrial][:min(np.shape(matrices[stage][ntrial])[0], np.shape(matrices_0[ntrial])[0]), :]
                                                for ntrial in range(n_trials_conf)], axis=0) for stage in stages_id}
            else:
                matrix_ratio = {stage: np.mean([matrices[stage][ntrial][:, :min(np.shape(matrices[stage][ntrial])[1], np.shape(matrices_0[ntrial])[1])] /
                                                matrices_0[ntrial][:, :min(np.shape(matrices[stage][ntrial])[1], np.shape(matrices_0[ntrial])[1])]
                                                for ntrial in range(n_trials_conf)], axis=0) for stage in stages_id}
            if conn_type == 'cx_cx':
                conn_specs[conn_type]['data'][conf] = {'mean': matrix_mean, 'std': matrix_std, 'ratio': matrix_ratio, 'trials_list': trials_id_conf, 'trials_failed': trials_failed_conf, 'rngs': rngs_conf}
            else:
                conn_specs[conn_type]['data'][conf] = {'data': matrices[stages_id[-1]][-1] if n_trials_conf > 0 else None, 'mean': matrix_mean,
                                                       'std': matrix_std, 'ratio': matrix_ratio, 'trials_list': trials_id_conf, 'trials_failed': trials_failed_conf, 'rngs': rngs_conf}

        sum_specs = {'cx': {'store': w_cx_sum, 'input': {'cx': ('cx_cx', 0), 'th': ('th_cx', 0)}, 'output': {'cx': ('cx_cx', 1), 'th': ('cx_th', 1)}},
                     'th': {'store': w_th_sum, 'input': {'cx': ('cx_th', 0), 'th': None}, 'output': {'cx': ('th_cx', 1), 'th': None}}}
        for to_layer in ['cx', 'th']:
            w_input, w_output = {}, {}
            for from_layer in ['cx', 'th']:
                input_spec = sum_specs[to_layer]['input'][from_layer]
                output_spec = sum_specs[to_layer]['output'][from_layer]
                if input_spec is None:
                    w_input[from_layer] = {stage_id: [None for ntrial in range(n_trials_conf)] for stage_id in stages_id}
                else:
                    conn_type, axis = input_spec
                    w_input[from_layer] = {stage_id: [matrices_dict[conn_type][stage_id][ntrial].sum(axis=axis) for ntrial in range(n_trials_conf)] for stage_id in stages_id}
                if output_spec is None:
                    w_output[from_layer] = {stage_id: [None for ntrial in range(n_trials_conf)] for stage_id in stages_id}
                else:
                    conn_type, axis = output_spec
                    w_output[from_layer] = {stage_id: [matrices_dict[conn_type][stage_id][ntrial].sum(axis=axis) for ntrial in range(n_trials_conf)] for stage_id in stages_id}
            sum_specs[to_layer]['store'][conf] = {'input': w_input, 'output': w_output, 'synaptic_activity': None, 'trials_list': trials_id_conf, 'trials_failed': trials_failed_conf, 'rngs': rngs_conf}

        ncycles_conf = np.unique([ncycle_stage[stage_id] for stage_id in stages_id])
        syn_cat_mask_dict = {}
        for conn_type in ['cx_cx', 'cx_th', 'th_cx']:
            if not conn_specs[conn_type]['mask_key']:
                continue
            conn_specs[conn_type]['mask_store'][conf] = {}
            for ncycle in ncycles_conf:
                if conn_type == 'cx_cx':
                    if train_example_shuffling:
                        conn_specs[conn_type]['mask_store'][conf][ncycle] = {trial_id: SynCategoriesMask(conn_type, labels=labels_train[conf][trial_id], net_params=net_params, n_cycles=ncycle)
                                                                             for trial_id in trials_id_conf}
                    else:
                        syn_cat_mask_trial = SynCategoriesMask(conn_type, labels='same', net_params=net_params, n_cycles=ncycle)
                        conn_specs[conn_type]['mask_store'][conf][ncycle] = {trial_id: syn_cat_mask_trial for trial_id in trials_id_conf}
                else:
                    conn_specs[conn_type]['mask_store'][conf][ncycle] = {trial_id: SynCategoriesMask(conn_type, th_features=th_features_train[conf][trial_id],
                                                                                                      labels=labels_train[conf][trial_id], net_params=net_params,
                                                                                                      n_cycles=ncycle)
                                                                         for trial_id in trials_id_conf}
            syn_cat_mask_dict[conn_type] = conn_specs[conn_type]['mask_store'][conf]

        p13_cx_ids, trials_id_p13, indices_trials = None, None, None

        for conn_type in ['cx_cx', 'cx_th', 'th_cx']:
            mask_key = conn_specs[conn_type]['mask_key']
            category_key = syn_cats_key or (conn_type == 'cx_cx' and syn_matrix_trial_key)
            if not mask_key or not category_key:
                continue

            matrices = matrices_dict[conn_type]
            syn_cat_mask = syn_cat_mask_dict[conn_type]
            w_cats_mu, w_cats_median, w_cats_std = {}, {}, {}
            for cat in syn_cats:
                w_cats_mu[cat], w_cats_median[cat], w_cats_std[cat] = {}, {}, {}
                for stage in stages_id:
                    w_cats_mu[cat][stage] = np.array([np.mean(matrices[stage][ntrial][syn_cat_mask[ncycle_stage[stage]][trial_id][cat]]) for ntrial, trial_id in enumerate(trials_id_conf)])
                    w_cats_median[cat][stage] = np.array([np.median(matrices[stage][ntrial][syn_cat_mask[ncycle_stage[stage]][trial_id][cat]]) for ntrial, trial_id in enumerate(trials_id_conf)])
                    w_cats_std[cat][stage] = np.array([np.std(matrices[stage][ntrial][syn_cat_mask[ncycle_stage[stage]][trial_id][cat]]) / np.sqrt(np.sum(syn_cat_mask[ncycle_stage[stage]][trial_id][cat])) for ntrial, trial_id in enumerate(trials_id_conf)])

            if syn_cats_key:
                data_path = os.path.join(save_paths[conf], conn_specs[conn_type]['cats_file'])
                np.save(data_path, {'data': None, 'mean': w_cats_mu, 'median': w_cats_median, 'std': w_cats_std,
                                    'trials_list': trials_id_conf, 'trials_failed': trials_failed_conf, 'rngs': rngs_conf})

            if conn_type == 'cx_cx' and syn_matrix_trial_key:
                w_class_mean_trials = np.array([w_cats_mu['class'][stage] for stage in stages_id])
                p1, p3 = np.percentile(w_class_mean_trials[-1], 40), np.percentile(w_class_mean_trials[-1], 60)
                p13_cx_ids = [ntrial for ntrial, w_trial in enumerate(w_class_mean_trials[-1]) if w_trial > p1 and w_trial < p3]
                if not p13_cx_ids:
                    p13_cx_ids = np.arange(n_trials_conf, dtype=int)
                w_matrix_trials = {stage: [matrices[stage][idx] for idx in p13_cx_ids] for stage in [stages_id[0], stages_id[-1]]}
                trials_id_p13 = [trials_id_conf[ntrial] for ntrial in p13_cx_ids]
                indices_trials = [trial_dictionaries[conf][trial_id] for trial_id in trials_id_p13]
                conn_specs[conn_type]['data'][conf]['data'] = {'matrices': w_matrix_trials, 'indices': indices_trials, 'trials_id': trials_id_p13}

        if syn_matrix_trial_key:
            for conn_type in ['cx_th', 'th_cx']:
                w_matrix_trials = {stage: [matrices_dict[conn_type][stage][idx] for idx in p13_cx_ids] for stage in [stages_id[0], stages_id[-1]]}
                conn_specs[conn_type]['data'][conf]['data'] = {'matrices': w_matrix_trials, 'indices': indices_trials, 'trials_id': trials_id_p13}

        if syn_activity_key:
            fr_dict = {'cx': {stage_id: np.asarray([spikes_count_neuron[conf]['cx']['exc'][stage_id]['classification'][trial_id] for trial_id in trials_id_conf]) for stage_id in stages_id},
                       'th': {stage_id: np.asarray([spikes_count_neuron[conf]['th']['exc'][stage_id]['classification'][trial_id] for trial_id in trials_id_conf]) for stage_id in stages_id}}
            fr_inh_dict = {'cx': {stage_id: np.asarray([spikes_count_neuron[conf]['cx']['inh'][stage_id]['classification'][trial_id] for trial_id in trials_id_conf]) for stage_id in stages_id},
                           'th': {stage_id: np.asarray([spikes_count_neuron[conf]['th']['inh'][stage_id]['classification'][trial_id] for trial_id in trials_id_conf]) for stage_id in stages_id}}
            activity_specs = {'cx': {'cx': 'cx_cx', 'th': 'th_cx'}, 'th': {'cx': 'cx_th', 'th': None}}
            for to_layer, sum_store in [('cx', w_cx_sum), ('th', w_th_sum)]:
                syn_activity = {}
                for from_layer in ['cx', 'th']:
                    conn_type = activity_specs[to_layer][from_layer]
                    if conn_type is None:
                        syn_activity[from_layer] = {stage_id: [None for ntrial in range(n_trials_conf)] for stage_id in stages_id}
                    else:
                        syn_activity[from_layer] = {stage_id: [fr_dict[from_layer][stage_id][ntrial] @ matrices_dict[conn_type][stage_id][ntrial] for ntrial in range(n_trials_conf)] for stage_id in stages_id}
                if to_layer == 'cx':
                    syn_activity['inh'] = {stage_id: [cx_inh_exc * np.sum(fr_inh_dict['cx'][stage_id][ntrial]) * np.ones(np.shape(matrices_dict['cx_cx'][stage_id][ntrial])[1]) for ntrial in range(n_trials_conf)] for stage_id in stages_id}
                elif to_layer == 'th':
                    syn_activity['inh'] = {stage_id: [th_inh_exc * np.sum(fr_inh_dict['th'][stage_id][ntrial]) * np.ones(np.shape(matrices_dict['cx_th'][stage_id][ntrial])[1]) for ntrial in range(n_trials_conf)] for stage_id in stages_id}
                sum_store[conf]['synaptic_activity'] = syn_activity

#----------------------------- SAVE DATA
for nconf, conf in enumerate(configs):
    trials_id_conf, rngs_conf = trials_id_sorted[conf], rngs_sorted[conf]
    trials_failed_conf = trials_incomplete[conf]
    n_trials_conf = len(trials_id_conf)

    data_path = os.path.join(save_paths[conf], 'trial_dicts.npy')
    np.save(data_path, [trial_dictionaries[conf][trial_id] for trial_id in trials_id_conf])

    if accuracy_key:
        accuracy_full = {pred: {stage: np.array([accuracy[conf][pred][stage][trial_id] for trial_id in trials_id_conf]) for stage in stages_id} for pred in prediction}
        accuracy_mean = {pred: {stage: np.mean(accuracy_full[pred][stage]) for stage in stages_id} for pred in prediction}
        accuracy_std = {pred: {stage: np.std(accuracy_full[pred][stage]) / np.sqrt(n_trials_conf) for stage in stages_id} for pred in prediction}

        data_path = os.path.join(save_paths[conf], 'accuracy.npy')
        np.save(data_path, {'data': accuracy_full, 'mean': accuracy_mean, 'std': accuracy_std,
                            'trials_list': trials_id_conf, 'trials_failed': trials_failed_conf, 'rngs': rngs_conf})

    if spikes_count_key:
        spikes_count_dict = {pop: {neur_type: {stage: {substage: np.array([spikes_count_neuron[conf][pop][neur_type][stage][substage][trial_id] for trial_id in trials_id_conf], dtype=float)
                                                       for substage in spikes_count_substages[stage[3:]]} for stage in stages_id} for neur_type in spikes_count_pops} for pop in ['th', 'cx']}
        data_path = os.path.join(save_paths[conf], 'spikes_count.npy')
        np.save(data_path, {'data': spikes_count_dict, 'n_trials': n_trials_conf, 't_img_presentation': t_img_test,
                            'trials_list': trials_id_conf, 'trials_failed': trials_failed_conf, 'rngs': rngs_conf})

    if firing_rate_key:
        for conv_type in conv_types:
            dt_s, dt_ds, nu_t_high, nu_t_low = dt_s_fr[conv_type], dt_ds_fr[conv_type], nu_t_high_fr[conv_type], nu_t_low_fr[conv_type]
            raster_cx = [np.concatenate([spikes_neur_stage[nneur] for spikes_neur_stage in spikes_cx[conf]]) for nneur in range(n_neur_cx)]
            raster_th = [np.concatenate([spikes_neur_stage[nneur] for spikes_neur_stage in spikes_th[conf]]) for nneur in range(n_neur_th)]
            fr_cx = FiringRate(raster_cx[:180], t_start_train[nconf], t_start_train[nconf] + t_tot, dt_s=dt_s, dt_ds=dt_ds, nu_t_high=nu_t_high, nu_t_low=nu_t_low, remove_pause=None)
            fr_th = FiringRate(raster_th, t_start_train[nconf], t_start_train[nconf] + t_tot, dt_s=dt_s, dt_ds=dt_ds, nu_t_high=nu_t_high, nu_t_low=nu_t_low, remove_pause=None)
            data_path = os.path.join(save_paths[conf], f'lfp_single_track_{conv_type}.npy')
            np.save(data_path, {'data': {'cx': {'fr': fr_cx, 'raster': raster_cx}, 'th': {'fr': fr_th, 'raster': raster_th}},
                                't_img_presentation': t_img_test, 'dt_s': dt_s, 'dt_ds': dt_ds, 'nu_t_high': nu_t_high, 'nu_t_low': nu_t_low,
                                'trials_list': [trials_id_loaded[conf][0]] if len(trials_id_loaded[conf]) > 0 else [],
                                'trials_failed': trials_failed_conf, 'rngs': {'nest_seed': [rngs_dict[conf][0][1]], 'numpy_seed': [rngs_dict[conf][0][2]]} if len(rngs_dict[conf]) > 0 else {'nest_seed': [], 'numpy_seed': []}})

    if oscillations_key:
        osc_dict = {stage: {substage: [groups_oscillations[conf][stage][trial_id][substage] for trial_id in trials_id_conf if trial_id in groups_oscillations[conf][stage]]
                            for substage in substages_dict[stage[3:]]} for stage in stages_id}
        osc_mean = {stage: {substage: [np.mean(arr) for arr in osc_dict[stage][substage]] for substage in substages_dict[stage[3:]]} for stage in stages_id}
        osc_std = {stage: {substage: [np.std(arr) / np.sqrt(len(arr)) for arr in osc_dict[stage][substage]] for substage in substages_dict[stage[3:]]} for stage in stages_id}
        data_path = os.path.join(save_paths[conf], 'cx_groups_oscillations.npy')
        np.save(data_path, {'data': osc_dict, 'mean': osc_mean, 'std': osc_std, 'n_trials': n_trials_conf, 'dt_fr': dt_fr,
                            'thresh_fr': thresh_fr, 'thresh_osc': thresh_osc, 'sigma_t_osc_nrem': sigma_t_osc_nrem,
                            'sigma_t_osc_awake': sigma_t_osc_awake, 'trials_list': trials_id_conf, 'trials_failed': trials_failed_conf, 'rngs': rngs_conf})

    if synapses_key:
        data_path = os.path.join(save_paths[conf], 'w_cx_sum.npy')
        np.save(data_path, w_cx_sum[conf])

        data_path = os.path.join(save_paths[conf], 'w_th_sum.npy')
        np.save(data_path, w_th_sum[conf])

        data_path = os.path.join(save_paths[conf], 'w_th_cx_matrix.npy')
        np.save(data_path, matrix_data_th_cx[conf])

        data_path = os.path.join(save_paths[conf], 'w_cx_th_matrix.npy')
        np.save(data_path, matrix_data_cx_th[conf])

        data_path = os.path.join(save_paths[conf], 'w_cx_cx_matrix.npy')
        np.save(data_path, matrix_data_cx_cx[conf])

    if dataset_similarity_key:
        features_dataset, labels_dataset = GetDatasetFeatures(input_models[nconf], 'training', root_input_path)
        features_dataset, labels_dataset = DatasetClassSample(features_dataset, labels_dataset, max_features_per_class,
                                                              seed=dataset_similarity_seed)

        data_path = os.path.join(save_paths[conf], 'dataset_similarity_matrix.npy')
        np.save(data_path, DatasetSimilarityMatrix(features_dataset, labels_dataset, norm_factor=dataset_similarity_norm))

        data_path = os.path.join(save_paths[conf], 'dataset_intra_class_similarity.npy')
        np.save(data_path, DatasetSimilarityHistogram(features_dataset, labels_dataset, norm_factor=dataset_similarity_norm,
                                                      mode='intra'))

        data_path = os.path.join(save_paths[conf], 'dataset_inter_class_similarity.npy')
        np.save(data_path, DatasetSimilarityHistogram(features_dataset, labels_dataset, norm_factor=dataset_similarity_norm,
                                                      mode='inter'))

        data_path = os.path.join(save_paths[conf], f'dataset_umap_{umap_n_components}d.npy')
        np.save(data_path, DatasetUmap(features_dataset, labels_dataset, n_components=umap_n_components,
                                       max_samples=max_umap_samples, seed=dataset_similarity_seed,
                                       n_neighbors=umap_n_neighbors, min_dist=umap_min_dist))

time_total_stop = time.time()
time_execution = SecondsConverter(time_total_stop - time_total_start)
print(f'\nExecution time: {time_execution[0]}h {time_execution[1]}m {int(time_execution[2])}s')
