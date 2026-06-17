import numpy as np
import os.path
import gc, time
from lib import (LoadWeights, NetAccuracy, MemoryUsage, SecondsConverter, FiringRate, SpikesCount,
                 GroupsActivation, GetDatasetFeatures, SynCategoriesMask,
                 DatasetClassSample, DatasetSimilarityHistogram, DatasetSimilarityMatrix, DatasetUmap,
                 UpStateEvents, SleepSynchronization, SleepReactivation, Mask, LoadDictFromYaml)

config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
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
up_state_config = config_dict['up_state']
synchronization_config = config_dict['synchronization']
reactivation_config = config_dict['reactivation']

debug = execution_config['debug']

accuracy_key = analysis_keys_config['accuracy_key']
spikes_count_key = analysis_keys_config['spikes_count_key']
firing_rate_key = analysis_keys_config['firing_rate_single_key']
oscillations_key = analysis_keys_config['oscillations_key']
syn_matrix_trial_key = analysis_keys_config['syn_matrix_trial_key']
syn_cats_key = analysis_keys_config['syn_cats_key']
syn_activity_key = analysis_keys_config['syn_activity_key']
dataset_similarity_key = analysis_keys_config['dataset_similarity_key']
up_state_key = analysis_keys_config['up_state_key']
synchronization_key = analysis_keys_config['synchronization_key']
sleep_reactivation_key = analysis_keys_config['sleep_reactivation_key']

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

#SLEEP REACTIVATION
up_state_population_firing_rate_config = up_state_config['population_firing_rate']
up_state_detection_config = up_state_config['detection']
synchronization_pairs = tuple(tuple(pair) for pair in synchronization_config['pairs'])
synchronization_population_keys = tuple(sorted(set([population_key for pair in synchronization_pairs for population_key in pair])))
reactivation_firing_rate_config = reactivation_config['firing_rate_single']
reactivation_dt_react = reactivation_firing_rate_config['dt_ds_fr'] if reactivation_firing_rate_config['dt_ds_fr'] > 0 else reactivation_firing_rate_config['dt_s_fr']
up_state_dt_up_detect = up_state_population_firing_rate_config['dt_ds'] if up_state_population_firing_rate_config['dt_ds'] > 0 else up_state_population_firing_rate_config['dt_s']
reactivation_event_padding_bins = max(int(round(reactivation_firing_rate_config['event_padding_ms'] / reactivation_dt_react)), 0)
synchronization_event_padding_bins = max(int(round(synchronization_config['event_padding_ms'] / up_state_dt_up_detect)), 0)
reactivation_analysis_stage_ids = [stage_id for stage_id in stages_id if stage_id.endswith('_nrem')]
reactivation_stage_substages = {stage_id: [reactivation_config['analysis_substage']] for stage_id in reactivation_analysis_stage_ids}
reactivation_areas = ('cx', 'th')
reactivation_n_seq_train = n_ranks_train * n_class
reactivation_template_class_id = np.repeat(np.arange(n_class, dtype=int), n_ranks_train)
synchronization_inh_keys = tuple(population_key for population_key in synchronization_population_keys if population_key.endswith('_inh'))
up_state_measure_key = up_state_key or synchronization_key or sleep_reactivation_key

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
params['up_state'] = {
    'dt': {
        'population_firing_rate': up_state_population_firing_rate_config['dt_s'],
        'up_state_detection': up_state_dt_up_detect},
    'population_firing_rate': up_state_population_firing_rate_config,
    'detection': up_state_detection_config}
params['synchronization'] = synchronization_config | {
    'pairs': synchronization_pairs,
    'population_keys': synchronization_population_keys,
    'event_padding_bins': synchronization_event_padding_bins}
params['reactivation'] = {
    'analysis_stage_ids': reactivation_analysis_stage_ids,
    'stage_substages': reactivation_stage_substages,
    'template_substage': reactivation_config['template_substage'],
    'analysis_substage': reactivation_config['analysis_substage'],
    'representative_trial_id': reactivation_config['representative_trial_id'],
    'dt': {
        'firing_rate': reactivation_firing_rate_config['dt_s_fr'],
        'reactivation': reactivation_dt_react},
    'firing_rate_single': reactivation_firing_rate_config | {
        'event_padding_bins': reactivation_event_padding_bins},
    'templates': {
        'n_seq_train': reactivation_n_seq_train,
        'class_id': reactivation_template_class_id}}

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
up_states_data = {conf: {
    'events': {event_key: {stage_id: {substage: {} for substage in reactivation_stage_substages[stage_id]}
                           for stage_id in reactivation_analysis_stage_ids}
               for event_key in ('up_state_id', 'tstart_ms', 'tstop_ms', 'tpeak_ms', 'duration_ms')},
    'down_state_duration_ms': {stage_id: {substage: {} for substage in reactivation_stage_substages[stage_id]}
                               for stage_id in reactivation_analysis_stage_ids},
    'iwi_ms': {stage_id: {substage: {} for substage in reactivation_stage_substages[stage_id]}
               for stage_id in reactivation_analysis_stage_ids},
    'firing_rate': {area: {stage_id: {substage: {} for substage in reactivation_stage_substages[stage_id]}
                           for stage_id in reactivation_analysis_stage_ids}
                    for area in reactivation_areas}} for conf in configs}
reactivation_data = {conf: {
    'similarity': {
        'templates': {area: {stage_id: {substage: {} for substage in reactivation_stage_substages[stage_id]}
                             for stage_id in reactivation_analysis_stage_ids}
                      for area in reactivation_areas},
        'best_template_id': {area: {stage_id: {substage: {} for substage in reactivation_stage_substages[stage_id]}
                                    for stage_id in reactivation_analysis_stage_ids}
                             for area in reactivation_areas},
        'time_resolved_best_template': {area: {stage_id: {substage: {} for substage in reactivation_stage_substages[stage_id]}
                                               for stage_id in reactivation_analysis_stage_ids}
                                        for area in reactivation_areas},
        'collected_best_template': {area: {stage_id: {substage: {} for substage in reactivation_stage_substages[stage_id]}
                                           for stage_id in reactivation_analysis_stage_ids}
                                    for area in reactivation_areas}},
    'strength': {
        'templates': {area: {stage_id: {substage: {} for substage in reactivation_stage_substages[stage_id]}
                             for stage_id in reactivation_analysis_stage_ids}
                      for area in reactivation_areas},
        'best_template': {area: {stage_id: {substage: {} for substage in reactivation_stage_substages[stage_id]}
                                 for stage_id in reactivation_analysis_stage_ids}
                          for area in reactivation_areas},
        'collected_best_template': {area: {stage_id: {substage: {} for substage in reactivation_stage_substages[stage_id]}
                                           for stage_id in reactivation_analysis_stage_ids}
                                    for area in reactivation_areas}}} for conf in configs}
synchronization_data = {conf: {
    'population_firing_rate_trace': {population_key: {stage_id: {substage: {} for substage in reactivation_stage_substages[stage_id]}
                                                      for stage_id in reactivation_analysis_stage_ids}
                                     for population_key in synchronization_population_keys},
    'transition_time': {time_key: {population_key: {stage_id: {substage: {} for substage in reactivation_stage_substages[stage_id]}
                                                    for stage_id in reactivation_analysis_stage_ids}
                                   for population_key in synchronization_population_keys}
                        for time_key in ('tstart_ms', 'tstop_ms', 'activation_time_ms', 'deactivation_time_ms')}} for conf in configs}
similarity_single_track_data = {conf: {
    'raster_cx': {stage_id: {substage: np.array([], dtype=object) for substage in reactivation_stage_substages[stage_id]}
                  for stage_id in reactivation_analysis_stage_ids},
    'similarity': {area: {stage_id: {substage: np.empty((0, reactivation_n_seq_train)) for substage in reactivation_stage_substages[stage_id]}
                          for stage_id in reactivation_analysis_stage_ids}
                   for area in reactivation_areas},
    'strength': {area: {stage_id: {substage: np.empty((0, reactivation_n_seq_train)) for substage in reactivation_stage_substages[stage_id]}
                        for stage_id in reactivation_analysis_stage_ids}
                 for area in reactivation_areas}} for conf in configs}

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
spikes_cx_exc_key = accuracy_key or firing_rate_key or oscillations_key or up_state_measure_key or 'exc' in spikes_count_neuron_measure_pops
spikes_cx_inh_key = 'inh' in spikes_count_neuron_measure_pops or (synchronization_key and 'cx_inh' in synchronization_inh_keys)
spikes_th_exc_key = firing_rate_key or up_state_measure_key or 'exc' in spikes_count_neuron_measure_pops
spikes_th_inh_key = 'inh' in spikes_count_neuron_measure_pops or (synchronization_key and 'th_inh' in synchronization_inh_keys)
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
        reactivation_template_leaf = None
        reactivation_representative_trial_id = params['reactivation']['representative_trial_id']
        if sleep_reactivation_key and (reactivation_representative_trial_id is None or reactivation_representative_trial_id not in trials_id_sorted[conf]):
            reactivation_representative_trial_id = trials_id_sorted[conf][0]

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

            if up_state_measure_key:
                if stage == 'awake_training':
                    if sleep_reactivation_key:
                        substage = params['reactivation']['template_substage']
                        spikes_leaf = {'cx': np.array(spikes_cx_trial, dtype=object),
                                       'th': np.array(spikes_th_trial, dtype=object)}
                        reactivation_template_leaf = SleepReactivation(spikes_leaf, None, stage, substage, params, nconf=nconf)

                elif stage == 'nrem' and stage_id in reactivation_analysis_stage_ids:
                    substage = params['reactivation']['analysis_substage']
                    up_state = params['up_state']
                    time_window = params['times'][stage][substage]
                    tstart, tstop = time_window['start'][nconf], time_window['stop'][nconf]
                    population_firing_rate_config = up_state['population_firing_rate']
                    spikes_leaf = {'cx': np.array(spikes_cx_trial, dtype=object),
                                   'th': np.array(spikes_th_trial, dtype=object)}
                    population_firing_rate_leaf = {
                        area: FiringRate(spikes_leaf[area], tstart, tstop,
                                         population_firing_rate_config['dt_s'], population_firing_rate_config['dt_ds'],
                                         population_firing_rate_config['nu_t_high'], population_firing_rate_config['nu_t_low'],
                                         remove_zeros=False, remove_pause=False)
                        for area in reactivation_areas}

                    up_state_leaf = UpStateEvents(spikes_leaf, population_firing_rate_leaf, stage, substage, params, nconf=nconf)
                    if up_state_key:
                        up_states_trial = up_state_leaf['data']
                        for event_key in up_states_trial['events'].keys():
                            up_states_data[conf]['events'][event_key][stage_id][substage][trial_id] = up_states_trial['events'][event_key]
                        up_states_data[conf]['down_state_duration_ms'][stage_id][substage][trial_id] = up_states_trial['down_state_duration_ms']
                        up_states_data[conf]['iwi_ms'][stage_id][substage][trial_id] = up_states_trial['iwi_ms']
                        for area in reactivation_areas:
                            up_states_data[conf]['firing_rate'][area][stage_id][substage][trial_id] = up_states_trial['firing_rate'][area]

                    if synchronization_key:
                        events_leaf = {
                            'cx': {'evt_inh': np.array(spikes_cx_inh_trial, dtype=object) if spikes_cx_inh_trial is not None else np.array([], dtype=object)},
                            'th': {'evt_inh': np.array(spikes_th_inh_trial, dtype=object) if spikes_th_inh_trial is not None else np.array([], dtype=object)}}
                        synchronization_leaf = SleepSynchronization(events_leaf, up_state_leaf, stage, substage, params, nconf=nconf)
                        for population_key in synchronization_leaf['population_firing_rate_trace'].keys():
                            synchronization_data[conf]['population_firing_rate_trace'][population_key][stage_id][substage][trial_id] = synchronization_leaf['population_firing_rate_trace'][population_key]
                        for time_key in synchronization_leaf['transition_time'].keys():
                            for population_key in synchronization_leaf['transition_time'][time_key].keys():
                                synchronization_data[conf]['transition_time'][time_key][population_key][stage_id][substage][trial_id] = synchronization_leaf['transition_time'][time_key][population_key]

                    if sleep_reactivation_key:
                        is_representative_trial = trial_id == reactivation_representative_trial_id
                        reactivation_leaf = SleepReactivation(spikes_leaf, up_state_leaf, stage, substage, params, nconf=nconf,
                                                              template_leaf=reactivation_template_leaf, representative=is_representative_trial)
                        for measure_key in reactivation_leaf['reactivation'].keys():
                            for observable_key in reactivation_leaf['reactivation'][measure_key].keys():
                                for area in reactivation_areas:
                                    reactivation_data[conf][measure_key][observable_key][area][stage_id][substage][trial_id] = reactivation_leaf['reactivation'][measure_key][observable_key][area]

                        if is_representative_trial and reactivation_leaf['single_track'] is not None:
                            similarity_single_track_data[conf]['raster_cx'][stage_id][substage] = np.array([
                                (Mask(spikes_neur, tstart, tstop) - tstart) / 1000.0
                                for spikes_neur in spikes_leaf['cx']], dtype=object)
                            for area in reactivation_areas:
                                similarity_single_track_data[conf]['similarity'][area][stage_id][substage] = reactivation_leaf['single_track']['similarity'][area]
                                similarity_single_track_data[conf]['strength'][area][stage_id][substage] = reactivation_leaf['single_track']['strength'][area]

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

    if up_state_measure_key:
        up_state = params['up_state']
        synchronization = params['synchronization']
        reactivation = params['reactivation']
        reactivation_save_path = save_paths[conf]

        up_state_save_dt = {'population_firing_rate': up_state['dt']['population_firing_rate'],
                            'up_state_detection': up_state['dt']['up_state_detection']}
        react_save_dt = {'firing_rate': reactivation['dt']['firing_rate'],
                         'reactivation': reactivation['dt']['reactivation']}
        sync_save_dt = {'population_firing_rate': up_state['dt']['population_firing_rate'],
                        'up_state_detection': up_state['dt']['up_state_detection'],
                        'synchronization': up_state['dt']['up_state_detection']}
        up_state_save_params = {'areas': reactivation_areas, 'up_state': up_state['detection'],
                                'analysis_substage': reactivation['analysis_substage'],
                                'performance_stage_ids': reactivation['analysis_stage_ids']}
        react_save_params = {'areas': reactivation_areas, 'n_class': n_class, 'n_ranks_train': n_ranks_train,
                             'n_seq_train': reactivation['templates']['n_seq_train'],
                             'event_firing_rate_padding_ms': reactivation['firing_rate_single']['event_padding_ms'],
                             'event_firing_rate_padding_bins': reactivation['firing_rate_single']['event_padding_bins'],
                             'template_class_id': reactivation['templates']['class_id']}
        sync_save_params = {'synchronization_pairs': synchronization['pairs'],
                            'synchronization_population_keys': synchronization['population_keys'],
                            'event_padding_ms': synchronization['event_padding_ms']}

        if up_state_key:
            up_states_trials_data = {
                'events': {event_key: {stage_id: {substage: [
                    up_states_data[conf]['events'][event_key][stage_id][substage][trial_id]
                    for trial_id in trials_id_conf]
                    for substage in reactivation_stage_substages[stage_id]}
                    for stage_id in reactivation_analysis_stage_ids}
                    for event_key in up_states_data[conf]['events'].keys()},
                'down_state_duration_ms': {stage_id: {substage: [
                    up_states_data[conf]['down_state_duration_ms'][stage_id][substage][trial_id]
                    for trial_id in trials_id_conf]
                    for substage in reactivation_stage_substages[stage_id]}
                    for stage_id in reactivation_analysis_stage_ids},
                'iwi_ms': {stage_id: {substage: [
                    up_states_data[conf]['iwi_ms'][stage_id][substage][trial_id]
                    for trial_id in trials_id_conf]
                    for substage in reactivation_stage_substages[stage_id]}
                    for stage_id in reactivation_analysis_stage_ids},
                'firing_rate': {area: {stage_id: {substage: [
                    up_states_data[conf]['firing_rate'][area][stage_id][substage][trial_id]
                    for trial_id in trials_id_conf]
                    for substage in reactivation_stage_substages[stage_id]}
                    for stage_id in reactivation_analysis_stage_ids}
                    for area in reactivation_areas}}
            data_path = os.path.join(reactivation_save_path, 'up_states.npy')
            np.save(data_path, {'data': up_states_trials_data, 'n_trials': n_trials_conf,
                                'trials_list': trials_id_conf, 'trials_failed': trials_failed_conf, 'rngs': rngs_conf,
                                'dt': up_state_save_dt, 'params': up_state_save_params}, allow_pickle=True)

        if synchronization_key:
            sync_trials_data = {
                'population_firing_rate_trace': {population_key: {stage_id: {substage: [
                    synchronization_data[conf]['population_firing_rate_trace'][population_key][stage_id][substage][trial_id]
                    for trial_id in trials_id_conf]
                    for substage in reactivation_stage_substages[stage_id]}
                    for stage_id in reactivation_analysis_stage_ids}
                    for population_key in synchronization_population_keys},
                'transition_time': {time_key: {population_key: {stage_id: {substage: [
                    synchronization_data[conf]['transition_time'][time_key][population_key][stage_id][substage][trial_id]
                    for trial_id in trials_id_conf]
                    for substage in reactivation_stage_substages[stage_id]}
                    for stage_id in reactivation_analysis_stage_ids}
                    for population_key in synchronization_population_keys}
                    for time_key in synchronization_data[conf]['transition_time'].keys()}}
            data_path = os.path.join(reactivation_save_path, 'synchronization.npy')
            np.save(data_path, {'data': sync_trials_data, 'n_trials': n_trials_conf,
                                'trials_list': trials_id_conf, 'trials_failed': trials_failed_conf, 'rngs': rngs_conf,
                                'dt': sync_save_dt, 'params': sync_save_params}, allow_pickle=True)

        if sleep_reactivation_key:
            representative_trial_id = reactivation['representative_trial_id']
            if representative_trial_id is None or representative_trial_id not in trials_id_conf:
                representative_trial_id = trials_id_conf[0] if n_trials_conf > 0 else None
            representative_trial_idx = trials_id_conf.index(representative_trial_id) if representative_trial_id in trials_id_conf else None
            representative_rngs = {'nest_seed': np.array([rngs_conf['nest_seed'][representative_trial_idx]], dtype=object),
                                   'numpy_seed': np.array([rngs_conf['numpy_seed'][representative_trial_idx]], dtype=object)} if representative_trial_idx is not None else {'nest_seed': np.array([], dtype=object), 'numpy_seed': np.array([], dtype=object)}
            react_trials_data = {
                'similarity': {
                    'templates': {area: {stage_id: {substage: [
                        reactivation_data[conf]['similarity']['templates'][area][stage_id][substage][trial_id]
                        for trial_id in trials_id_conf]
                        for substage in reactivation_stage_substages[stage_id]}
                        for stage_id in reactivation_analysis_stage_ids}
                        for area in reactivation_areas},
                    'best_template_id': {area: {stage_id: {substage: [
                        reactivation_data[conf]['similarity']['best_template_id'][area][stage_id][substage][trial_id]
                        for trial_id in trials_id_conf]
                        for substage in reactivation_stage_substages[stage_id]}
                        for stage_id in reactivation_analysis_stage_ids}
                        for area in reactivation_areas},
                    'time_resolved_best_template': {area: {stage_id: {substage: [
                        reactivation_data[conf]['similarity']['time_resolved_best_template'][area][stage_id][substage][trial_id]
                        for trial_id in trials_id_conf]
                        for substage in reactivation_stage_substages[stage_id]}
                        for stage_id in reactivation_analysis_stage_ids}
                        for area in reactivation_areas},
                    'collected_best_template': {area: {stage_id: {substage: [
                        reactivation_data[conf]['similarity']['collected_best_template'][area][stage_id][substage][trial_id]
                        for trial_id in trials_id_conf]
                        for substage in reactivation_stage_substages[stage_id]}
                        for stage_id in reactivation_analysis_stage_ids}
                        for area in reactivation_areas}},
                'strength': {
                    'templates': {area: {stage_id: {substage: [
                        reactivation_data[conf]['strength']['templates'][area][stage_id][substage][trial_id]
                        for trial_id in trials_id_conf]
                        for substage in reactivation_stage_substages[stage_id]}
                        for stage_id in reactivation_analysis_stage_ids}
                        for area in reactivation_areas},
                    'best_template': {area: {stage_id: {substage: [
                        reactivation_data[conf]['strength']['best_template'][area][stage_id][substage][trial_id]
                        for trial_id in trials_id_conf]
                        for substage in reactivation_stage_substages[stage_id]}
                        for stage_id in reactivation_analysis_stage_ids}
                        for area in reactivation_areas},
                    'collected_best_template': {area: {stage_id: {substage: [
                        reactivation_data[conf]['strength']['collected_best_template'][area][stage_id][substage][trial_id]
                        for trial_id in trials_id_conf]
                        for substage in reactivation_stage_substages[stage_id]}
                        for stage_id in reactivation_analysis_stage_ids}
                        for area in reactivation_areas}}}
            single_track_save_params = react_save_params | {'representative_trial_id': representative_trial_id}
            data_path = os.path.join(reactivation_save_path, 'reactivation.npy')
            np.save(data_path, {'data': react_trials_data, 'n_trials': n_trials_conf,
                                'trials_list': trials_id_conf, 'trials_failed': trials_failed_conf, 'rngs': rngs_conf,
                                'dt': react_save_dt, 'params': react_save_params}, allow_pickle=True)
            data_path = os.path.join(reactivation_save_path, 'similarity_single_track.npy')
            np.save(data_path, {'data': similarity_single_track_data[conf], 'n_trials': 1 if representative_trial_id is not None else 0,
                                'trials_list': [representative_trial_id] if representative_trial_id is not None else [],
                                'trials_failed': trials_failed_conf, 'rngs': representative_rngs,
                                'dt': react_save_dt, 'params': single_track_save_params}, allow_pickle=True)

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
