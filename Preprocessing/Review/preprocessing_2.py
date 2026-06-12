import numpy as np
import os, sys
import psutil
import time
import yaml
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib import (CosineSimilarity, DetectUpStates, Mask,
                 LoadOrComputeFR, LoadOrComputePopulationFR, MapStatesToReferenceBins, MovingAverage)


#----------------------------- CONFIG
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(script_dir, 'config_2.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

model_config = config['model']
paths_config = config['paths']
network_config = config['network']
timing_config = config['timing']
firing_rate_config = config['firing_rate']
population_firing_rate_config = config['population_firing_rate']
up_state_config = config['up_state']
synchronization_config = config['synchronization']
preproc_config = config['preproc']


#----------------------------- MODEL CONFIG
input_type = model_config['input']
plasticity_type = model_config['plasticity']
model_type = f'{input_type}_{plasticity_type}plast'

n_class = network_config['n_class']
n_ranks_train = network_config['n_ranks_train']
n_ranks_test = network_config['n_ranks_test']
n_seq_train = n_class * n_ranks_train

areas = ('cx', 'th')


#----------------------------- PATHS
simdata_path = paths_config['simdata_path']
simdata_config = f'startreview_{model_type}'
root_save_path = paths_config['root_save_path']
root_preprocdata_path = paths_config['root_preprocdata_path']
fr_cache_path = os.path.join(root_save_path, 'FR_cache', model_type)
model_preprocdata_path = os.path.join(root_preprocdata_path, model_type)
up_states_path = os.path.join(model_preprocdata_path, 'up_states.npy')
reactivation_path = os.path.join(model_preprocdata_path, 'reactivation.npy')
synchronization_path = os.path.join(model_preprocdata_path, 'synchronization.npy')
similarity_single_track_path = os.path.join(model_preprocdata_path, 'similarity_single_track.npy')

os.makedirs(fr_cache_path, exist_ok=True)
os.makedirs(root_preprocdata_path, exist_ok=True)
os.makedirs(model_preprocdata_path, exist_ok=True)


#----------------------------- TIMING
t_img_train = timing_config['t_img_train']
t_img_test = timing_config['t_img_test']
t_pause = timing_config['t_pause']
t_relaxation = timing_config['t_relaxation']
t_nrem_therm = timing_config['t_nrem_therm']
t_nrem = timing_config['t_nrem']
t_start_train = timing_config['t_start_train']
t_start_nrem_therm = timing_config['t_start_nrem_therm']

t_training = n_ranks_train * n_class * (t_img_train + t_pause)
t_test = n_ranks_test * n_class * (t_img_test + t_pause)

t_stop_train = t_start_train + t_training
t_start_classification_pre = t_stop_train + t_relaxation
t_stop_classification_pre = t_start_classification_pre + t_test

t_stop_nrem_therm = t_start_nrem_therm + t_nrem_therm
t_start_nrem = t_stop_nrem_therm
t_stop_nrem = t_start_nrem + t_nrem

t_start_classification_post = t_stop_nrem + t_relaxation
t_stop_classification_post = t_start_classification_post + t_test

times_dict = {
    'awake_training': {
        'learning': {'start': t_start_train, 'stop': t_stop_train},
        'classification': {'start': t_start_classification_pre, 'stop': t_stop_classification_pre}},
    'nrem': {
        'thermalization': {'start': t_start_nrem_therm, 'stop': t_stop_nrem_therm},
        'sleep': {'start': t_start_nrem, 'stop': t_stop_nrem},
        'classification': {'start': t_start_classification_post, 'stop': t_stop_classification_post}}}

#----------------------------- FIRING RATE
dt_s_fr = firing_rate_config['dt_s_fr']
dt_ds_fr = firing_rate_config['dt_ds_fr']
nu_t_high_fr = firing_rate_config['nu_t_high_fr']
nu_t_low_fr = firing_rate_config['nu_t_low_fr']

#----------------------------- POPULATION FIRING RATE
dt_s_pop = population_firing_rate_config['dt_s']
dt_ds_pop = population_firing_rate_config['dt_ds']
nu_t_high_pop = population_firing_rate_config['nu_t_high']
nu_t_low_pop = population_firing_rate_config['nu_t_low']

#----------------------------- REACTIVATION PARAMETERS
dt_react = dt_ds_fr if dt_ds_fr > 0 else dt_s_fr
dt_up_detect = dt_ds_pop if dt_ds_pop > 0 else dt_s_pop

awake_template_substage = preproc_config['template_substage']
analysis_substage = preproc_config['analysis_substage']

#----------------------------- TRIALS AND STAGES
trial_simdata_path = os.path.join(simdata_path, 'MainOutput', simdata_config)
trials_id = sorted(os.listdir(trial_simdata_path)) #[:10]
if '.DS_Store' in trials_id: trials_id.remove('.DS_Store')
trial_path_dict = {trial_id: os.path.join(trial_simdata_path, trial_id) for trial_id in trials_id}
trials_id_conf = trials_id
trials_failed_conf = {}
rngs_conf = {
    'nest_seed': np.full(len(trials_id_conf), None, dtype=object),
    'numpy_seed': np.full(len(trials_id_conf), None, dtype=object)}
n_trials_conf = len(trials_id_conf)

analysis_stage_ids = sorted([stage_id for stage_id in os.listdir(trial_path_dict[trials_id[0]]) if stage_id.endswith('_nrem')])
stages_id = ['00_awake_training'] + analysis_stage_ids
stage_substages_map = {'00_awake_training': (awake_template_substage,)}
for stage_id in stages_id[1:]:
    stage_substages_map[stage_id] = (analysis_substage,)

plot_trial_id_config = preproc_config['representative_trial_id']
plot_trial_id = plot_trial_id_config if plot_trial_id_config in trials_id else trials_id[0]

analysis_up_state_config = up_state_config
compute_synchronization = synchronization_config['compute']
synchronization_pairs = tuple(tuple(pair) for pair in synchronization_config['pairs'])
synchronization_population_keys = tuple(sorted(set([population_key for pair in synchronization_pairs for population_key in pair])))
synchronization_downsample_bins = max(int(round(synchronization_config['downsample_ms'] / dt_up_detect)), 1)
synchronization_event_padding_bins = max(int(round(synchronization_config['event_padding_ms'] / dt_up_detect)), 0)
dt_synchronization = dt_up_detect * synchronization_downsample_bins

#----------------------------- OUTPUT METADATA
template_class_id = np.repeat(np.arange(n_class, dtype=int), n_ranks_train)
common_params = {
    'model_type': model_type,
    'input_type': input_type,
    'plasticity_type': plasticity_type,
    'analysis_substage': analysis_substage,
    'performance_stage_ids': analysis_stage_ids}
up_states_dt = {
    'population_firing_rate': dt_s_pop,
    'up_state_detection': dt_up_detect}
up_states_params = common_params | {
    'areas': areas,
    'up_state': analysis_up_state_config}
reactivation_dt = {
    'firing_rate': dt_s_fr,
    'reactivation': dt_react}
reactivation_params = common_params | {
    'areas': areas,
    'n_class': n_class,
    'n_ranks_train': n_ranks_train,
    'n_seq_train': n_seq_train,
    'template_class_id': template_class_id}
synchronization_dt = {
    'population_firing_rate': dt_s_pop,
    'up_state_detection': dt_up_detect,
    'synchronization': dt_synchronization}
synchronization_params = common_params | {
    'synchronization_pairs': synchronization_pairs,
    'synchronization_population_keys': synchronization_population_keys,
    'event_padding_ms': synchronization_config['event_padding_ms'],
    'downsample_ms': synchronization_config['downsample_ms']}
similarity_single_track_dt = {
    'firing_rate': dt_s_fr,
    'reactivation': dt_react}
similarity_single_track_params = common_params | {
    'areas': areas,
    'representative_trial_id': plot_trial_id,
    'template_class_id': template_class_id}

#----------------------------- DATA STRUCTURE
up_states_data = {
    'events': {event_key: {stage_id: {substage: {} for substage in stage_substages_map[stage_id]}
                           for stage_id in stages_id[1:]}
               for event_key in ('up_state_id', 'tstart_ms', 'tstop_ms', 'tpeak_ms', 'duration_ms')},
    'down_state_duration_ms': {stage_id: {substage: {} for substage in stage_substages_map[stage_id]}
                               for stage_id in stages_id[1:]},
    'iwi_ms': {stage_id: {substage: {} for substage in stage_substages_map[stage_id]}
               for stage_id in stages_id[1:]},
    'firing_rate': {area: {stage_id: {substage: {} for substage in stage_substages_map[stage_id]}
                           for stage_id in stages_id[1:]}
                    for area in areas}}

reactivation_data = {
    'similarity': {
        'templates': {area: {stage_id: {substage: {} for substage in stage_substages_map[stage_id]}
                             for stage_id in stages_id[1:]}
                      for area in areas},
        'best_template_id': {area: {stage_id: {substage: {} for substage in stage_substages_map[stage_id]}
                                    for stage_id in stages_id[1:]}
                             for area in areas},
        'time_resolved_best_template': {area: {stage_id: {substage: {} for substage in stage_substages_map[stage_id]}
                                               for stage_id in stages_id[1:]}
                                        for area in areas},
        'collected_best_template': {area: {stage_id: {substage: {} for substage in stage_substages_map[stage_id]}
                                           for stage_id in stages_id[1:]}
                                    for area in areas}},
    'strength': {
        'templates': {area: {stage_id: {substage: {} for substage in stage_substages_map[stage_id]}
                             for stage_id in stages_id[1:]}
                      for area in areas},
        'best_template': {area: {stage_id: {substage: {} for substage in stage_substages_map[stage_id]}
                                 for stage_id in stages_id[1:]}
                          for area in areas},
        'collected_best_template': {area: {stage_id: {substage: {} for substage in stage_substages_map[stage_id]}
                                           for stage_id in stages_id[1:]}
                                    for area in areas}}}

synchronization_data = {
    'population_firing_rate_trace': {population_key: {stage_id: {substage: {} for substage in stage_substages_map[stage_id]}
                                                      for stage_id in stages_id[1:]}
                                     for population_key in synchronization_population_keys},
    'transition_time': {time_key: {population_key: {stage_id: {substage: {} for substage in stage_substages_map[stage_id]}
                                                    for stage_id in stages_id[1:]}
                                   for population_key in synchronization_population_keys}
                        for time_key in ('tstart_ms', 'tstop_ms', 'activation_time_ms', 'deactivation_time_ms')}}

similarity_single_track_data = {
    'raster_cx': {stage_id: {substage: np.array([], dtype=object) for substage in stage_substages_map[stage_id]}
                  for stage_id in stages_id[1:]},
    'similarity': {area: {stage_id: {substage: np.empty((0, n_seq_train)) for substage in stage_substages_map[stage_id]}
                          for stage_id in stages_id[1:]}
                   for area in areas},
    'strength': {area: {stage_id: {substage: np.empty((0, n_seq_train)) for substage in stage_substages_map[stage_id]}
                        for stage_id in stages_id[1:]}
                 for area in areas}}


#----------------------------- PREPROCESS TRIALS
awake_stage_id = '00_awake_training'
awake_tstart = times_dict['awake_training'][awake_template_substage]['start']
reactivation_event_count_by_stage = {}
n_trials = len(trials_id)
n_reactivation_stages = len(analysis_stage_ids)
steps_per_trial = 1 + 2 * n_reactivation_stages
t_start_preproc = time.time()
process = psutil.Process(os.getpid())

for ntrial, trial_id in enumerate(trials_id):
    trial_start_time = time.time()
    trial_path = trial_path_dict[trial_id]
    fr_cache_trial_path = os.path.join(fr_cache_path, trial_id)
    os.makedirs(fr_cache_trial_path, exist_ok=True)

    # Awake training: load firing rates and build binary training templates.
    stage_id = awake_stage_id
    stage = stage_id[3:]
    substage = awake_template_substage
    spikes_leaf = {area: np.array(np.load(os.path.join(trial_path, stage_id, 'Events', f'{area}_{stage_id}.npy'),
                                          allow_pickle=True).item()['evt_exc'], dtype=object)
                   for area in areas}
    tstart = times_dict[stage][substage]['start']
    tstop = times_dict[stage][substage]['stop']
    firing_rate_leaf = {area: LoadOrComputeFR(
        spikes_leaf[area], area, stage_id, substage, tstart, tstop,
        fr_cache_trial_path, dt_s_fr, dt_ds_fr, nu_t_high_fr, nu_t_low_fr)
        for area in areas}
    awake_sequence_leaf = {area: firing_rate_leaf[area].T if firing_rate_leaf[area] is not None else None for area in areas}

    awake_template_leaf = {area: [] for area in areas}
    for seq in range(n_seq_train):
        template_tstart = awake_tstart + t_pause + seq * (t_img_train + t_pause)
        template_tstop = template_tstart + t_img_train
        istart = int(template_tstart // dt_react)
        istop = int(template_tstop // dt_react)
        for area in areas:
            awake_template_leaf[area].append((np.max(awake_sequence_leaf[area][istart:istop], axis=0) > 0).astype(float))

    awake_template_leaf = {area: np.asarray(awake_template_leaf[area], dtype=float) for area in areas}
    awake_template_norm = {area: np.linalg.norm(awake_template_leaf[area], axis=1) for area in areas}
    awake_template_norm = {area: np.where(awake_template_norm[area] > 0, awake_template_norm[area], 1.0) for area in areas}
    del spikes_leaf, firing_rate_leaf, awake_sequence_leaf
    gc.collect()

    reactivation_event_count_by_stage[trial_id] = {}
    for nstage, stage_id in enumerate(analysis_stage_ids):
        # NREM signal loading: spikes, firing rates and plot-only raster data.
        stage = stage_id[3:]
        substage = analysis_substage
        current_trial_step = 2 + 2 * nstage
        current_trial_fraction = current_trial_step / steps_per_trial
        trial_elapsed = time.time() - trial_start_time
        if ntrial > 0:
            completed_trial_mean_time = (trial_start_time - t_start_preproc) / ntrial
            t_left = max(completed_trial_mean_time - trial_elapsed, 0) + completed_trial_mean_time * (n_trials - ntrial - 1)
        else:
            current_trial_estimated_time = trial_elapsed / current_trial_fraction
            t_left = max(current_trial_estimated_time - trial_elapsed, 0) + current_trial_estimated_time * (n_trials - 1)
        t_elapsed = time.time() - t_start_preproc
        h_elapsed, remainder = divmod(t_elapsed, 3600)
        m_elapsed, s_elapsed = divmod(remainder, 60)
        h_left, remainder = divmod(t_left, 3600)
        m_left, s_left = divmod(remainder, 60)
        memory_mb = process.memory_info().rss / (1024 * 1024)
        print(f'Trial: {ntrial + 1}/{n_trials} - Stage: {stage_id} {nstage + 1}/{n_reactivation_stages}'
              f' - Memory: {memory_mb:.2f} MB - Elapsed: {int(h_elapsed)}h {int(m_elapsed)}m {int(s_elapsed)}s -'
              f' Time left: {int(h_left)}h {int(m_left)}m {int(s_left)}s',end='\r', flush=True)

        events_leaf = {area: np.load(os.path.join(trial_path, stage_id, 'Events', f'{area}_{stage_id}.npy'),
                                     allow_pickle=True).item()
                       for area in areas}
        spikes_leaf = {area: np.array(events_leaf[area]['evt_exc'], dtype=object) for area in areas}
        tstart = times_dict[stage][substage]['start']
        tstop = times_dict[stage][substage]['stop']
        firing_rate_leaf = {area: LoadOrComputeFR(
            spikes_leaf[area], area, stage_id, substage, tstart, tstop,
            fr_cache_trial_path, dt_s_fr, dt_ds_fr, nu_t_high_fr, nu_t_low_fr)
            for area in areas}
        population_firing_rate_leaf = {area: LoadOrComputePopulationFR(
            spikes_leaf[area], area, stage_id, substage, tstart, tstop,
            fr_cache_trial_path, dt_s_pop, dt_ds_pop, nu_t_high_pop, nu_t_low_pop)
            for area in areas}
        sleep_sequence_leaf = {area: firing_rate_leaf[area].T if firing_rate_leaf[area] is not None else None for area in areas}

        if trial_id == plot_trial_id:
            analysis_tstart = times_dict[stage][substage]['start']
            analysis_tstop = times_dict[stage][substage]['stop']
            raster_cx = np.array([
                (Mask(spikes_neur, analysis_tstart, analysis_tstop) - analysis_tstart) / 1000.0
                for spikes_neur in spikes_leaf['cx']], dtype=object)
            similarity_single_track_data['raster_cx'][stage_id][substage] = raster_cx

        # Reactivation observables: template similarity, event timing and event content.
        current_trial_step = 3 + 2 * nstage
        current_trial_fraction = current_trial_step / steps_per_trial
        trial_elapsed = time.time() - trial_start_time
        if ntrial > 0:
            completed_trial_mean_time = (trial_start_time - t_start_preproc) / ntrial
            t_left = max(completed_trial_mean_time - trial_elapsed, 0) + completed_trial_mean_time * (n_trials - ntrial - 1)
        else:
            current_trial_estimated_time = trial_elapsed / current_trial_fraction
            t_left = max(current_trial_estimated_time - trial_elapsed, 0) + current_trial_estimated_time * (n_trials - 1)
        t_elapsed = time.time() - t_start_preproc
        h_elapsed, remainder = divmod(t_elapsed, 3600)
        m_elapsed, s_elapsed = divmod(remainder, 60)
        h_left, remainder = divmod(t_left, 3600)
        m_left, s_left = divmod(remainder, 60)
        memory_mb = process.memory_info().rss / (1024 * 1024)
        substage_tstart = times_dict[stage][substage]['start']
        n_react_bins = sleep_sequence_leaf['cx'].shape[0]
        if trial_id == plot_trial_id:
            templates_similarity_leaf = {area: CosineSimilarity(sleep_sequence_leaf[area], awake_template_leaf[area]) for area in areas}
            reactivation_strength_leaf = {area: (sleep_sequence_leaf[area] @ awake_template_leaf[area].T) / awake_template_norm[area] for area in areas}
        else:
            templates_similarity_leaf = {}
            reactivation_strength_leaf = {}

        reactivation_leaf = {
            'events': {
                'index': [],
                'area': {area: {
                    'templates_similarity': np.zeros((0, n_seq_train)),
                    'time_resolved_similarity': {'trace': np.empty((0, 0), dtype=np.float16)},
                    'collected_best_similarity': np.array([], dtype=np.float32),
                    'strength': {
                        'templates': np.zeros((0, n_seq_train)),
                        'best_template': np.array([]),
                        'collected_best_template': np.array([], dtype=np.float32)}} for area in areas}}}

        n_population_bins = min([np.asarray(population_firing_rate_leaf[area]).size for area in areas])
        detected_up_states = DetectUpStates(
            population_firing_rate_leaf['cx'],
            substage_tstart,
            dt_up_detect,
            analysis_up_state_config,
            len(spikes_leaf['cx']))

        # Keep only Up-states that can be represented by complete event-aligned windows.
        detected_up_states['states'] = [
            up_state for up_state in detected_up_states['states']
            if (up_state['istart'] - synchronization_event_padding_bins >= 0)
            and (up_state['istop'] + synchronization_event_padding_bins <= n_population_bins)]

        MapStatesToReferenceBins(detected_up_states['states'], substage_tstart, dt_react, n_react_bins, prefix='react')
        up_state_tstart = np.asarray([up_state['tstart'] for up_state in detected_up_states['states']], dtype=float)
        up_state_tstop = np.asarray([up_state['tstop'] for up_state in detected_up_states['states']], dtype=float)
        up_state_tpeak = np.asarray([up_state['tpeak'] for up_state in detected_up_states['states']], dtype=float)

        # Up-state intervals are stored once and shared by cx/th observables.
        for up_state in detected_up_states['states']:
            istart = up_state['istart_react']
            istop = up_state['istop_react']
            event_id = len(reactivation_leaf['events']['index'])
            reactivation_leaf['events']['index'].append({
                'id': event_id,
                'up_state_id': up_state['id'],
                'istart': istart,
                'istop': istop,
                'ipeak': up_state['ipeak_react'],
                'tstart': substage_tstart + istart * dt_react,
                'tstop': substage_tstart + istop * dt_react,
                'tpeak': up_state['tpeak'],
                'duration': (istop - istart) * dt_react})

        up_state_duration = np.asarray([
            event_k['duration'] for event_k in reactivation_leaf['events']['index']], dtype=np.float32)
        down_state_duration = (up_state_tstart[1:] - up_state_tstop[:-1]).astype(np.float32)
        iwi = (up_state_tpeak[1:] - up_state_tpeak[:-1]).astype(np.float32)
        up_states_data['events']['up_state_id'][stage_id][substage][trial_id] = np.asarray([
            event_k['up_state_id'] for event_k in reactivation_leaf['events']['index']], dtype=np.int32)
        up_states_data['events']['tstart_ms'][stage_id][substage][trial_id] = np.asarray([
            event_k['tstart'] for event_k in reactivation_leaf['events']['index']], dtype=np.float32)
        up_states_data['events']['tstop_ms'][stage_id][substage][trial_id] = np.asarray([
            event_k['tstop'] for event_k in reactivation_leaf['events']['index']], dtype=np.float32)
        up_states_data['events']['tpeak_ms'][stage_id][substage][trial_id] = np.asarray([
            event_k['tpeak'] for event_k in reactivation_leaf['events']['index']], dtype=np.float32)
        up_states_data['events']['duration_ms'][stage_id][substage][trial_id] = up_state_duration
        up_states_data['down_state_duration_ms'][stage_id][substage][trial_id] = down_state_duration
        up_states_data['iwi_ms'][stage_id][substage][trial_id] = iwi

        n_events = len(reactivation_leaf['events']['index'])
        population_rate_leaf = {area: np.asarray(population_firing_rate_leaf[area], dtype=float).ravel() / len(spikes_leaf[area])
                                for area in areas}
        event_window_index = [
            (max(up_state['istart'] - synchronization_event_padding_bins, 0),
             min(up_state['istop'] + synchronization_event_padding_bins, n_population_bins))
            for up_state in detected_up_states['states']]
        event_window_slice = [slice(istart, istop, synchronization_downsample_bins)
                              for istart, istop in event_window_index]
        population_trace_bins = max([len(np.arange(event_slice.start, event_slice.stop, event_slice.step))
                                     for event_slice in event_window_slice]) if n_events > 0 else 0
        smooth_bins = max(int(round(analysis_up_state_config['smooth_ms'] / dt_up_detect)), 1)
        population_rate_smooth_leaf = {area: MovingAverage(population_rate_leaf[area], smooth_bins)
                                       for area in areas}
        for area in areas:
            population_rate = population_rate_smooth_leaf[area]
            up_state_population_rate = np.concatenate([
                population_rate[up_state['istart']:up_state['istop']]
                for up_state in detected_up_states['states']]) if n_events > 0 else np.array([])
            up_states_data['firing_rate'][area][stage_id][substage][trial_id] = up_state_population_rate.astype(np.float32)

        # Event-wise population traces and transition times around detected Up-states.
        if compute_synchronization:
            for population_key in synchronization_population_keys:
                area, neuron_type = population_key.split('_')
                if neuron_type == 'exc':
                    population_rate = population_rate_smooth_leaf[area]
                else:
                    spikes_inh = np.array(events_leaf[area]['evt_inh'], dtype=object)
                    population_rate = LoadOrComputePopulationFR(
                        spikes_inh, population_key, stage_id, substage, tstart, tstop,
                        fr_cache_trial_path, dt_s_pop, dt_ds_pop, nu_t_high_pop, nu_t_low_pop)
                    population_rate = np.asarray(population_rate, dtype=float).ravel() / len(spikes_inh)
                    population_rate = MovingAverage(population_rate, smooth_bins)

                population_trace = np.full((n_events, population_trace_bins), np.nan, dtype=np.float32)
                activation_time_ms = np.full(n_events, np.nan, dtype=np.float32)
                deactivation_time_ms = np.full(n_events, np.nan, dtype=np.float32)
                tstart_ms = np.full(n_events, np.nan, dtype=np.float32)
                tstop_ms = np.full(n_events, np.nan, dtype=np.float32)
                for nevent, (up_state, event_slice) in enumerate(zip(detected_up_states['states'], event_window_slice)):
                    event_trace = population_rate[event_slice]
                    population_trace[nevent, :event_trace.size] = event_trace
                    tstart_ms[nevent] = up_state['tstart']
                    tstop_ms[nevent] = up_state['tstop']
                    if population_key == 'cx_exc':
                        activation_time_ms[nevent] = up_state['tstart']
                        deactivation_time_ms[nevent] = up_state['tstop']
                    else:
                        activation_istart = up_state['istart']
                        activation_istop = min(up_state['istop'] + synchronization_event_padding_bins, population_rate.size)
                        activation_id = np.where(population_rate[activation_istart:activation_istop] > 0)[0]

                        if activation_id.size > 0:
                            iact = activation_istart + activation_id[0]
                            activation_time_ms[nevent] = substage_tstart + iact * dt_up_detect
                            deactivation_istart = iact
                            deactivation_istop = min(up_state['istop'] + synchronization_event_padding_bins, population_rate.size)
                            active_id = np.where(population_rate[deactivation_istart:deactivation_istop] > 0)[0]
                            if active_id.size > 0:
                                ideact = deactivation_istart + active_id[-1]
                                deactivation_time_ms[nevent] = substage_tstart + ideact * dt_up_detect

                synchronization_data['population_firing_rate_trace'][population_key][stage_id][substage][trial_id] = population_trace.astype(np.float16)
                synchronization_data['transition_time']['tstart_ms'][population_key][stage_id][substage][trial_id] = tstart_ms
                synchronization_data['transition_time']['tstop_ms'][population_key][stage_id][substage][trial_id] = tstop_ms
                synchronization_data['transition_time']['activation_time_ms'][population_key][stage_id][substage][trial_id] = activation_time_ms
                synchronization_data['transition_time']['deactivation_time_ms'][population_key][stage_id][substage][trial_id] = deactivation_time_ms

        if not compute_synchronization:
            for population_key in synchronization_population_keys:
                synchronization_data['population_firing_rate_trace'][population_key][stage_id][substage][trial_id] = np.empty((0, 0), dtype=np.float16)
                synchronization_data['transition_time']['tstart_ms'][population_key][stage_id][substage][trial_id] = np.array([], dtype=np.float32)
                synchronization_data['transition_time']['tstop_ms'][population_key][stage_id][substage][trial_id] = np.array([], dtype=np.float32)
                synchronization_data['transition_time']['activation_time_ms'][population_key][stage_id][substage][trial_id] = np.array([], dtype=np.float32)
                synchronization_data['transition_time']['deactivation_time_ms'][population_key][stage_id][substage][trial_id] = np.array([], dtype=np.float32)

        # Event-wise area observables are computed inside the whole detected Up-state.
        for area in areas:
            if n_events > 0:
                if trial_id == plot_trial_id:
                    templates_similarity_event = np.asarray([
                        np.median(templates_similarity_leaf[area][event_k['istart']:event_k['istop']], axis=0)
                        for event_k in reactivation_leaf['events']['index']])
                else:
                    event_similarity_leaf = []
                    event_strength_leaf = []
                    for event_k in reactivation_leaf['events']['index']:
                        event_sequence = sleep_sequence_leaf[area][event_k['istart']:event_k['istop']]
                        event_similarity_leaf.append(CosineSimilarity(event_sequence, awake_template_leaf[area]))
                        event_strength_leaf.append((event_sequence @ awake_template_leaf[area].T) / awake_template_norm[area])
                    templates_similarity_event = np.asarray([
                        np.median(event_similarity, axis=0)
                        for event_similarity in event_similarity_leaf])
                best_template_id = np.argmax(templates_similarity_event, axis=1)
                reactivation_leaf['events']['area'][area]['templates_similarity'] = templates_similarity_event

                max_event_bins = max([event_k['istop'] - event_k['istart'] for event_k in reactivation_leaf['events']['index']])
                time_resolved_trace = np.full((n_events, max_event_bins), np.nan, dtype=np.float16)
                collected_best_similarity = []
                collected_best_strength = []
                for event_k in reactivation_leaf['events']['index']:
                    if trial_id == plot_trial_id:
                        event_trace = templates_similarity_leaf[area][event_k['istart']:event_k['istop'], best_template_id[event_k['id']]]
                        event_strength = reactivation_strength_leaf[area][event_k['istart']:event_k['istop'], best_template_id[event_k['id']]]
                    else:
                        event_trace = event_similarity_leaf[event_k['id']][:, best_template_id[event_k['id']]]
                        event_strength = event_strength_leaf[event_k['id']][:, best_template_id[event_k['id']]]
                    if len(event_trace) > 0:
                        time_resolved_trace[event_k['id'], :len(event_trace)] = event_trace.astype(np.float16)
                        collected_best_similarity.append(event_trace.astype(np.float32))
                        collected_best_strength.append(event_strength.astype(np.float32))
                reactivation_leaf['events']['area'][area]['time_resolved_similarity']['trace'] = time_resolved_trace
                reactivation_leaf['events']['area'][area]['collected_best_similarity'] = np.concatenate(collected_best_similarity).astype(np.float32)
                reactivation_data['similarity']['templates'][area][stage_id][substage][trial_id] = templates_similarity_event
                reactivation_data['similarity']['best_template_id'][area][stage_id][substage][trial_id] = best_template_id.astype(np.int32)
                reactivation_data['similarity']['time_resolved_best_template'][area][stage_id][substage][trial_id] = time_resolved_trace
                reactivation_data['similarity']['collected_best_template'][area][stage_id][substage][trial_id] = reactivation_leaf[
                    'events']['area'][area]['collected_best_similarity']

                if trial_id == plot_trial_id:
                    strength_templates = np.asarray([
                        np.median(reactivation_strength_leaf[area][event_k['istart']:event_k['istop']], axis=0)
                        for event_k in reactivation_leaf['events']['index']])
                else:
                    strength_templates = np.asarray([
                        np.median(event_strength, axis=0)
                        for event_strength in event_strength_leaf])
                strength_best_template = strength_templates[np.arange(n_events), best_template_id]
                reactivation_leaf['events']['area'][area]['strength']['templates'] = strength_templates
                reactivation_leaf['events']['area'][area]['strength']['best_template'] = strength_best_template
                reactivation_leaf['events']['area'][area]['strength']['collected_best_template'] = np.concatenate(collected_best_strength).astype(np.float32)
                reactivation_data['strength']['templates'][area][stage_id][substage][trial_id] = strength_templates
                reactivation_data['strength']['best_template'][area][stage_id][substage][trial_id] = strength_best_template
                reactivation_data['strength']['collected_best_template'][area][stage_id][substage][trial_id] = reactivation_leaf[
                    'events']['area'][area]['strength']['collected_best_template']
                if trial_id != plot_trial_id:
                    del event_similarity_leaf, event_strength_leaf, event_sequence
                del templates_similarity_event, best_template_id, time_resolved_trace, collected_best_similarity
                del collected_best_strength, strength_templates, strength_best_template
            else:
                reactivation_data['similarity']['templates'][area][stage_id][substage][trial_id] = np.zeros((0, n_seq_train))
                reactivation_data['similarity']['best_template_id'][area][stage_id][substage][trial_id] = np.array([], dtype=np.int32)
                reactivation_data['similarity']['time_resolved_best_template'][area][stage_id][substage][trial_id] = np.empty((0, 0), dtype=np.float16)
                reactivation_data['similarity']['collected_best_template'][area][stage_id][substage][trial_id] = np.array([], dtype=np.float32)
                reactivation_data['strength']['templates'][area][stage_id][substage][trial_id] = np.zeros((0, n_seq_train))
                reactivation_data['strength']['best_template'][area][stage_id][substage][trial_id] = np.array([], dtype=np.float32)
                reactivation_data['strength']['collected_best_template'][area][stage_id][substage][trial_id] = np.array([], dtype=np.float32)

        # Plot-only data are saved for every sleep stage of the representative trial.
        if trial_id == plot_trial_id:
            for area in areas:
                similarity_single_track_data['similarity'][area][stage_id][substage] = templates_similarity_leaf[area]
                similarity_single_track_data['strength'][area][stage_id][substage] = reactivation_strength_leaf[area]

        reactivation_event_count_by_stage[trial_id][stage_id] = len(reactivation_leaf['events']['index'])
        del events_leaf, spikes_leaf, firing_rate_leaf, population_firing_rate_leaf, sleep_sequence_leaf
        del population_rate_leaf, population_rate_smooth_leaf
        del templates_similarity_leaf, reactivation_strength_leaf
        del detected_up_states, reactivation_leaf

    gc.collect()

print()

#----------------------------- TRIAL-LIST DATA STRUCTURE
up_states_trials_data = {
    'events': {event_key: {stage_id: {substage: [
        up_states_data['events'][event_key][stage_id][substage][trial_id]
        for trial_id in trials_id_conf]
        for substage in stage_substages_map[stage_id]}
        for stage_id in stages_id[1:]}
        for event_key in up_states_data['events'].keys()},
    'down_state_duration_ms': {stage_id: {substage: [
        up_states_data['down_state_duration_ms'][stage_id][substage][trial_id]
        for trial_id in trials_id_conf]
        for substage in stage_substages_map[stage_id]}
        for stage_id in stages_id[1:]},
    'iwi_ms': {stage_id: {substage: [
        up_states_data['iwi_ms'][stage_id][substage][trial_id]
        for trial_id in trials_id_conf]
        for substage in stage_substages_map[stage_id]}
        for stage_id in stages_id[1:]},
    'firing_rate': {area: {stage_id: {substage: [
        up_states_data['firing_rate'][area][stage_id][substage][trial_id]
        for trial_id in trials_id_conf]
        for substage in stage_substages_map[stage_id]}
        for stage_id in stages_id[1:]}
        for area in areas}}

reactivation_trials_data = {
    'similarity': {
        'templates': {area: {stage_id: {substage: [
            reactivation_data['similarity']['templates'][area][stage_id][substage][trial_id]
            for trial_id in trials_id_conf]
            for substage in stage_substages_map[stage_id]}
            for stage_id in stages_id[1:]}
            for area in areas},
        'best_template_id': {area: {stage_id: {substage: [
            reactivation_data['similarity']['best_template_id'][area][stage_id][substage][trial_id]
            for trial_id in trials_id_conf]
            for substage in stage_substages_map[stage_id]}
            for stage_id in stages_id[1:]}
            for area in areas},
        'time_resolved_best_template': {area: {stage_id: {substage: [
            reactivation_data['similarity']['time_resolved_best_template'][area][stage_id][substage][trial_id]
            for trial_id in trials_id_conf]
            for substage in stage_substages_map[stage_id]}
            for stage_id in stages_id[1:]}
            for area in areas},
        'collected_best_template': {area: {stage_id: {substage: [
            reactivation_data['similarity']['collected_best_template'][area][stage_id][substage][trial_id]
            for trial_id in trials_id_conf]
            for substage in stage_substages_map[stage_id]}
            for stage_id in stages_id[1:]}
            for area in areas}},
    'strength': {
        'templates': {area: {stage_id: {substage: [
            reactivation_data['strength']['templates'][area][stage_id][substage][trial_id]
            for trial_id in trials_id_conf]
            for substage in stage_substages_map[stage_id]}
            for stage_id in stages_id[1:]}
            for area in areas},
        'best_template': {area: {stage_id: {substage: [
            reactivation_data['strength']['best_template'][area][stage_id][substage][trial_id]
            for trial_id in trials_id_conf]
            for substage in stage_substages_map[stage_id]}
            for stage_id in stages_id[1:]}
            for area in areas},
        'collected_best_template': {area: {stage_id: {substage: [
            reactivation_data['strength']['collected_best_template'][area][stage_id][substage][trial_id]
            for trial_id in trials_id_conf]
            for substage in stage_substages_map[stage_id]}
            for stage_id in stages_id[1:]}
            for area in areas}}}

synchronization_trials_data = {
    'population_firing_rate_trace': {population_key: {stage_id: {substage: [
        synchronization_data['population_firing_rate_trace'][population_key][stage_id][substage][trial_id]
        for trial_id in trials_id_conf]
        for substage in stage_substages_map[stage_id]}
        for stage_id in stages_id[1:]}
        for population_key in synchronization_population_keys},
    'transition_time': {time_key: {population_key: {stage_id: {substage: [
        synchronization_data['transition_time'][time_key][population_key][stage_id][substage][trial_id]
        for trial_id in trials_id_conf]
        for substage in stage_substages_map[stage_id]}
        for stage_id in stages_id[1:]}
        for population_key in synchronization_population_keys}
        for time_key in synchronization_data['transition_time'].keys()}}

similarity_single_track_trials_data = similarity_single_track_data

#----------------------------- SAVE DATA
np.save(up_states_path, {
    'data': up_states_trials_data,
    'n_trials': n_trials_conf,
    'trials_list': trials_id_conf,
    'trials_failed': trials_failed_conf,
    'rngs': rngs_conf,
    'dt': up_states_dt,
    'params': up_states_params}, allow_pickle=True)
np.save(reactivation_path, {
    'data': reactivation_trials_data,
    'n_trials': n_trials_conf,
    'trials_list': trials_id_conf,
    'trials_failed': trials_failed_conf,
    'rngs': rngs_conf,
    'dt': reactivation_dt,
    'params': reactivation_params}, allow_pickle=True)
np.save(synchronization_path, {
    'data': synchronization_trials_data,
    'n_trials': n_trials_conf,
    'trials_list': trials_id_conf,
    'trials_failed': trials_failed_conf,
    'rngs': rngs_conf,
    'dt': synchronization_dt,
    'params': synchronization_params}, allow_pickle=True)
np.save(similarity_single_track_path, {
    'data': similarity_single_track_trials_data,
    'n_trials': 1,
    'trials_list': [plot_trial_id],
    'trials_failed': trials_failed_conf,
    'rngs': {'nest_seed': np.array([None], dtype=object), 'numpy_seed': np.array([None], dtype=object)},
    'dt': similarity_single_track_dt,
    'params': similarity_single_track_params}, allow_pickle=True)

t_total = time.time() - t_start_preproc
t_total_h = int(t_total // 3600)
t_total_m = int((t_total - t_total_h * 3600) // 60)
t_total_s = t_total - t_total_h * 3600 - t_total_m * 60

print(f'Config: {config_path}')
print(f'Model: {model_type}')
print(f'Simulation data: {trial_simdata_path}')
print(f'Trials: {trials_id}')
print(f'Reactivation event count by stage: {reactivation_event_count_by_stage}')
print(f'Up-states output: {up_states_path}')
print(f'Reactivation output: {reactivation_path}')
print(f'Synchronization output: {synchronization_path}')
print(f'Similarity single-track output: {similarity_single_track_path}')
print(f'Total execution time: {t_total_h}h {t_total_m}m {t_total_s:.2f}s')
