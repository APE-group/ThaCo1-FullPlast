import os, sys
import csv
import numpy as np
import yaml


#----------------------------- CONFIG
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(script_dir, '..', 'Preprocessing', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

model_config = config['model']
paths_config = config['paths']

configs = [experiment['config'] for experiment in model_config['experiments']]
root_preprocdata_path = paths_config['root_save_path']
root_preprocdata_path = root_preprocdata_path if os.path.isabs(root_preprocdata_path) else os.path.abspath(os.path.join(os.path.dirname(config_path), root_preprocdata_path))
table_csv_path = os.path.join(script_dir, 'table1.csv')


#----------------------------- TABLE SETTINGS
columns = ('pre-sleep awake', 'whole sleep', 'early sleep', 'mid sleep', 'late sleep', 'post-sleep awake')
empty_cells = {column: '--' for column in columns}
pooled_measure_type = 'median and quartiles across pooled data'
trial_measure_type = 'mean and sem across trial averages'

awake_stage_pre = '00_awake_training'
awake_stage_post = '20_nrem'
classification_substage = 'classification'
sleep_stage_groups_config = {
    'early sleep': ('01_nrem', '02_nrem'),
    'mid sleep': ('04_nrem', '05_nrem', '06_nrem', '07_nrem'),
    'late sleep': ('12_nrem', '13_nrem', '14_nrem', '15_nrem', '16_nrem', '17_nrem', '18_nrem', '19_nrem', '20_nrem')}


#----------------------------- LOAD AND MEASURE
csv_rows = []

for conf in configs:
    model_preprocdata_path = os.path.join(root_preprocdata_path, conf)
    spikes_count_dict = np.load(os.path.join(model_preprocdata_path, 'spikes_count.npy'), allow_pickle=True).item()
    up_states_dict = np.load(os.path.join(model_preprocdata_path, 'up_states.npy'), allow_pickle=True).item()

    spikes_count = spikes_count_dict['data']
    up_states = up_states_dict['data']
    n_trials = up_states_dict['n_trials']
    analysis_substage = up_states_dict['params']['analysis_substage']
    performance_stage_ids = tuple(up_states_dict['params']['performance_stage_ids'])

    sleep_stage_groups = {'whole sleep': performance_stage_ids} | sleep_stage_groups_config
    table_rows = []

    # Single-neuron firing rates are already stored per neuron and trial, then pooled before statistics.
    for area, area_label in [('cx', 'cx'), ('th', 'th')]:
        observable = f'{area_label} single-neuron firing rate [Hz]'
        row = {'observable': observable, 'measure type': pooled_measure_type} | empty_cells.copy()

        values = np.asarray(spikes_count[area]['exc'][awake_stage_pre][classification_substage], dtype=float).ravel()
        row['pre-sleep awake'] = f'{np.median(values):.3f} [{np.percentile(values, 25):.3f}, {np.percentile(values, 75):.3f}]'

        values = np.asarray(spikes_count[area]['exc'][awake_stage_post][classification_substage], dtype=float).ravel()
        row['post-sleep awake'] = f'{np.median(values):.3f} [{np.percentile(values, 25):.3f}, {np.percentile(values, 75):.3f}]'

        for stage_key, stage_ids in sleep_stage_groups.items():
            values = np.concatenate([np.asarray(spikes_count[area]['exc'][stage_id][analysis_substage], dtype=float).ravel()
                                     for stage_id in stage_ids])
            row[stage_key] = f'{np.median(values):.3f} [{np.percentile(values, 25):.3f}, {np.percentile(values, 75):.3f}]'

        table_rows.append(row)

    # Trial-mean firing rates average neurons within each trial, then report mean and SEM across trials.
    for area, area_label in [('cx', 'cx'), ('th', 'th')]:
        observable = f'{area_label} single-neuron firing rate trial-mean [Hz]'
        row = {'observable': observable, 'measure type': trial_measure_type} | empty_cells.copy()

        values = np.asarray(spikes_count[area]['exc'][awake_stage_pre][classification_substage], dtype=float)
        values = np.mean(values.reshape(n_trials, -1), axis=1)
        row['pre-sleep awake'] = f'{np.mean(values):.3f} +/- {np.std(values, ddof=1) / np.sqrt(values.size):.3f}'

        values = np.asarray(spikes_count[area]['exc'][awake_stage_post][classification_substage], dtype=float)
        values = np.mean(values.reshape(n_trials, -1), axis=1)
        row['post-sleep awake'] = f'{np.mean(values):.3f} +/- {np.std(values, ddof=1) / np.sqrt(values.size):.3f}'

        table_rows.append(row)

    # SO rate is estimated event-by-event as the inverse of the inter-wave interval.
    observable = 'SO rate [Hz]'
    row = {'observable': observable, 'measure type': pooled_measure_type} | empty_cells.copy()
    for stage_key, stage_ids in sleep_stage_groups.items():
        values = []
        for stage_id in stage_ids:
            for trial_values in up_states['iwi_ms'][stage_id][analysis_substage]:
                trial_values = np.asarray(trial_values, dtype=float)
                trial_values = trial_values[(trial_values > 0)]
                values.append(1000 / trial_values)
        values = np.concatenate(values)
        row[stage_key] = f'{np.median(values):.3f} [{np.percentile(values, 25):.3f}, {np.percentile(values, 75):.3f}]'
    table_rows.append(row)

    # Up-state durations are pooled across detected events in the selected sleep stages.
    observable = 'up-state duration [ms]'
    row = {'observable': observable, 'measure type': pooled_measure_type} | empty_cells.copy()
    for stage_key, stage_ids in sleep_stage_groups.items():
        values = []
        for stage_id in stage_ids:
            for trial_values in up_states['events']['duration_ms'][stage_id][analysis_substage]:
                trial_values = np.asarray(trial_values, dtype=float)
                values.append(trial_values)
        values = np.concatenate(values)
        row[stage_key] = f'{np.median(values):.3f} [{np.percentile(values, 25):.3f}, {np.percentile(values, 75):.3f}]'
    table_rows.append(row)

    # Down-state durations are pooled in ms, consistently with up-state durations.
    observable = 'down-state duration [ms]'
    row = {'observable': observable, 'measure type': pooled_measure_type} | empty_cells.copy()
    for stage_key, stage_ids in sleep_stage_groups.items():
        values = []
        for stage_id in stage_ids:
            for trial_values in up_states['down_state_duration_ms'][stage_id][analysis_substage]:
                trial_values = np.asarray(trial_values, dtype=float)
                values.append(trial_values)
        values = np.concatenate(values)
        row[stage_key] = f'{np.median(values)} [{np.percentile(values, 25)}, {np.percentile(values, 75)}]'
    table_rows.append(row)

    # Up states are counted per trial after summing the selected sleep stages.
    observable = 'n up-states [trial]'
    row = {'observable': observable, 'measure type': pooled_measure_type} | empty_cells.copy()
    for stage_key, stage_ids in sleep_stage_groups.items():
        values = []
        for ntrial in range(n_trials):
            values.append(np.sum([len(up_states['events']['up_state_id'][stage_id][analysis_substage][ntrial])
                                  for stage_id in stage_ids]))
        values = np.asarray(values, dtype=float)
        row[stage_key] = f'{np.median(values):.3f} [{np.percentile(values, 25):.3f}, {np.percentile(values, 75):.3f}]'
    table_rows.append(row)

    print(f'\n{conf}')
    print(f'n_trials = {n_trials}')
    print('values = median [q1, q3], except trial-mean firing rates = mean +/- sem across trials')
    print('| observable | measure type | ' + ' | '.join(columns) + ' |')
    print('|---|---|' + '|'.join(['---' for _ in columns]) + '|')
    for row in table_rows:
        print('| ' + row['observable'] + ' | ' + row['measure type'] + ' | ' + ' | '.join([row[column] for column in columns]) + ' |')
        csv_rows.append({'config': conf, 'n_trials': n_trials} | row)

with open(table_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['config', 'n_trials', 'observable', 'measure type'] + list(columns))
    writer.writeheader()
    writer.writerows(csv_rows)

print(f'\nCSV: {table_csv_path}')
