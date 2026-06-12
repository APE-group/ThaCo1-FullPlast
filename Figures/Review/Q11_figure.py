import os, sys
import csv

#os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '.mplconfig'))

import matplotlib.colors as mcolors
import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import binomtest as Btest
from scipy.stats import mannwhitneyu as MWU


#----------------------------- CONFIG
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(script_dir, 'config_figure.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

figure_style_config = config['figure_style']
model_config = config['model']
paths_config = config['paths']
network_config = config['network']
timing_config = config['timing']
stages_config = config['stages']
reactivation_config = config['reactivation']

fig_width = figure_style_config['fig_width'] / 25.4
fig_height = figure_style_config['fig_height'] / 25.4
dpi = figure_style_config['dpi']
linewidth = figure_style_config['linewidth']
fontsize_label = figure_style_config['fontsize_labels']
fontsize_panel_letter = figure_style_config['fontsize_panel_letters']
fontsize_title = figure_style_config['fontsize_title']
fontsize_ticks = figure_style_config['fontsize_ticks']
labelpad = figure_style_config['labelpad']



#----------------------------- MODEL CONFIG
input_type = model_config['input']
plasticity_type = model_config['plasticity']
model_type = f'startreview_{input_type}_{plasticity_type}plast'

n_ranks_train = network_config['n_ranks_train']

t_nrem_therm = timing_config['t_nrem_therm']
t_nrem = timing_config['t_nrem']
t_start_nrem_therm = timing_config['t_start_nrem_therm']

t_stop_nrem_therm = t_start_nrem_therm + t_nrem_therm
t_start_nrem = t_stop_nrem_therm
t_stop_nrem = t_start_nrem + t_nrem


#----------------------------- PATHS
root_save_path = paths_config['root_save_path']
root_preprocdata_path = f"{paths_config['root_preprocdata_path']}"
plot_save_path = os.path.join(root_save_path, 'Plots')

os.makedirs(plot_save_path, exist_ok=True)

figure_filename = f"{paths_config['figure_filename']}_{model_type}.png"
figure_stem, figure_ext = os.path.splitext(figure_filename)
figure_ext = figure_ext if figure_ext else '.png'
fig1_path = os.path.join(plot_save_path, f'{figure_stem}{figure_ext}')
performance_stats_path = os.path.join(plot_save_path, f'{figure_stem}_performance_binomial_summary.csv')
model_preprocdata_path = os.path.join(root_preprocdata_path, model_type)
up_states_path = os.path.join(model_preprocdata_path, 'up_states.npy')
reactivation_path = os.path.join(model_preprocdata_path, 'reactivation.npy')
similarity_single_track_path = os.path.join(model_preprocdata_path, 'similarity_single_track.npy')


#----------------------------- LOAD DATA
up_states_dict = np.load(up_states_path, allow_pickle=True).item()
reactivation_dict = np.load(reactivation_path, allow_pickle=True).item()
similarity_single_track_dict = np.load(similarity_single_track_path, allow_pickle=True).item()

up_states = up_states_dict['data']
reactivation = reactivation_dict['data']
similarity_single_track = similarity_single_track_dict['data']

trials_list = tuple(reactivation_dict['trials_list'])
n_trials = len(trials_list)
representative_trial_id = similarity_single_track_dict['trials_list'][0]
representative_trial_position = trials_list.index(representative_trial_id)

plot_areas = tuple(reactivation_dict['params']['areas'])
performance_stage_ids = tuple(up_states_dict['params']['performance_stage_ids'])
analysis_substage = up_states_dict['params']['analysis_substage']
template_class_id = np.asarray(reactivation_dict['params']['template_class_id'])
dt_react = reactivation_dict['dt']['reactivation']
single_track_observable = reactivation_config['single_track_observable']

#----------------------------- CONFIG-DEFINED SLEEP GROUPS
sleep_stage_keys = tuple(stages_config['plot_stage_id'].keys())
plot_stage_ids = {stage_key: stages_config['plot_stage_id'][stage_key] for stage_key in sleep_stage_keys}
plot_stage_key = stages_config['plot_stage_key']


#----------------------------- FIGURE SETTINGS
t_sleep_interval_raster = timing_config['t_sleep_interval']['rastergram']
t_sleep_interval_single_track_start = timing_config['t_sleep_interval']['single_track']['start']
t_sleep_interval_single_track_stop = timing_config['t_sleep_interval']['single_track']['stop']
n_templates = len(template_class_id)
activation_strength_quantile = reactivation_config['activation_strength_quantile']

#----------------------------- FIGURE 1 REPRESENTATIVE OBSERVABLES
single_track_start_bin = int(np.floor(t_sleep_interval_single_track_start / dt_react))
single_track_stop_bin = int(np.ceil(t_sleep_interval_single_track_stop / dt_react))
single_track_full = {area: similarity_single_track[single_track_observable][area][plot_stage_ids[plot_stage_key]][analysis_substage]
                     for area in plot_areas}
single_track_plot = {area: single_track_full[area][single_track_start_bin:single_track_stop_bin] for area in plot_areas}
if single_track_observable == 'similarity':
    single_track_norm = {area:mcolors.Normalize(0, 1)  for area in plot_areas}
    single_track_cbar_title = r'$S$'
    single_track_row_title = 'time-resolved reactivation similarity'
else:
    single_track_vmax = {area:np.max(single_track_plot[area]) for area in plot_areas}
    single_track_norm = {area:mcolors.Normalize(0, single_track_vmax[area]) for area in plot_areas}
    single_track_cbar_title = r'$R$'
    single_track_row_title = 'time-resolved reactivation strength'
single_track_x = np.arange(len(single_track_plot['cx']), dtype=int)
single_track_y = np.arange(single_track_plot['cx'].shape[1], dtype=int)
single_track_xticks = single_track_x[::100]
single_track_xlabels = (single_track_start_bin + single_track_xticks) * dt_react / 1000.0
single_track_yticks = single_track_y[::3]
single_track_ylabels = single_track_yticks // 3

if analysis_substage != 'sleep':
    raise ValueError('Q11 figure 1 expects sleep analysis_substage.')
analysis_tstart = t_start_nrem
analysis_tstop = t_stop_nrem
analysis_tstop_raster = min(analysis_tstart + t_sleep_interval_raster, analysis_tstop)
raster_duration_ms = max(analysis_tstop_raster - analysis_tstart, 0.0)
raster_time_s = np.arange(0.0, raster_duration_ms / 1000.0, dt_react / 1000.0, dtype=float)
raster_stop_s = raster_duration_ms / 1000.0

raster_cx = {stage_key: np.array([
    spikes_neur[(spikes_neur >= 0.0) & (spikes_neur <= raster_stop_s)]
    for spikes_neur in similarity_single_track['raster_cx'][plot_stage_ids[stage_key]][analysis_substage]], dtype=object)
    for stage_key in sleep_stage_keys}

event_spans_s = {stage_key: [] for stage_key in sleep_stage_keys}
for stage_key in sleep_stage_keys:
    event_tstart = up_states['events']['tstart_ms'][plot_stage_ids[stage_key]][analysis_substage][representative_trial_position]
    event_tstop = up_states['events']['tstop_ms'][plot_stage_ids[stage_key]][analysis_substage][representative_trial_position]
    for tstart, tstop in zip(event_tstart, event_tstop):
        event_start = max(tstart, analysis_tstart)
        event_stop = min(tstop, analysis_tstop_raster)
        if event_stop > event_start:
            event_spans_s[stage_key].append(((event_start - analysis_tstart) / 1000.0,
                                             (event_stop - analysis_tstart) / 1000.0))

#----------------------------- FIGURE 1 ALL-SLEEP PERFORMANCE OBSERVABLES
performance_templates_similarity = {area:
    np.concatenate([reactivation['similarity']['templates'][area][stage_id][analysis_substage][ntrial]
                    for ntrial in range(n_trials) for stage_id in performance_stage_ids], axis=0)
    for area in plot_areas}
performance_templates_strength = {area:
    np.concatenate([reactivation['strength']['templates'][area][stage_id][analysis_substage][ntrial]
                    for ntrial in range(n_trials) for stage_id in performance_stage_ids], axis=0)
    for area in plot_areas}

#----------------------------- FIGURE 1 CORTICAL ACTIVATION MULTIPLICITY
activation_area = 'cx'
activation_order = ('singlet', 'doublet_same', 'doublet_mixed', 'triplet_same', 'triplet_mixed', 'multiplet')
activation_colors = {
    'singlet': 'royalblue',
    'doublet_same': 'lightskyblue',
    'doublet_mixed': 'indianred',
    'triplet_same': 'lightskyblue',
    'triplet_mixed': 'indianred',
    'multiplet': '0.5'}
activation_strength_pool = performance_templates_strength[activation_area][performance_templates_strength[activation_area] > 0]
activation_strength_threshold = np.quantile(activation_strength_pool, activation_strength_quantile)
activation_count = {key: 0 for key in activation_order}
activation_same_class = {'doublet': [], 'triplet': []}

for strength_k in performance_templates_strength[activation_area]:
    active_template_id = np.flatnonzero(strength_k >= activation_strength_threshold)
    n_active_templates = len(active_template_id)
    if n_active_templates == 1:
        activation_count['singlet'] += 1
    elif n_active_templates == 2:
        active_class_id = template_class_id[active_template_id]
        same_class_event = len(np.unique(active_class_id)) == 1
        activation_count['doublet_same' if same_class_event else 'doublet_mixed'] += 1
        activation_same_class['doublet'].append(same_class_event)
    elif n_active_templates == 3:
        active_class_id = template_class_id[active_template_id]
        same_class_event = len(np.unique(active_class_id)) == 1
        activation_count['triplet_same' if same_class_event else 'triplet_mixed'] += 1
        activation_same_class['triplet'].append(same_class_event)
    elif n_active_templates > 3:
        activation_count['multiplet'] += 1

activation_n_events = np.sum([activation_count[key] for key in activation_order])
activation_fraction = {key: activation_count[key] / activation_n_events for key in activation_order}

#----------------------------- FIGURE 1 REACTIVATION PERFORMANCE
performance_similarity_order = ('singlet', 'doublet_same', 'doublet_mixed',
                                'triplet_same', 'triplet_mixed', 'multiplet')
performance_similarity_style = {
    'singlet': {'color': activation_colors['singlet']},
    'doublet_same': {'color': activation_colors['doublet_same']},
    'doublet_mixed': {'color': activation_colors['doublet_mixed']},
    'triplet_same': {'color': activation_colors['triplet_same']},
    'triplet_mixed': {'color': activation_colors['triplet_mixed']},
    'multiplet': {'color': activation_colors['multiplet']}}
performance_similarity_pooled = {area: {key: [] for key in performance_similarity_order} for area in plot_areas}

for nevent, strength_k in enumerate(performance_templates_strength[activation_area]):
    active_template_id = np.flatnonzero(strength_k >= activation_strength_threshold)
    n_active_templates = len(active_template_id)
    if n_active_templates == 1:
        for area in plot_areas:
            active_similarity = performance_templates_similarity[area][nevent, active_template_id]
            performance_similarity_pooled[area]['singlet'].append(active_similarity[0])
    elif n_active_templates >= 2:
        active_template_class_id = template_class_id[active_template_id]
        same_class_event = len(np.unique(active_template_class_id)) == 1
        if n_active_templates == 2:
            performance_key = 'doublet_same' if same_class_event else 'doublet_mixed'
        elif n_active_templates == 3:
            performance_key = 'triplet_same' if same_class_event else 'triplet_mixed'
        else:
            performance_key = 'multiplet'
        for area in plot_areas:
            active_similarity = performance_templates_similarity[area][nevent, active_template_id]
            performance_similarity_pooled[area][performance_key].extend(active_similarity)

performance_similarity_pooled = {area: {key:
    np.asarray(performance_similarity_pooled[area][key], dtype=float)
    for key in performance_similarity_order} for area in plot_areas}

#----------------------------- FIGURE 1 PERFORMANCE STATISTICS
doublet_same_class_p0 = (n_ranks_train - 1) / (n_templates - 1)
triplet_same_class_p0 = ((n_ranks_train - 1) * (n_ranks_train - 2)) / ((n_templates - 1) * (n_templates - 2))
activation_stat_pvalues = {'doublet': np.nan, 'triplet': np.nan}
similarity_stat_pvalues = {area: {'doublet': np.nan, 'triplet': np.nan} for area in plot_areas}
performance_stat_rows = []
for event_key, chance_fraction in [('doublet', doublet_same_class_p0), ('triplet', triplet_same_class_p0)]:
    same_class = np.asarray(activation_same_class[event_key], dtype=bool)
    n_events = len(same_class)
    n_same_class = int(np.sum(same_class))
    stat_test = Btest(n_same_class, n_events, p=chance_fraction, alternative='greater')
    statistic = stat_test.statistic
    pvalue = stat_test.pvalue
    observed_fraction = n_same_class / n_events
    activation_stat_pvalues[event_key] = pvalue
    performance_stat_rows.append({
        'area': activation_area,
        'test_type': 'one_sided_binomial',
        'event': f'{event_key}_same_class_activation',
        'n_events': n_events,
        'n_same_class': n_same_class,
        'observed_fraction': observed_fraction,
        'chance_fraction': chance_fraction,
        'statistic': statistic,
        'pvalue': pvalue})

for area in plot_areas:
    for event_key, same_key, mixed_key in [
            ('doublet', 'doublet_same', 'doublet_mixed'),
            ('triplet', 'triplet_same', 'triplet_mixed')]:
        same_values = performance_similarity_pooled[area][same_key]
        mixed_values = performance_similarity_pooled[area][mixed_key]
        stat_test = MWU(same_values, mixed_values, alternative='greater')
        statistic = stat_test.statistic
        pvalue = stat_test.pvalue
        similarity_stat_pvalues[area][event_key] = pvalue
        performance_stat_rows.append({
            'area': area,
            'test_type': 'one_sided_mann_whitney_u',
            'event': f'{event_key}_same_vs_mixed_similarity',
            'n_events': len(same_values) + len(mixed_values),
            'n_same_class': len(same_values),
            'observed_fraction': np.nan,
            'chance_fraction': np.nan,
            'statistic': statistic,
            'pvalue': pvalue})

with open(performance_stats_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(
        f,
        fieldnames=['area', 'test_type', 'event', 'n_events', 'n_same_class',
                    'observed_fraction', 'chance_fraction', 'statistic', 'pvalue'])
    writer.writeheader()
    writer.writerows(performance_stat_rows)

activation_position = {
    'singlet': 1.0,
    'doublet_same': 2.15,
    'doublet_mixed': 2.50,
    'triplet_same': 3.65,
    'triplet_mixed': 4.00,
    'multiplet': 5.15}
activation_positions = [activation_position[key] for key in activation_order]
activation_group_positions = [activation_position['singlet'],
                              0.5 * (activation_position['doublet_same'] + activation_position['doublet_mixed']),
                              0.5 * (activation_position['triplet_same'] + activation_position['triplet_mixed']),
                              activation_position['multiplet']]
activation_group_labels = ['singlet', 'doublet', 'triplet', 'multiplet']
activation_width = 0.28
performance_similarity_position = {
    'singlet': 1.0,
    'doublet_same': 1.85,
    'doublet_mixed': 2.15,
    'triplet_same': 2.85,
    'triplet_mixed': 3.15,
    'multiplet': 4.00}
performance_similarity_positions = [performance_similarity_position[key] for key in performance_similarity_order]
performance_similarity_group_positions = [performance_similarity_position['singlet'],
                                          0.5 * (performance_similarity_position['doublet_same'] +
                                                 performance_similarity_position['doublet_mixed']),
                                          0.5 * (performance_similarity_position['triplet_same'] +
                                                 performance_similarity_position['triplet_mixed']),
                                          performance_similarity_position['multiplet']]
performance_similarity_group_labels = ['singlet', 'doublet', 'triplet', 'multiplet']
similarity_major_ticks = [0, 0.5, 1.0]
similarity_minor_ticks = [0.25, 0.75]


#------------------------------------------------ PLOT ----------------------------------------------------------------#
try:
    from nmmn.plots import parulacmap as parula
    cmap = parula()
except ImportError:
    cmap = plt.get_cmap('viridis')


#FIG 1 - Q11 TIME-RESOLVED REACTIVATION
fig1 = plt.figure(figsize=(fig_width, fig_height))
gs = fig1.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.5)
gs_raster = gs[0, 0].subgridspec(1, 3, wspace=0.08)
gs_heatmap = gs[1, 0].subgridspec(1, 2, wspace=0.35)
gs_performance = gs[2, 0].subgridspec(1, 3, width_ratios=[1.05, 1, 1], wspace=0.35)

ax_raster = [fig1.add_subplot(gs_raster[0, nstage]) for nstage in range(len(sleep_stage_keys))]
ax_heatmap = [fig1.add_subplot(gs_heatmap[0, narea]) for narea in range(len(plot_areas))]
ax_multiplicity = fig1.add_subplot(gs_performance[0, 0])
ax_performance = [fig1.add_subplot(gs_performance[0, narea + 1]) for narea in range(len(plot_areas))]
for ax in ax_raster + ax_heatmap + [ax_multiplicity] + ax_performance:
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)

# Panel A - rastergrams.
for nstage, stage_key in enumerate(sleep_stage_keys):
    ax = ax_raster[nstage]
    for t0, t1 in event_spans_s[stage_key]:
        ax.axvspan(t0, t1, color='grey', alpha=0.4, zorder=0, )
    ax.eventplot(raster_cx[stage_key], colors='black', linewidths=0.5, zorder=2)
    ax.set_title(stage_key, fontsize=fontsize_title)
    ax.set_xlim(0.0, raster_time_s[-1])
    if stage_key == plot_stage_key:
        y0, y1 = ax.get_ylim()
        single_track_start_s = min(t_sleep_interval_single_track_start / 1000.0, raster_time_s[-1])
        single_track_stop_s = min(t_sleep_interval_single_track_stop / 1000.0, raster_time_s[-1])
        ax.add_patch(Rectangle(
            (single_track_start_s, min(y0, y1)),
            max(single_track_stop_s - single_track_start_s, 0.0),
            abs(y1 - y0),
            facecolor='lightcoral',
            edgecolor='lightcoral',
            linewidth=0.5 * linewidth,
            alpha=0.2,
            zorder=4))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.set_yticks([])
    if nstage == 0:
        ax.spines['left'].set_visible(True)
        ax.set_ylabel('cx neuron id', fontsize=fontsize_label, labelpad=labelpad)
    else:
        ax.tick_params(labelleft=False)
    if nstage == 1:
        ax.set_xlabel('t (s)', fontsize=fontsize_label, labelpad=labelpad)
    if nstage < len(sleep_stage_keys) - 1:
        ax.text(1.01, 0, '//', transform=ax.transAxes, fontsize=0.7 * fontsize_panel_letter, va='center', ha='left')

# Panel B - single-track heatmaps.
for narea, area in enumerate(plot_areas):
    ax = ax_heatmap[narea]
    im = ax.imshow(np.transpose(single_track_plot[area]), norm=single_track_norm[area], cmap=cmap, aspect='auto', origin='lower')
    ax.set_title(area, fontsize=fontsize_title)
    ax.set_ylabel('template class', fontsize=fontsize_label, labelpad=labelpad)
    ax.set_xlabel('t (s)', fontsize=fontsize_label, labelpad=labelpad)
    ax.set_xticks(single_track_xticks)
    ax.set_xticklabels([f'{x:.0f}' for x in single_track_xlabels])
    ax.set_yticks(single_track_yticks)
    ax.set_yticklabels([f'{y:.0f}' for y in single_track_ylabels])
    ax.set_ylim(0, 29)
    cax = inset_axes(
        ax, width="2.5%", height="100%", loc="lower left",
        bbox_to_anchor=(1.02, 0.0, 1.0, 1.0), bbox_transform=ax.transAxes, borderpad=0)
    cbar = fig1.colorbar(im, cax=cax)
    cbar.ax.set_title(single_track_cbar_title, fontsize=fontsize_label, pad=labelpad)
    if single_track_observable == 'similarity':
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(['0', '0.5', '1'])
        cbar.ax.yaxis.set_ticks([0.25, 0.75], minor=True)
    else:
        cbar.ax.ticklabel_format(axis='y', style='sci', scilimits=(4, 4), useMathText=True)
    cbar.ax.tick_params(labelsize=fontsize_ticks)
    cbar.ax.tick_params(which='minor', length=2)

# Panel C - cortical activation multiplicity.
activation_values = [activation_fraction[key] for key in activation_order]
ax_multiplicity.bar(
    activation_positions,
    activation_values,
    width=activation_width,
    color=[activation_colors[key] for key in activation_order],
    edgecolor=[activation_colors[key] for key in activation_order],
    linewidth=linewidth,
    alpha=1)
for event_key, key1, key2 in [
        ('doublet', 'doublet_same', 'doublet_mixed'),
        ('triplet', 'triplet_same', 'triplet_mixed')]:
    pvalue = activation_stat_pvalues[event_key]
    if pvalue < 0.05:
        stars = '***' if pvalue < 0.001 else '**' if pvalue < 0.01 else '*'
        x1 = activation_position[key1]
        x2 = activation_position[key2]
        y1 = activation_fraction[key1]
        y2 = activation_fraction[key2]
        y = min(max(y1, y2) + 0.06, 1.08)
        h = 0.03
        ax_multiplicity.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color='black', lw=0.65 * linewidth)
        ax_multiplicity.text(0.5 * (x1 + x2), y + h, stars, ha='center', va='bottom',
                             fontsize=fontsize_label, fontweight='bold')
ax_multiplicity.set_xlim(activation_position['singlet'] - 0.65, activation_position['multiplet'] + 0.65)
ax_multiplicity.set_ylabel('up-states fraction', fontsize=fontsize_label, labelpad=labelpad + 2, rotation=90)
ax_multiplicity.set_xticks(activation_group_positions)
ax_multiplicity.set_xticklabels(activation_group_labels, rotation=45, ha='right', fontsize=fontsize_ticks)
ax_multiplicity.set_yticks([0, 0.5, 1])
ax_multiplicity.set_yticks([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], minor=True)
ax_multiplicity.tick_params(which='minor', length=2)
ax_multiplicity.set_ylim(0, 0.75)

# Panel D - reactivation performance.
for narea, area in enumerate(plot_areas):
    ax = ax_performance[narea]
    data = [performance_similarity_pooled[area][key] for key in performance_similarity_order]
    for x, values, key in zip(performance_similarity_positions, data, performance_similarity_order):
        boxplot = ax.boxplot(
            [values],
            positions=[x],
            patch_artist=True,
            widths=0.1,
            showfliers=False,
        )
        boxplot['boxes'][0].set_facecolor('white')
        boxplot['boxes'][0].set_alpha(0.4)
        boxplot['boxes'][0].set_edgecolor('black')
        boxplot['boxes'][0].set_linewidth(0.2 * linewidth)
        for element_key in ['whiskers', 'caps']:
            for element in boxplot[element_key]:
                element.set_color('black')
                element.set_linewidth(0.2 * linewidth)
        boxplot['medians'][0].set_color('black')
        boxplot['medians'][0].set_linewidth(0.8*linewidth)

        violinplot = ax.violinplot(
            [values],
            positions=[x],
            widths=0.25,
            showmeans=False,
            showmedians=False,
            showextrema=False)
        violinplot['bodies'][0].set_facecolor(performance_similarity_style[key]['color'])
        violinplot['bodies'][0].set_edgecolor("black")
        violinplot['bodies'][0].set_alpha(1)
        violinplot['bodies'][0].set_linewidth(linewidth/2)

    ax.set_title(area, fontsize=fontsize_title)
    ax.set_xlim(performance_similarity_positions[0] - 0.45, performance_similarity_positions[-1] + 0.45)
    ax.set_ylim(-0.1, 1.25)
    ax.set_xticks(performance_similarity_group_positions)
    ax.set_xticklabels(performance_similarity_group_labels,
                       rotation=45, ha='right', fontsize=fontsize_ticks)
    ax.set_ylabel(r'$\~ S$', fontsize=fontsize_label, labelpad=labelpad + 4, rotation=0)
    ax.set_yticks(similarity_major_ticks)
    ax.set_yticks(similarity_minor_ticks, minor=True)
    ax.tick_params(which='minor', length=2)
    for event_key, key1, key2, y in [
            ('doublet', 'doublet_same', 'doublet_mixed', 1.08),
            ('triplet', 'triplet_same', 'triplet_mixed', 1.08)]:
        pvalue = similarity_stat_pvalues[area][event_key]
        stars = '***' if pvalue < 0.001 else '**' if pvalue < 0.01 else '*' if pvalue < 0.05 else 'n.s.'
        x1 = performance_similarity_position[key1]
        x2 = performance_similarity_position[key2]
        h = 0.03
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color='black', lw=0.65 * linewidth)
        ax.text(0.5 * (x1 + x2), y + h, stars, ha='center', va='bottom',
                fontsize=fontsize_label, fontweight='bold' if stars != 'n.s.' else 'normal')

fig1.tight_layout()
letter_x = min(ax.get_position().x0 for ax in [ax_raster[0], ax_heatmap[0], ax_multiplicity]) - 0.03
for label, ax in zip(['A', 'B'], [ax_raster[0], ax_heatmap[0]]):
    fig1.text(letter_x, ax.get_position().y1 + 0.01, label,
              fontsize=fontsize_panel_letter, fontweight='bold', va='bottom', ha='right')
for label, ax in zip(['C', 'D'], [ax_multiplicity, ax_performance[0]]):
    fig1.text(ax.get_position().x0 - 0.03, ax.get_position().y1 + 0.01, label,
              fontsize=fontsize_panel_letter, fontweight='bold', va='bottom', ha='right')
fig1.text(
    0.5 * (min(ax.get_position().x0 for ax in ax_heatmap) + max(ax.get_position().x1 for ax in ax_heatmap)),
    max(ax.get_position().y1 for ax in ax_heatmap) + 0.025,
    single_track_row_title,
    ha='center', va='bottom', fontsize=fontsize_title, fontweight='bold')
fig1.text(
    0.5 * (min(ax.get_position().x0 for ax in [ax_multiplicity] + ax_performance) +
           max(ax.get_position().x1 for ax in [ax_multiplicity] + ax_performance)),
    max(ax.get_position().y1 for ax in [ax_multiplicity] + ax_performance) + 0.025,
    'reactivation performance',
    ha='center', va='bottom', fontsize=fontsize_title, fontweight='bold')

fig_path = fig1_path
fig1.savefig(fig_path, dpi=dpi, bbox_inches='tight')
plt.close(fig1)
