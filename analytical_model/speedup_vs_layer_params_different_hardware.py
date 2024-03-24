"""
Plots speedup vs varying one of the layer parameters. Has the capability of drawing a graph per layer specified.
It assumes that the last subplot is always free to be used for the legend.
It depends on data from CPU/GPU speedup is already available at a pickle file in DATA_ROOT.
It adds the FPGA speedup and plots all speedups on different layers.
"""
from model import *

import pickle
from pathlib import Path
import pandas as pd
from itertools import cycle
import pdb
import matplotlib.pyplot as plt
import matplotlib
from utils.result_saver import *
import numpy as np

network = 'PECAN'
draw_individual_layers = True # TODO: This doesn't work when False. Needs fixing.

layers = [0,6,7,12,13]
def map_layer(name):
    result = {
        0: 'Block1-Conv1',
        6: 'Block2-Conv1',
        7: 'Block2-Conv2',
        12: 'Block3-Conv1',
        13: 'Block3-Conv2',
    }
    return result[name]

# ----------------------------------------------

DATA_ROOT = './data'
DATA_FILE_NAME = 'different_networks_cpu_gpu_pq.pkl'
RESULTS_ROOT = "./results/speedup_vs_layer_params_different_hardware/"

# ----------------------------------------------

font = {'size'   : 18}

matplotlib.rc('font', **font)

NROWS = 1
NCOLS = 1
if draw_individual_layers:
    NCOLS = math.ceil(len(layers)/NROWS) + 1
print(NCOLS)
cnt_draw = 0

def get_x_from_cnt_draw(cnt_draw):
    return cnt_draw // NCOLS

def get_y_from_cnt_draw(cnt_draw):
    return cnt_draw % NCOLS

def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(data, path):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def map_name(name):
    result = {
        "L_s": "$L_s$",
    }
    if name in result:
        return result[name]
    return name

df = read_pickle(f'{DATA_ROOT}/{DATA_FILE_NAME}')


def run_on_hardware(inH, inW, inCh, outCh, kernel, L_s, N_p):
    input_shape = [
        # unrolled_input[0],
        1,
        inCh,
        inH,
        inW,
    ]
    # if L_s == 128:
    #     pdb.set_trace()
    filter_shape = [outCh, kernel, kernel]

    params = {
        "NUM_OUT_VEC": 16,
        "PROTO_LENGTH_VEC": 16,
        "NUM_PROTO_VEC": 16,
        "NUM_PROTO_LAYER": N_p,
        "NUM_SUBSPACE_VEC": 16,
        "PROTO_LENGTH_LAYER": L_s,
    }

    result = simulate_layer(
        INPUT_SHAPE             = input_shape,
        FILTER_SHAPE            = filter_shape,
        STRIDES                 = 1,
        FREQUENCY_DLA           = freq_dla,
        FREQUENCY_PQ            = freq_pq,
        MODEL_FREQUENCY         = model_frequency,
        OVERLAPPING_IN_CALC     = overlapping_calculation_with_loading_pq,
        USE_CUSTOM_EQUATION     = dla_custom_equation,
        **params
    )

    return result['PQ']['Ncycles_initialization'], result['PQ']['Ncycles_calculation_proto'], result['PQ']['Ncycles_calculation_pq'], result['PQ']['Ncycles_calculation'], result['PQ']['Ncycles'], result['DLA']['Ncycles']

# network = 'net.8.0.pointwise'

# sample = df[(df.network == network) & (df.batch == 64)].copy()
# sample = df[(df.network == network)]
sample = df
# pdb.set_trace()
if draw_individual_layers:
    sample = sample[(sample.layer.isin(layers)) & (sample['L_s'] < 65) & (sample['N_p'] < 65)].copy()
# pdb.set_trace()
pq_init, pq_calc_proto, pq_calc_pq, pq_calc, pq_total, dla = zip(*[run_on_hardware(inH, inW, inCh, outCh, kernel, L_s, N_p) for inH, inW, inCh, outCh, kernel, L_s, N_p in zip(sample['inH'], sample['inW'], sample['inCh'], sample['outCh'], sample['kernel'], sample['L_s'], sample['N_p'])])

sample['cyclesPQInit'] = pq_init
sample['cyclesPQCalc'] = pq_calc
sample['cyclesPQCalcProto'] = pq_calc_proto
sample['cyclesPQCalcPQ'] = pq_calc_pq
sample['cyclesPQ'] = pq_total
sample['cyclesDLA'] = dla

aggregation_functions = {
    'pq_cpu': 'sum', 
    'pq_gpu': 'sum', 
    'conventional_cpu': 'sum',
    'conventional_gpu': 'sum',
    'cyclesPQInit': 'sum',
    'cyclesPQCalc': 'sum',
    'cyclesPQCalcProto': 'sum',
    'cyclesPQCalcPQ': 'sum',
    'cyclesPQ': 'sum',
    'cyclesDLA': 'sum',
    'inH': 'first',
    'inW': 'first',
    'inCh': 'first',
    'outCh': 'first',
    'kernel': 'first',
    'kernel': 'first',
}

if not(draw_individual_layers):
    aggregation_functions['layer'] = 'min'

group_by = [sample['network'], sample['L_s'], sample['N_p']]
if draw_individual_layers:
    group_by.append(sample['layer'])

aggregated = sample.groupby(group_by).aggregate(aggregation_functions).reset_index()

sample['bound'] = np.where(sample['cyclesPQInit'] > sample['cyclesPQCalc'], 'Memory', 'Compute')
aggregated['bound'] = np.where(aggregated['cyclesPQInit'] > aggregated['cyclesPQCalc'], 'Memory', 'Compute')

if not os.path.exists(RESULTS_ROOT):
    os.makedirs(RESULTS_ROOT)

save_pickle(sample, f'{RESULTS_ROOT}/different_networks_layers_all_stats.pkl')
save_pickle(aggregated, f'{RESULTS_ROOT}/different_networks_aggregated_all_stats.pkl')

sample.to_csv(f'{RESULTS_ROOT}/different_networks_layers_all_stats.csv')
aggregated.to_csv(f'{RESULTS_ROOT}/different_networks_aggregated_all_stats.csv')




params = ['N_p', 'L_s']
other = ['L_s', 'N_p']


for param, exclude in zip(params, other):
    ch = 'a'
    plt.clf()
    plt.figure()
    cnt_draw = 0
    fig, ax = plt.subplots(nrows=NROWS,ncols=NCOLS, sharex=False, sharey=False, figsize=(20,6), squeeze=False)
    for layer in layers:
        plt_x = get_x_from_cnt_draw(cnt_draw)
        plt_y = get_y_from_cnt_draw(cnt_draw)
        cnt_draw+= 1
        # Filter by the network that we want to plot and make sure we only have points for the default value of the 
        # other value we're not plotting against now. Also, sort the values by our parameter to have the line going
        # from left to right and avoid drawing a point then go backwards drawing another point which spoils line plots.
        to_draw = aggregated[(aggregated.network==network) & (aggregated[exclude] == 16) & (aggregated.layer == layer)].sort_values(by=[param])
        # pdb.set_trace()
        ax[plt_x,plt_y].plot(to_draw[param], (to_draw['conventional_cpu'] - to_draw['pq_cpu']) / to_draw['pq_cpu'] * 100, label = f'CPU')
        ax[plt_x,plt_y].plot(to_draw[param], (to_draw['conventional_gpu'] - to_draw['pq_gpu']) / to_draw['pq_gpu'] * 100, label = f'GPU')

        ax[plt_x,plt_y].plot(to_draw[param], (to_draw['cyclesDLA'] - to_draw['cyclesPQ']) / to_draw['cyclesPQ'] * 100, label = f'PQA')

        ax[plt_x,plt_y].axhline(y=0, color='grey', linestyle='dotted')
        ax[plt_x,plt_y].set_xticks(range(0,65,16))
        to_draw_pq = to_draw[(to_draw.bound == 'Memory')]
        # ax[plt_x,plt_y].scatter(to_draw_pq[param], (to_draw_pq['cyclesDLA'] - to_draw_pq['cyclesPQ']) / to_draw_pq['cyclesPQ'] * 100, marker=r'$M$', s=60, label='Memory Bound')

        to_draw_pq = to_draw[(to_draw.bound == 'Compute')]
        # ax[plt_x,plt_y].scatter(to_draw_pq[param], (to_draw_pq['cyclesDLA'] - to_draw_pq['cyclesPQ']) / to_draw_pq['cyclesPQ'] * 100, marker=r'$C$', s=60, label='Compute Bound')

        ax[plt_x,plt_y].set_title(map_name(f'{map_layer(layer)}'))
        handles, labels = ax[plt_x,plt_y].get_legend_handles_labels()
    fig.supylabel('Latency Improvement (%)')
    # fig.suptitle(f'Latency Improvement on different platforms for {network}')
    ax[NROWS-1,NCOLS-1].axis('off')
    ax[NROWS-1,NCOLS-1].legend(handles = handles, labels= labels, loc='center')
    fig.supxlabel(map_name(param))
    fig.tight_layout()
    save_plt_figure_result(f'{RESULTS_ROOT}/speedup_vs_layer_params_different_hardware_{network}_{param}', paper=(param=='L_s'), appendix=(param=='N_p'))
