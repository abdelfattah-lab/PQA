"""
Generates a plot showing mean square error on the y-axis and latency on the x-axis.
It depends on the presence of a dataframe that contains the layers data including the dimensions
and the mean square errors. It looks for a specific layer, batch and tau and then simulates the rows
on the hardware with their given parameters(Ls and Np) and plots the graph.
"""
from utils.result_saver import *
from model import *

import pickle
from pathlib import Path
import pandas as pd
from itertools import cycle
import matplotlib
import numpy as np


DATA_ROOT = './data'
RESULTS_ROOT = "./results/mse_vs_latency"
LAYER_NAME = 'net.8.0.pointwise'
BATCH = 64
TAU = 0.01

if not os.path.exists(RESULTS_ROOT):
    os.makedirs(RESULTS_ROOT)

def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(data, path):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

pickles = list(Path(f'{DATA_ROOT}').rglob('*study_df.pkl'))
print(f"{len(pickles)} found!")

df = pd.concat([read_pickle(p) for p in pickles])

def run_on_hardware(unrolled_input, num_out_channel, num_proto, proto_length):
    assert(math.sqrt(unrolled_input[2]) * math.sqrt(unrolled_input[2]) == unrolled_input[2])
    input_shape = [
        # unrolled_input[0],
        1,
        unrolled_input[1],
        math.sqrt(unrolled_input[2]),
        math.sqrt(unrolled_input[2]),
    ]
    filter_shape = [num_out_channel, 1, 1]

    params = {
        "NUM_OUT_VEC": 16,
        "PROTO_LENGTH_VEC": 16,
        "NUM_PROTO_VEC": 16,
        "NUM_PROTO_LAYER": num_proto,
        "NUM_SUBSPACE_VEC": 16,
        "PROTO_LENGTH_LAYER": proto_length,
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

    return result['PQ']['Ncycles'], result['DLA']['Ncycles']


sample = df[(df.layer_name == LAYER_NAME) & (df.batch == BATCH) & (df.pq_tau == TAU)].copy()
# sample = df[(df.layer_name == LAYER_NAME)].copy()
# sample = df.copy()
pq, dla = zip(*[run_on_hardware(a,b,c,d) for a,b,c,d in zip(sample['unrolled_input'], sample['outCh'], sample['N_p'], sample['L_s'])])

sample['cyclesPQ'] = pq
sample['cyclesDLA'] = dla


save_pickle(sample, f'{RESULTS_ROOT}/single_batch_cycles_df.pkl')

sample.to_csv(f'{RESULTS_ROOT}/df_with_cycles.csv')

sample = sample[sample['cyclesPQ'] < 6e4].copy()

plots = ['fintune_all', 'fintune_w', 'fintune_p', 'output_frozen', 'params', 'enc_flops', 'size_table']
symbols = ['v','d','s', 'o', 'P', 'X', 'D']

font = {'size'   : 18}

matplotlib.rc('font', **font)

for item in plots:
    plt.clf()
    fig = plt.figure(figsize=(11, 6))

    cmap = cycle(plt.cm.get_cmap("tab10", 5).colors)
    color_np = []
    symbol_legends = []
    np_legends = []
    for i, np_ in enumerate([4,8,16,32, 64]):
        color = next(cmap)
        color_np.append(color)
        for j, ls in enumerate(sorted(list(set(sample['L_s'])))): # for all resulting L_s for this layer
            df_layer_filter = sample[(sample['N_p']==np_) & (sample['L_s']==ls)]
            if df_layer_filter.empty:
                print(f'empty datframe: np_:{np_}, ls: {ls}')
                continue

            cycles_count = df_layer_filter['cyclesPQ']
            df_finetune_all = np.log10(df_layer_filter[item].values)
            
            plt.scatter(cycles_count/1e4, df_finetune_all, label=f'$N_p$ = {np_} -- $L_s$={ls}', alpha=0.5, color=color, marker=symbols[j], s=130)
            if i == 0:
                symbol_legends.append(plt.scatter([],[],label=f'Ls={ls}',s=30, color='k', marker=symbols[j], alpha=0.9))

        np_legends.append(plt.scatter([],[],label=f'Np={np_}',s=0, color=color, alpha=0.8))

    plt.scatter(sample['cyclesDLA'].values[0]/1e4, -2.35,alpha=1, marker='v', color='purple', s=120)
    plt.text(x= sample['cyclesDLA'].values[0]/1e4 - 0.12, y= -2.30, s="DLA", color='purple')
    plt.grid(True, which='both')
    plt.ylabel('MSE Layer output (Log)')
    plt.xlabel("Latency (x 1e4 Cycles)")
    np_legend_done = plt.legend(handles=np_legends, ncol=1, bbox_to_anchor=(0.8, 1), fancybox=True, labelcolor=color_np, handletextpad=0, handlelength=0)
    plt.gca().add_artist(np_legend_done)
    plt.legend(handles=symbol_legends, ncol=1, loc='upper right', markerscale=2)
    fig.tight_layout()
    save_plt_figure_result(f'{RESULTS_ROOT}/mse_vs_latency_{item}', appendix=(item=='fintune_all'), paper=False)
