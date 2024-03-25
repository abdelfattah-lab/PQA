"""
Plots a heatmap that shows speedup of PQA over DLA for different values of Np, Ls, input dimensions.
KERNEL_SIZE controls which kernel size is used for convolution.
"""
from utils.result_saver import *
from model import *
from utils.heatmap import *
import csv
import pdb

KERNEL_SIZE = 3

def generate_layer(channel, dimension):
    """Generates a layer with the given number of channels(Input and output are assumed to be of the same size),
    and a 2D image of size dimension x dimension and a kernel of size kernel x kernel. Stride and Dilation are 1.
    Batch of input is assumed to be 1 as well."""
    return [[1, channel, dimension, dimension], [channel, KERNEL_SIZE, KERNEL_SIZE], 1]


def generate_networks(
    lower_ch    = 8,
    upper_ch    = 1024,
    lower_dim   = 4,
    upper_dim   = 224,
    horizontal  = [16,32,64,128,256],
):
    networks = {}
    for val in horizontal:
        layers = []
        channel = upper_ch
        while channel >= lower_ch:
            dimension = lower_dim
            while dimension <= upper_dim:
                layers.append(generate_layer(channel, dimension))
                dimension*= 2
            channel//= 2
        networks[val] = layers.copy()
    return networks

def vary_input_params(memory):

    trials = {
        "PROTO_LENGTH_LAYER": [8,16,32,64,128,256],
    }

    networks = generate_networks()

    RESULTS_ROOT = "./results/layer_params_dimensions_speedup_heatmap"
    if not os.path.exists(RESULTS_ROOT):
        os.makedirs(RESULTS_ROOT)


    for item, rang in trials.items():
        results = []
        limiting_factors_results = []

        for trial_i in rang:
            for network, layers in networks.items():
                network_root = f'{RESULTS_ROOT}/{memory}/{network}'
                if not os.path.exists(network_root):
                    os.makedirs(network_root)

                input_dimensions = []
                number_of_channels = []
                speedups = []
                limiting_factors = []

                with open(f'{network_root}/layer_numbers.csv', 'w') as f:
                    writer = csv.writer(f)

                    writer.writerow(['Input Dimensions', 'Filter Dimensions', 'Number of Channels', 'Cycles Initialize PQ', 'Cycles Compute PQ', 'Cycles PQ', 'Limiting Factor PQ','Cycles Initialize DLA', 'Cycles Compute DLA', 'Cycles DLA', 'Limiting Factor DLA', 'PQ Speedup'])

                    freq_dla = 300e6
                    # TODO: Note that the frequency here is the result for synthesizing an all 16 hardware.
                    freq_pq = 1e9 / 5.3 # NOTE: This is ignored if model_frequency is set to false. freuqnecy of DLA is used instead.
                    if (not(model_frequency)):
                        freq_pq = freq_dla
                    
                    params = {
                        "NUM_OUT_VEC": 16,
                        "PROTO_LENGTH_VEC": 16,
                        "NUM_PROTO_VEC": 16,
                        "NUM_PROTO_LAYER": network,
                        "NUM_SUBSPACE_VEC": 16,
                        "PROTO_LENGTH_LAYER": 16,
                    }
                    
                    params[item] = trial_i
                        
                    for layer in layers:
                        result = simulate_layer(
                            INPUT_SHAPE             = layer[0],
                            FILTER_SHAPE            = layer[1],
                            STRIDES                 = layer[2],
                            FREQUENCY_DLA           = freq_dla,
                            FREQUENCY_PQ            = freq_pq,
                            MODEL_FREQUENCY         = model_frequency,
                            OVERLAPPING_IN_CALC     = overlapping_calculation_with_loading_pq,
                            USE_CUSTOM_EQUATION     = dla_custom_equation,
                            MEMORY_TYPE             = memory,
                            **params
                        )
                        cycles_pq = result[PQ_KEY]['Ncycles']
                        cycles_dla = result[DLA_KEY]['Ncycles']
                        limiting_factor_pq  = get_limiting_factor(result['PQ']['Ncycles_initialization'], result['PQ']['Ncycles_calculation'])
                        limiting_factor_dla = get_limiting_factor(result['DLA']['Ncycles_initialization'], result['DLA']['Ncycles_calculation'])
                        speedup_layer = math.log((cycles_dla/cycles_pq), 10)
                        writer.writerow([layer[0][2], layer[1][1], layer[0][1], result['PQ']['Ncycles_initialization'], result['PQ']['Ncycles_calculation'], cycles_pq, limiting_factor_pq, result['DLA']['Ncycles_initialization'], result['DLA']['Ncycles_calculation'], cycles_dla, limiting_factor_dla, speedup_layer])
                        input_dimensions.append(layer[0][2])
                        number_of_channels.append(layer[0][1])
                        speedups.append(speedup_layer)
                        limiting_factors.append(limiting_factor_pq)
                    input_dimensions = np.array(input_dimensions)
                    # number_of_channels = np.array(number_of_channels)[::-1] # I don't understand why this was here to reverse but I removed it because it doesn't make sense. I sanity checked the results and without it makes sense.
                    number_of_channels = np.array(number_of_channels)
                    speedups = np.array(speedups)
                    # pdb.set_trace()
                    x_coords, y_coords, result, limiting_factors_result = convert_to_heatmap_matrix(input_dimensions, number_of_channels, speedups, limiting_factors)
                    results.append(result)
                    limiting_factors_results.append(limiting_factors_result)

        # Prepare to draw the heatmap
        h_titles = []
        for val in networks.keys():
            h_titles.append(f'Np = {val}')

        v_titles = []
        for val in list(trials.values())[0]:
            v_titles.append(f'Ls = {val}')

        plt.clf()
        plt.figure()
        draw_heatmap(
            x_coords, y_coords, results, 
            labels = limiting_factors_results, 
            nrows= 6, 
            h_label='Input Dimensions', 
            v_label='Number of Channels', 
            h_titles=h_titles, 
            v_titles=v_titles,
            fn_cbar_label_mapper=map_log_to_x
        )

        save_plt_figure_result(f'{RESULTS_ROOT}/layer_params_dimensions_speedup_heatmap_kernel_{KERNEL_SIZE}_memory_{memory}', paper=(KERNEL_SIZE == 3), appendix=False)

vary_input_params(DDR)
vary_input_params(HBM)
