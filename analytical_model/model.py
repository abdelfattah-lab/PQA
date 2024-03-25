"""
Analytical Model that can be used to simulate running a layer on DLA and PQA with lots of configurable options.
See simulate_layer function for available options and their default values.
"""
import math
import argparse

# Assumed shape of input: [Batch, Channels, Height, Width]
IN_BATCH_DIM    = 0
IN_CHAN_DIM     = 1
IN_HEIGHT_DIM   = 2
IN_WIDTH_DIM    = 3

# Assumed shape of filters: [Channels, Height, Width]
F_CHAN_DIM      = 0
F_HEIGHT_DIM    = 1
F_WIDTH_DIM     = 2

# Constants
PQ_KEY          = 'PQ'
DLA_KEY         = 'DLA'
MEMORY_BOUND    = 'M'
COMPUTE_BOUND   = ''
ROOFLINE_BOUND  = ''
HBM             = 'HBM'
DDR             = 'DDR'

def simulate_layer_on_pq(
    # Layer parameters.
    FILTER_SHAPE        ,
    INPUT_SHAPE         ,
    STRIDES             ,
    NUM_INPUT_COLS      ,
    NUM_OUT_CHAN        ,

    NUM_PROTO_LAYER     ,
    PROTO_LENGTH_LAYER  ,
    #-------------------------------------------#
    # PQ configurable parameters.
    NUM_PROTO_VEC       ,
    PROTO_LENGTH_VEC    ,
    NUM_SUBSPACE_VEC    ,
    NUM_OUT_VEC         ,
    FREQUENCY_PQ        ,
    MAX_OUT_CHANNEL     ,
    PROTO_TABLE_ENTRY   ,
    PQ_TABLE_ENTRY      ,
    MEM_BITS_PER_UNIT   ,
    LIMITED_INTERFACE   ,
    DUAL_PORT           ,
    OVERLAPPING_IN_CALC ,
    L2_DIFF             ,
    #-------------------------------------------#
    # Internal hardware assumptions.
    M20K_CLBS           ,
    DSP_CLBS            ,
    #-------------------------------------------#
    # External hardware parameters.
    ACTUAL_MAX_IN_BITS_PER_SEC ,
    #-------------------------------------------#
    # Configurable options.
    MODEL_FREQUENCY     ,
    #-------------------------------------------#
    # Resources Available (TODO: (Not Critical) Currently has no effect.
    MAX_DSPs            ,
    MAX_BRAMs           ,
    **others
):
    """Simulates a layer on the PQ hardware and returns a dictionary containing the summary of the results"""

    """## Constants Used"""


    """## Calculated Parameters"""

    # The length of the prototype in the layer. This is equal to the number of rows of the in matrix / the number of subspaces.
    # The number of rows of the in matrix is equal to input_channels * Filter Height * Filter Width.
    NUM_SUBSPACE_LAYER = math.ceil(INPUT_SHAPE[IN_CHAN_DIM]*FILTER_SHAPE[F_HEIGHT_DIM]*FILTER_SHAPE[F_WIDTH_DIM]/PROTO_LENGTH_LAYER)

    # If we assume that each cycle, we produce NUM_OUT_VEC, the total number of cycles it would take us to finish producing output would be NUM_OUT_CHAN / NUM_OUT_VEC.
    # We can use all that time to get new inputs without wasting hardware resources.
    CYCLE_ALLOWANCE_FOR_PROTO_LOOKUP = math.ceil(NUM_OUT_CHAN / NUM_OUT_VEC)

    # Based on the allowed number of cycles for proto lookup, we can decrease the number of subspaces results being produced per cycle to be
    # equal to the number of subspaces to be divided on the cycle allowance.
    NUM_SUBSPACE_VECS_IN_PROTOTYPE_PER_CYCLE = NUM_SUBSPACE_VEC / CYCLE_ALLOWANCE_FOR_PROTO_LOOKUP

    # Minimum number of inputs needed on the interface per cycle to utilize the hardware.
    NUMBER_OF_INPUTS_PER_CYCLE = PROTO_LENGTH_VEC * NUM_SUBSPACE_VECS_IN_PROTOTYPE_PER_CYCLE

    # Size of the tables that are needed for the operations. These are the actual data sizes for this layer, not the size allocated on the hardware.
    PQ_TABLE_SIZE = NUM_SUBSPACE_LAYER * NUM_OUT_CHAN * NUM_PROTO_LAYER
    PROTO_TABLE_SIZE = NUM_SUBSPACE_LAYER * NUM_PROTO_LAYER * PROTO_LENGTH_LAYER   # Number of data items that needs to be loaded into the proto table. Size of the max allowed proto table in hardware differs.

    # The maximum number of bits that can be fetched from an external memory per cycle.
    # It is calculated from the bit rate of the external memory and the frequency of the clock cycle of the hardware.
    MAX_IN_BITS_PER_CYCLE_PQ = math.ceil(ACTUAL_MAX_IN_BITS_PER_SEC / FREQUENCY_PQ)

    # Number of memory ports or in other words, the number of parallel reads/writes that can happen in that memory.
    MEMORY_PORTS = 1
    if (DUAL_PORT):
        MEMORY_PORTS = 2


    """## Initialization Cycles"""

    # Assuming that we're doing the minimum possible interface to utilize the hardware, this calculates the needed cycles for initialization given this interface.
    Min_Achievable_Ncycles_PQ_initialization_Limited_Interface_Proto = math.ceil(PQ_TABLE_SIZE / NUMBER_OF_INPUTS_PER_CYCLE)
    Min_Achievable_Ncycles_PQ_initialization_Limited_Interface_PQ = math.ceil(PROTO_TABLE_SIZE / NUMBER_OF_INPUTS_PER_CYCLE)
    Min_Achievable_Ncycles_PQ_initialization_Limited_Interface = Min_Achievable_Ncycles_PQ_initialization_Limited_Interface_Proto + Min_Achievable_Ncycles_PQ_initialization_Limited_Interface_PQ # Min Achievable given that we won't allow more input than the interface allows.

    # Number of actual bits to be stored in each of the tables.
    PROTO_TABLE_SIZE_BITS = PROTO_TABLE_SIZE * PROTO_TABLE_ENTRY
    PQ_TABLE_SIZE_BITS = PQ_TABLE_SIZE * PQ_TABLE_ENTRY

    # Number of cycles needed for initialization assuming full utilization of external memory link.
    Min_Achievable_Ncycles_PQ_initialization_Infinite_Interface_External_Memory_Limit = math.ceil((PROTO_TABLE_SIZE_BITS + PQ_TABLE_SIZE_BITS)/ MAX_IN_BITS_PER_CYCLE_PQ)

    # Number of cycles needed for initialization assuming full utilization of internal memory ports.
    Min_Achievable_Ncycles_PQ_initialization_Infinite_Interface_Proto = 1 # current pragmas say this: math.ceil(NUM_SUBSPACE_VEC * PROTO_LENGTH_VEC / 2). However, I think it must be fully partitioned.
    # To initialize PQ we need to enter NUM_SUBSPACE_VEC * NUM_OUT_CHAN * NUM_PROTO_VEC items but it is actually partitioned on NUM_SUBSPACE_VEC and NUM_OUT_VEC/number of memory ports.
    Min_Achievable_Ncycles_PQ_initialization_Infinite_Interface_PQ = math.ceil(NUM_OUT_CHAN * NUM_PROTO_LAYER * NUM_SUBSPACE_LAYER / (NUM_OUT_VEC * NUM_SUBSPACE_VEC)) # Right one
    # Min_Achievable_Ncycles_PQ_initialization_Infinite_Interface_PQ = math.ceil(NUM_OUT_CHAN * NUM_PROTO_LAYER / (NUM_OUT_VEC)) # Wrong one
    # pdb.set_trace()
    # print(f'{Min_Achievable_Ncycles_PQ_initialization_Infinite_Interface_PQ} < {Min_Achievable_Ncycles_PQ_initialization_Infinite_Interface_External_Memory_Limit}')
    # assert(Min_Achievable_Ncycles_PQ_initialization_Infinite_Interface_PQ < Min_Achievable_Ncycles_PQ_initialization_Infinite_Interface_External_Memory_Limit)
    # The needed cycles for initialization is the maximum between external memory limits and internal memory limits.
    Min_Achievable_Ncycles_PQ_initialization_Infinite_Interface = max(Min_Achievable_Ncycles_PQ_initialization_Infinite_Interface_Proto, Min_Achievable_Ncycles_PQ_initialization_Infinite_Interface_PQ, Min_Achievable_Ncycles_PQ_initialization_Infinite_Interface_External_Memory_Limit)

    # Choose the final number of cycles for initialization given the configuration.
    Ncycles_PQ_initialization = 0 # NUM_PROTO_VEC + NUM_OUT_CHAN * NUM_PROTO_VEC # Current TestBench rate. Ignored
    if (LIMITED_INTERFACE):
        Ncycles_PQ_initialization = Min_Achievable_Ncycles_PQ_initialization_Limited_Interface
    else:
        Ncycles_PQ_initialization = Min_Achievable_Ncycles_PQ_initialization_Infinite_Interface

    """## Calculation Cycles"""

    # Number of cycles needed to do the proto lookup part.
    Ncycles_PQ_proto_lookup             = math.ceil(PROTO_LENGTH_LAYER / PROTO_LENGTH_VEC) * math.ceil(NUM_PROTO_LAYER / NUM_PROTO_VEC)
    # Number of cycles needed to do the PQ Table lookup part.
    Ncycles_PQ_pq_lookup                = math.ceil(NUM_OUT_CHAN / NUM_OUT_VEC)
    # Bottleneck cycles needed to do the calculation which is the maximum of the above 2 numbers. This is the number it takes to calculate the output of a single input column.
    Ncycles_PQ_calculation_bottleneck   = max(Ncycles_PQ_proto_lookup, Ncycles_PQ_pq_lookup)
    # If not all subspaces can fit in the hardware(i.e: NUM_SUBSPACE_VEC < NUM_SUBSPACE_LAYER), we will need to repeat the process till we cover all NUM_SUBSPACE_LAYER.
    Ncycles_PQ_Vectorizing_Subspaces    = Ncycles_PQ_calculation_bottleneck * math.ceil(NUM_SUBSPACE_LAYER / NUM_SUBSPACE_VEC)
    # Number of cycles for calculation is equal to the number of elements in the final matrix which is equal to number of output height * output width * number of cycles needed
    # to calculate the output of a single input column. Output height * output width is equivalent to the unrolled input columns.
    Ncycles_PQ_calculation              = NUM_INPUT_COLS * Ncycles_PQ_Vectorizing_Subspaces # Assuming II=1

    """## Full Cycles Needed"""

    # Depending on the mode of calculation, we can either be doing initialization followed by calculation or doing initialization and calculation at the same time.
    Ncycles_PQ = Ncycles_PQ_initialization + Ncycles_PQ_calculation
    if (OVERLAPPING_IN_CALC):
        Ncycles_PQ = max(Ncycles_PQ_initialization, Ncycles_PQ_calculation)

    """## Time Needed"""

    # Time needed by the PQ hardware to process this layer.
    TIME_PQ = Ncycles_PQ / FREQUENCY_PQ

    MEM_NEEDED_BITS = NUM_INPUT_COLS * NUM_SUBSPACE_LAYER * PROTO_LENGTH_LAYER * PROTO_TABLE_ENTRY
    MEM_UNITS_NEEDED_FOR_INPUT_STORAGE = math.ceil(MEM_NEEDED_BITS / MEM_BITS_PER_UNIT)

    # Extracted from linear regression over multiple synthesized configurations. The number of CLBs not used for memory or DSPs.
    EXTRA_CLBS_PQ = 54.0152229 * NUM_OUT_VEC + 0/2 * PQ_TABLE_ENTRY + 0/2 * PROTO_TABLE_ENTRY + 197.18071766 * NUM_SUBSPACE_VEC + 59.99415826 * PROTO_LENGTH_VEC + 54.0152229 * NUM_PROTO_VEC - 1248.8141835982537
    DSPs          = 0

    if L2_DIFF:
        DSPs = NUM_OUT_VEC * NUM_SUBSPACE_VEC * PROTO_LENGTH_VEC

    """## Area Needed"""

    # Needed memory ports. It can be 1 if calculation is done separately from initialization.
    # It would be 2 if we assumed that calculation happens in parallel with initialization of the next layer.
    NEEDED_PORTS = 1
    if (OVERLAPPING_IN_CALC):
        NEEDED_PORTS = 2
    M20K_PE = math.ceil(NUM_SUBSPACE_VEC * NUM_OUT_VEC / MEMORY_PORTS) * NEEDED_PORTS
    pq_area_requirements = DSPs * DSP_CLBS + M20K_PE * M20K_CLBS + EXTRA_CLBS_PQ


    FLOPs = 3 * NUM_SUBSPACE_LAYER * NUM_PROTO_LAYER * PROTO_LENGTH_LAYER * INPUT_SHAPE[IN_HEIGHT_DIM] * INPUT_SHAPE[IN_WIDTH_DIM] + (NUM_SUBSPACE_LAYER - 1) * INPUT_SHAPE[IN_HEIGHT_DIM] * INPUT_SHAPE[IN_WIDTH_DIM] * NUM_OUT_CHAN

    """## Results"""

    # Container for the results.
    result = {}
    result['PROTO_TABLE_SIZE']          = PROTO_TABLE_SIZE
    result['PQ_TABLE_SIZE']             = PQ_TABLE_SIZE
    result['Ncycles_initialization']    = Ncycles_PQ_initialization
    result['Ncycles_calculation_proto'] = Ncycles_PQ_proto_lookup
    result['Ncycles_calculation_pq']    = Ncycles_PQ_pq_lookup
    result['Ncycles_calculation']       = Ncycles_PQ_calculation
    result['Ncycles']                   = Ncycles_PQ
    result['M20K_PE']                   = MEM_UNITS_NEEDED_FOR_INPUT_STORAGE
    result['area']                      = pq_area_requirements
    result['FLOPs']                     = FLOPs

    return result

def simulate_layer_on_dla(
    # Layer parameters.
    FILTER_SHAPE        ,
    INPUT_SHAPE         ,
    NUM_OUT_CHAN        ,
    STRIDES             ,
    DATA_TYPE           ,
    #-------------------------------------------#
    # DLA configurable parameters. Current default values are the best configuration mentioned in the paper.
    Cvec                ,
    Kvec                ,
    Qvec                ,
    Wvec                ,
    Svec                ,
    Lh                  ,
    Lw                  ,
    FREQUENCY_DLA       ,
    USE_WINOGRAD        ,
    USE_CUSTOM_EQUATION ,
    #-------------------------------------------#
    # Internal hardware assumptions
    M20K_CLBS           ,
    DSP_CLBS            ,
    EXTRA_CLBS_DLA      ,
    #-------------------------------------------#
    # External hardware parameters
    ACTUAL_MAX_IN_BITS_PER_SEC ,
    #-------------------------------------------#
    # Resources Available (TODO: (Not Critical) Currently has no effect.
    MAX_DSPs            ,
    MAX_BRAMs           ,
    **others            ,
):
    """## Translation of layer into parameter"""

    C   = INPUT_SHAPE[IN_CHAN_DIM]              # Number of input channels.
    K   = NUM_OUT_CHAN                          # Number of outpout channels.
    W   = INPUT_SHAPE[IN_WIDTH_DIM]             # Input Width.
    H   = INPUT_SHAPE[IN_HEIGHT_DIM]            # Input Height.
    Q   = INPUT_SHAPE[IN_WIDTH_DIM] / STRIDES   # Output Width.
    P   = INPUT_SHAPE[IN_HEIGHT_DIM] / STRIDES  # Output Height.
    FW  = FILTER_SHAPE[F_WIDTH_DIM]             # Filter Width.
    FH  = FILTER_SHAPE[F_HEIGHT_DIM]            # Filter Height.

    """## Calculated Parameters"""
    # The maximum number of bits that can be fetched from an external memory per cycle.
    # It is calculated from the bit rate of the external memory and the frequency of the clock cycle of the hardware.
    MAX_IN_BITS_PER_CYCLE = math.ceil(ACTUAL_MAX_IN_BITS_PER_SEC / FREQUENCY_DLA)

    """## Resource usage per PE"""

    DSP_PE = (Wvec - Qvec + 1) * Qvec * Kvec * Cvec * 0.5 # Not using Winograd.
    DSP_PE_WINOGRAD = math.ceil(DSP_PE * 0.5 + 200) # Using Winograd.
    if USE_WINOGRAD:
        DSP_PE = DSP_PE_WINOGRAD

    Nbanks = Wvec * Cvec
    M20K_FILTER = Nbanks * Kvec / 2
    Depth_in = C * W * H / Nbanks
    Depth_out = K * Q * P / Nbanks
    M20K_DATA = math.ceil(Depth_in + Depth_out/(512*2)) * Nbanks
    M20K_PE = M20K_FILTER #+ M20K_DATA

    """## Operations needed"""

    # Initialization Cycles.
    MEM_NEEDED_BITS = K * FW * FH * C * DATA_TYPE
    Min_Achievable_Ncycles_initialization_External_Memory_Limit = math.ceil(MEM_NEEDED_BITS / MAX_IN_BITS_PER_CYCLE)


    # Compute Cycles.
    NUMBER_OF_ENGINES = 1 # math.floor(MAX_DSPs / DSP_PE)
    DSPeff = Q/(math.ceil(Q/(Qvec * Lw)) * Qvec * Lw) * P/(math.ceil(P/(Lh)) * Lh)

    Nflops = 2 * K * C * Q * P * DSPeff

    Ncycles_Per_Engine = Nflops/(DSP_PE * 2)

    if USE_CUSTOM_EQUATION:
        Ncycles_Per_Engine = INPUT_SHAPE[IN_BATCH_DIM] * math.floor(math.ceil(K/Kvec) * (P * math.ceil(Q/Qvec)) * (FW/Svec) * FH * math.ceil(C/Cvec))

    Ncycles_Compute = INPUT_SHAPE[IN_BATCH_DIM] * math.ceil(Ncycles_Per_Engine / NUMBER_OF_ENGINES)

    # Total Cycles. It is always assumed that they have overlapping calculation with memory load.
    Ncycles_DLA = max(Min_Achievable_Ncycles_initialization_External_Memory_Limit, Ncycles_Compute)

    TIME_DLA = Ncycles_DLA / FREQUENCY_DLA

    # print(f'M20: {M20K_PE}, DSP: {DSP_PE}, Extra_CLB: {EXTRA_CLBS_DLA}, Total: {M20K_PE * M20K_CLBS + DSP_PE * DSP_CLBS + EXTRA_CLBS_DLA}')
    dla_area_requirements = M20K_PE * M20K_CLBS + DSP_PE * DSP_CLBS + EXTRA_CLBS_DLA

    """## Results"""

    result                              = {}

    result['Ncycles_initialization']    = Min_Achievable_Ncycles_initialization_External_Memory_Limit
    result['Ncycles_calculation']       = Ncycles_Compute
    result['Ncycles']                   = Ncycles_DLA
    result['DSP_PE']                    = DSP_PE
    result['M20K_PE']                   = M20K_PE
    result['area']                      = dla_area_requirements

    return result

# Extra function to simulate a layer on the accelerator in the following paper with some assumptions:
# Wu, D., Zhang, Y., Jia, X., Tian, L., Li, T., Sui, L., ... & Shan, Y. (2019, September). A high-performance CNN processor based on FPGA for MobileNets. In 2019 29th International Conference on Field Programmable Logic and Applications (FPL) (pp. 136-143). IEEE.
def simulate_layer_on_mobile_dla(
    # Layer parameters.
    FILTER_SHAPE        ,
    INPUT_SHAPE         ,
    NUM_OUT_CHAN        ,
    STRIDES             ,
    DATA_TYPE           ,
    #-------------------------------------------#

    #-------------------------------------------#
    # Internal hardware assumptions
    M20K_CLBS           ,
    DSP_CLBS            ,
    EXTRA_CLBS_DLA      ,
    #-------------------------------------------#
    # External hardware parameters
    ACTUAL_MAX_IN_BITS_PER_SEC ,
    #-------------------------------------------#
    # Resources Available (TODO: (Not Critical) Currently has no effect.
    MAX_DSPs            ,
    MAX_BRAMs           ,
    **others            ,
):
    """## Translation of layer into parameter"""

    C   = INPUT_SHAPE[IN_CHAN_DIM]              # Number of input channels.
    K   = NUM_OUT_CHAN                          # Number of outpout channels.
    W   = INPUT_SHAPE[IN_WIDTH_DIM]             # Input Width.
    H   = INPUT_SHAPE[IN_HEIGHT_DIM]            # Input Height.
    Q   = INPUT_SHAPE[IN_WIDTH_DIM] / STRIDES   # Output Width.
    P   = INPUT_SHAPE[IN_HEIGHT_DIM] / STRIDES  # Output Height.
    FW  = FILTER_SHAPE[F_WIDTH_DIM]             # Filter Width.
    FH  = FILTER_SHAPE[F_HEIGHT_DIM]            # Filter Height.


    ICP = 16
    OCP = 16
    PP = 8

    Ncycles_DLA = math.ceil(K/OCP) * math.ceil(P/PP) * Q * math.ceil(C/ICP) * FW * FH

    TIME_DLA = Ncycles_DLA / FREQUENCY_DLA

    # Configurations from the paper on DPU_L. Assumed BRAM = M20K_PE. DSP48E2 = DSP_PE.

    M20K_PE = 771
    DSP_PE  = 2070
    LUTs    = 161944
    FFs     = 301416
    EXTRA_CLBS_DLA = math.ceil(max(( LUTs / 4), (FFs / 8)) / 2)

    dla_area_requirements = M20K_PE * M20K_CLBS + DSP_PE * DSP_CLBS + EXTRA_CLBS_DLA

    """## Results"""

    result                              = {}

    result['Ncycles']                   = Ncycles_DLA
    result['DSP_PE']                    = DSP_PE
    result['M20K_PE']                   = M20K_PE
    result['area']                      = dla_area_requirements

    return result


# -*- coding: utf-8 -*-
"""product quantization analytical model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Z-C5Ba9qUvSB3gbZTeSHYFsC18ALmdI1

## Edit Parameters Below
"""

FREQUENCY_DLA = 300e6
FREQUENCY_PQ  = 259e6

def simulate_layer(
    # Layer parameters.
    FILTER_SHAPE        = [128,1,1],    # The shape of the filter. [Channels, Height, Width]
    INPUT_SHAPE         = [1,84,5,25],  # The shape of the input. [Batches, Channels, Height, Width]
    STRIDES             = 1,            # The Strides that the filter moves at each step.
    DILATION            = 1,
    PADDING             = 0,

    NUM_PROTO_LAYER     = 8,            # The total number of possible prototypes for the specific layer being calculated now.
    PROTO_LENGTH_LAYER  = 24,           # The length of each prototype in the layer.
    DATA_TYPE           = 16,           # The data type used to represent the weights and the inputs.
    #-------------------------------------------#
    # PQ configurable parameters
    NUM_PROTO_VEC       = 8,            # Max. Number of possible Prototypes in the hardware to compare with.
    PROTO_LENGTH_VEC    = 16,           # Max. Length of each Prototype value in the hardware.
    NUM_SUBSPACE_VEC    = 24,           # Max. Number of subspaces to which the weights and inputs can be divided into.
    # Based on the above 2 parameters, the maximum number of columns in weights = maximum number of rows in input = NUM_SUBSPACE_VEC * PROTO_LENGTH_VEC.
    NUM_OUT_VEC         = 32,           # The maximum number of outputs that come out in a single cycle from the hardware.
    FREQUENCY_PQ        = FREQUENCY_PQ, # The frequency of the PQ hardware. NOTE: This field is ignored if MODEL_FREQUENCY is set to false. Frequency in that case is assumed to be the same of the DLA.
    MAX_OUT_CHANNEL     = 384,          # The maximum number of all outputs that can be calculated by the hardware for a single layer. Considered the maximum input that can enter too.
    PROTO_TABLE_ENTRY   = 8,            # The size(in bits) of each entry in the proto table.
    PQ_TABLE_ENTRY      = 16,           # The size(in bits) of each entry in the result table.
    MEM_BITS_PER_UNIT   = 18 * 8e3,     # The size(in bits) of each memory unit.
    LIMITED_INTERFACE   = False,        # Whether to assume limited interface for initialization cycles or not. Limited interface means the number of inputs is limited to the minimum needed to make best use of the possible parallelization.
    DUAL_PORT           = True,         # Whether to assume that the memory is true dual port or not.
    OVERLAPPING_IN_CALC = True,         # Whether to assume that the input of next layer is entered in parallel with calculating the output of the current layer.
    L2_DIFF             = True,         # Whether to assume the more accurate L2 diff is calculated or not.
    #-------------------------------------------#
    # DLA configurable parameters. Current default values are the best configuration mentioned in the paper.
    Cvec                = 8,
    Kvec                = 48,
    Qvec                = 4,
    Wvec                = 6,
    Svec                = 3,         
    Lh                  = 1,            
    Lw                  = 1,
    FREQUENCY_DLA       = FREQUENCY_DLA,
    USE_WINOGRAD        = False,
    USE_CUSTOM_EQUATION = True,         # Use our own calculation of the FLOPS since their equation doesn't consider the filter size.
    #-------------------------------------------#
    # Internal hardware assumptions
    M20K_CLBS           = 4,            # Number of CLBs equivalent to 1 Memory unit.
    DSP_CLBS            = 3,            # Number of CLBs equivalent to 1 DSP.
    EXTRA_CLBS_DLA_PE   = (3.3e3)//10, # Assumed usage of CLBs of components other than M20K and DSPs.
    #-------------------------------------------#
    # External hardware parameters
    MEM_RATE_HBM        = 400 * 8e9,    # Max number of bits per second that can be brought from external memory into the chip in case of using HBM.
    MEM_RATE_DDR        = 35.2 * 8e9,   # Max number of bits per second that can be brought from external memory into the chip in case of using DDR.
    MEMORY_EFF          = 0.8,          # The efficiency of the external memory.
    MEMORY_TYPE         = 'DDR',        # The type of memory used. Only possibilities are 'DDR' or 'HBM'
    #-------------------------------------------#
    # Configurable options.
    MODEL_FREQUENCY     = True,
    #-------------------------------------------#
    # Resources Available (TODO: (Not Critical) Currently has no effect.
    MAX_DSPs            = 2280,
    MAX_BRAMs           = 1440,
):
    """Simulates a layer running fully on the hardware and returns a summary of result."""

    # Shape of the output that gets filled by this function.
    result = {
        'PQ': {
            'PROTO_TABLE_SIZE':             0,
            'PQ_TABLE_SIZE':                0,
            'Ncycles_initialization':       0,
            'Ncycles_calculation':          0,
            'Ncycles':                      0,
            'images_per_second':            0,
            'M20K_PE':                      0,
        },
        'DLA': {
            'Ncycles':                      0,
            'DSP_PE':                       0,
            'M20K_PE':                      0,
            'images_per_second':            0,
        }
    }

    UNROLLED_INPUT = [INPUT_SHAPE[IN_BATCH_DIM], INPUT_SHAPE[IN_CHAN_DIM], INPUT_SHAPE[IN_HEIGHT_DIM] * INPUT_SHAPE[IN_WIDTH_DIM] / (STRIDES * STRIDES)]
    UIN_BATCH_DIM   = 0
    UIN_ROW_DIM     = 1
    UIN_COL_DIM     = 2


    NUM_INPUT_COLS  = UNROLLED_INPUT[UIN_COL_DIM]
    NUM_OUT_CHAN    = FILTER_SHAPE[F_CHAN_DIM]

    # Apply configurations.
    if (not(MODEL_FREQUENCY)):
        FREQUENCY_PQ = FREQUENCY_DLA

    MAX_IN_BITS_PER_SEC = MEM_RATE_DDR
    if MEMORY_TYPE == HBM:
        MAX_IN_BITS_PER_SEC = MEM_RATE_HBM

    ACTUAL_MAX_IN_BITS_PER_SEC = MAX_IN_BITS_PER_SEC * MEMORY_EFF

    EXTRA_CLBS_DLA = EXTRA_CLBS_DLA_PE * Kvec


    params = locals()
    """# **PQ Hardware**"""

    result[PQ_KEY]    = simulate_layer_on_pq(**params)

    """# DLA Hardware"""

    result[DLA_KEY]   = simulate_layer_on_dla(**params)

    return result


# Configurable Parameters.
dla_custom_equation = True
model_frequency = False
overlapping_calculation_with_loading_pq = True
freq_dla = 300e6
freq_pq = 1e9 / 5.3 # NOTE: This is ignored if model_frequency is set to false. freuqnecy of DLA is used instead.


if (not(model_frequency)):
    freq_pq = freq_dla

def get_limiting_factor(cycles_initialization, cycles_calculation):
    if cycles_initialization > cycles_calculation:
        return MEMORY_BOUND
    elif cycles_initialization < cycles_calculation:
        return COMPUTE_BOUND
    else:
        return ROOFLINE_BOUND



def map_item(item):
    result = {
        "NUM_SUBSPACE_LAYER": "Ns",
        "NUM_PROTO_LAYER": "Np",
        "PROTO_LENGTH_LAYER": "Ls",
    }
    return result[item]

def map_log_to_x(items):
    result = []
    for item in items:
        item = 10**item
        result.append(f'{item}X')
    return result

# Make main
if __name__ == "__main__":
    # Take arguments from command line.
    parser = argparse.ArgumentParser(description='Analytical Model for Product Quantization Hardware')
    parser.add_argument('--filter_shape', nargs='+', type=int, default=[128,1,1], help='The shape of the filter. [Channels, Height, Width]')
    parser.add_argument('--input_shape', nargs='+', type=int, default=[1,84,5,25], help='The shape of the input. [Batches, Channels, Height, Width]')
    parser.add_argument('--num_proto_layer', type=int, default=8, help='The total number of possible prototypes for the specific layer being calculated now.')
    parser.add_argument('--proto_length_layer', type=int, default=24, help='The length of each prototype in the layer.')
    parser.add_argument('--data_type', type=int, default=16, help='The data type used to represent the weights and the inputs.')
    parser.add_argument('--num_proto_vec', type=int, default=8, help='Max. Number of possible Prototypes in the hardware to compare with.')
    parser.add_argument('--proto_length_vec', type=int, default=16, help='Max. Length of each Prototype value in the hardware.')
    parser.add_argument('--num_subspace_vec', type=int, default=24, help='Max. Number of subspaces to which the weights and inputs can be divided into.')
    parser.add_argument('--num_out_vec', type=int, default=32, help='The maximum number of outputs that come out in a single cycle from the hardware.')
    parser.add_argument('--frequency_pq', type=float, default=freq_pq, help='The frequency of the PQ hardware. NOTE: This field is ignored if MODEL_FREQUENCY is set to false. Frequency in that case is assumed to be the same of the DLA.')
    parser.add_argument('--max_out_channel', type=int, default=384, help='The maximum number of all outputs that can be calculated by the hardware for a single layer. Considered the maximum input that can enter too.')

    args = parser.parse_args()

    print(simulate_layer(
        FILTER_SHAPE        = args.filter_shape,
        INPUT_SHAPE         = args.input_shape,
        NUM_PROTO_LAYER     = args.num_proto_layer,
        PROTO_LENGTH_LAYER  = args.proto_length_layer,
        DATA_TYPE           = args.data_type,
        NUM_PROTO_VEC       = args.num_proto_vec,
        PROTO_LENGTH_VEC    = args.proto_length_vec,
        NUM_SUBSPACE_VEC    = args.num_subspace_vec,
        NUM_OUT_VEC         = args.num_out_vec,
        MAX_OUT_CHANNEL     = args.max_out_channel,
        FREQUENCY_PQ        = args.frequency_pq,
    ))