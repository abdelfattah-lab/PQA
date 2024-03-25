#include "../inc/host.h"

int validate(proto_t* in, proto_t* proto_table, lut_t* lut_pq, acc_t* out, ind_t* ind,
    int WEIGHT_H, int IN_W, int N_S, int L_S, int N_P, bool test_ind);

#define FLT_ERROR_FACTOR 1000000
