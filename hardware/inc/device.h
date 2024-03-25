#ifndef _DEVICE_H_
#define _DEVICE_H_

#include "config.h"

#include "ihc_apint.h"
typedef int16_t proto_t; // same as input and weight type
typedef int16_t lut_t;
typedef int32_t dist_t;  // require dist_t to have larger size than proto_t
typedef int8_t ind_t;
typedef int16_t acc_t; // same as out type

struct NVec_Ind_t{
    ind_t indices[N_S_VEC] __attribute__((__aligned__(8)));
    int in_w;
    int n_s;
} __attribute__((__aligned__(8)));

typedef struct NVec_Ind_t nvec_ind_t __attribute__((__aligned__(8)));

struct NVec_Out_t{
    acc_t data[WEIGHT_H_MAX/WEIGHT_H_VEC][WEIGHT_H_VEC] __attribute__((__aligned__(8)));
} __attribute__((__aligned__(8)));

typedef struct NVec_Out_t nvec_out_t __attribute__((__aligned__(8)));

struct NVec_Acc_t{
    lut_t data[WEIGHT_H_VEC][N_S_VEC] __attribute__((__aligned__(8)));
    bool new_col;
    bool last_data;
    int weight_h;
} __attribute__((__aligned__(8)));

typedef struct NVec_Acc_t nvec_acc_t __attribute__((__aligned__(8)));


#endif 