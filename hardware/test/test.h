#ifndef _TEST_H_
#define _TEST_H_

#include "../inc/host.h"

typedef struct PQ_Config{
    int l_s;
    int n_s;
    int n_p;
    int weight_h;
    int in_w;
} pq_config;

static pq_config test[3] = {
    {
    .l_s = L_S_VEC,
    .n_s = N_S_VEC,
    .n_p = N_P_VEC,
    .weight_h = WEIGHT_H_VEC,
    .in_w = 4
    },
    {
    .l_s = L_S_VEC,
    .n_s = 2*N_S_VEC,
    .n_p = 2*N_P_VEC,
    .weight_h = 2*WEIGHT_H_VEC,
    .in_w = 4
    },
    {
    .l_s = L_S_VEC,
    .n_s = 4*N_S_VEC,
    .n_p = 4*N_P_VEC,
    .weight_h = 4*WEIGHT_H_VEC,
    .in_w = 32
    }, 
};

#endif