#include <cstdio>
#include "test.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include <bits/stdc++.h>
#include <iostream>
using namespace std;

void initialize_basic_array(proto_t* array, const int size)
{
    for (int i = 0; i < size; i += 2)
    {
        array[i] = (proto_t) 0;
        array[i + 1] = (proto_t) 1;
    }
}

float randomFloat()
{
    return (float)(rand()) / (float)(rand());
}

void initialize_random_array(proto_t* array, const int size)
{
  srand(time(0));
    for (int i = 0; i < size; i ++)
    {
        if(is_same<proto_t, float>::value) {
            array[i] = randomFloat();
        } else {
            // Limit for correctness with given bitwidth is 2^14
            array[i] = (proto_t) (rand()%(2^13));
        }            
    }
}

void initialize_numerate_array(proto_t* array, const int size)
{
    for (int i = 0; i < size; i ++)
    {
        array[i] = (proto_t) i;
    }
}

int main(int argc, char **argv) {

    for (int i = 0; i < argc; i ++) {
        cout << argv[i];
    }
    int test_i = (int(*argv[1] - '0'));

    int l_s = test[test_i].l_s;
    int n_s = test[test_i].n_s;
    int n_p = test[test_i].n_p;
    int weight_h = test[test_i].weight_h;
    int in_w = test[test_i].in_w;

    int in_size = n_s * in_w * l_s;
    int proto_table_size = n_p * l_s;
    int lut_pq_size = weight_h * n_s * n_p;
    int out_size = weight_h * in_w;
    int weight_size = weight_h * n_s * l_s;

    proto_t* in            = (proto_t*) malloc(in_size * sizeof(proto_t));
    proto_t* proto_table   = (proto_t*) malloc(proto_table_size * sizeof(proto_t));
    lut_t* lut_pq        = (lut_t*) malloc(lut_pq_size * sizeof(lut_t));
    acc_t* out           = (acc_t*) malloc(out_size * sizeof(acc_t));
    proto_t* weight        = (proto_t*) malloc(weight_size * sizeof(proto_t));
    int size[5] = {n_s, n_p, l_s, weight_h, in_w};
    
    // // Optional argument to specify whether the emulator should be used.
    if(argv[2] == ("random"s)) {
        initialize_random_array(in, in_size);
        initialize_random_array(proto_table, proto_table_size);
        initialize_random_array(weight, weight_size);
    } else if (argv[2] == ("numerate"s)) {
        initialize_numerate_array(in, in_size);
        initialize_numerate_array(proto_table, proto_table_size);
        initialize_numerate_array(weight, weight_size);
    } else {
        initialize_basic_array(in, in_size);
        initialize_basic_array(proto_table, proto_table_size);
        initialize_basic_array(weight, weight_size);
    }

    for (int i = 0; i < weight_h; i ++) {
        for (int j = 0; j < n_s; j ++) {
            for (int k = 0; k < n_p; k ++) {
                lut_t sum = 0;
                for (int l = 0; l < l_s; l++) {
                    sum += ((lut_t) weight[i*n_s*l_s+ j*l_s + l]) * ((lut_t) proto_table[k*l_s + l]);
                }
                lut_pq[i*n_s*n_p+ j*n_p + k] = sum;
            }   
        }   
    }


    FILE *fpIn = fopen("/home/ayc62/Documents/PQA/hardware/inc/in.bin", "wb");
    FILE *fpPrototable = fopen("/home/ayc62/Documents/PQA/hardware/inc/proto_table.bin", "wb");
    FILE *fpLutpq = fopen("/home/ayc62/Documents/PQA/hardware/inc/lut_pq.bin", "wb");
    FILE *fpSize = fopen("/home/ayc62/Documents/PQA/hardware/inc/size.bin", "wb");

    fwrite(in, sizeof(proto_t), in_size, fpIn);
    fwrite(proto_table, sizeof(proto_t), proto_table_size, fpPrototable);
    fwrite(lut_pq, sizeof(lut_t), lut_pq_size, fpLutpq);
    fwrite(&size, sizeof(int), 5, fpSize);

    fclose(fpIn);
    fclose(fpPrototable);
    fclose(fpLutpq);
    fclose(fpSize);
    
    free(in);
    free(proto_table);
    free(lut_pq);
    free(out);
    free(weight);

}
