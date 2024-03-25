#include "../inc/device.h"
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel nvec_ind_t indexChannel __attribute__((depth(256)));
channel nvec_acc_t accChannel __attribute__((depth(1024)));
channel nvec_out_t outChannel __attribute__((depth(256)));

__attribute__((max_global_work_dim(0)))
__kernel void
distance_calc(__global const short *restrict in,
              __global const short *restrict proto_table,
              short L_S,
              short N_S,
              short N_P,
              short WEIGHT_H,
              short IN_W
#ifdef DEBUG
              ,
              __global ind_t *restrict test_indices
#endif
)
{

  proto_t __attribute__((numbanks(N_S_VEC*L_S_VEC))) in_temp[N_S_MAX*IN_W_MAX*L_S_MAX/(N_S_VEC*L_S_VEC)][N_S_VEC][L_S_VEC];
  proto_t __attribute__((numbanks(N_P_VEC*L_S_MAX))) proto_table_temp[N_P_MAX/N_P_VEC][N_P_VEC][L_S_MAX];

  // TODO: unroll based on compute structure

  //---------------
  //  load prototype table
  //---------------
  for (short i = 0; i < N_P_MAX/N_P_VEC; i++) {
    for (short j = 0; j < N_P_VEC; j++) {
      for (short k = 0; k < L_S; k++) {
        proto_table_temp[i][j][k] = (proto_t) proto_table [(i*N_P_VEC + j)*L_S + k];
        
      }
    }
  }

  //---------------
  //  load in
  //---------------

  short NUM_NS = N_S/N_S_VEC;
  for (short i = 0; i < NUM_NS; i++) {
    for (short j = 0; j < N_S_VEC; j++) {
      for (short k = 0; k < IN_W; k++) {
        for (short l = 0; l < L_S; l++) {
          in_temp[k*NUM_NS+i][j][l] = (proto_t) in[i*IN_W*L_S*N_S_VEC + j*IN_W*L_S + k*L_S + l];
        }
      }
    }
  }

  short pl_ns_counter = 0;
  short pl_in_width_pos = 0;
  short pl_np_counter = 0; // store current index of n_p

  nvec_ind_t index_local;
  dist_t min_dist_arr[N_S_VEC]; 

  #pragma ii 1
  do {
    index_local.in_w = pl_in_width_pos;
    index_local.n_s = pl_ns_counter;

    // store min dist for n_s_vec group for calculations across n_p_vec

    #pragma unroll
    for (short i = 0; i < N_S_VEC; i++) {

      // store dist for each prototype to calculate min dist
      dist_t dist_arr[N_P_VEC];

      // distance calculation
      #pragma unroll
      for (short j = 0; j < N_P_VEC; j++) {

        dist_arr[j] = 0;
        #pragma unroll
        for (short k = 0; k < L_S_VEC; k++) {
          dist_t diff = in_temp[pl_in_width_pos*NUM_NS + pl_ns_counter][i][k] - proto_table_temp[pl_np_counter/N_P_VEC][j][k];
          dist_t dist;
          #ifdef L1
          dist = (diff < 0) ? (-diff) : diff;
          #else
          dist = diff * diff;
          #endif
          dist_arr[j] += dist;
        }
      }



      // comparator to find the smallest in
      dist_t min_dist = dist_arr[0];
      ind_t min_idx = 0 + pl_np_counter;

      #pragma unroll
      for (short j = 1; j < N_P_VEC; j++) {
        if (dist_arr[j] < min_dist) {
          min_dist = dist_arr[j];
          min_idx = j + pl_np_counter;
        }
      }

      // registers to store min
      if (pl_np_counter == 0 || min_dist < min_dist_arr[i])  {
        index_local.indices[i] = min_idx;
        min_dist_arr[i] = min_dist;
      }
    }

    // Got rid of modulo to decrease II
    pl_np_counter = (pl_np_counter + N_P_VEC == N_P) ? 0 : pl_np_counter + N_P_VEC;
    if (pl_np_counter == 0) {
      pl_ns_counter = (pl_ns_counter + 1 == NUM_NS) ? 0 : pl_ns_counter + 1;
    }
      
    if (pl_ns_counter == 0 && pl_np_counter == 0) {
      pl_in_width_pos++;
    }
      

    // write to channel after finishing calculations for n_s_vec
    if (pl_np_counter == 0) {

      #ifdef DEBUG
      for (short i = 0; i < N_S_VEC; i++) {
        printf("\nindex: %d\n", (i + index_local.n_s) * IN_W + index_local.in_w);
        printf("\n index: %d\n", index_local.indices[i]);
        test_indices[(i + index_local.n_s*N_S_VEC) * IN_W + index_local.in_w] = index_local.indices[i];
      }
      #endif
      write_channel_intel(indexChannel, index_local);
    }

  } while (pl_in_width_pos < IN_W);
}

__attribute__((max_global_work_dim(0)))
__kernel
void product_lookup (__global const lut_t *restrict lut_pq, 
  short L_S,
  short N_S,
  short N_P,
  short WEIGHT_H,
  short IN_W
  ) 
{
  lut_t __attribute__((numbanks(N_S_VEC*WEIGHT_H_VEC))) lut_pq_temp[N_S_MAX/N_S_VEC * WEIGHT_H_MAX/WEIGHT_H_VEC][N_P_MAX][WEIGHT_H_VEC][N_S_VEC];
  //---------------
  //  load lut 
  //---------------

  short NUM_NS = N_S/N_S_VEC;
  short NUM_WH = WEIGHT_H/WEIGHT_H_VEC;
  for (short i = 0; i < WEIGHT_H/WEIGHT_H_VEC; i++) {
    for (short j = 0; j < WEIGHT_H_VEC; j++) {
      for (short k = 0; k < N_S/N_S_VEC; k++) {
        for (short l = 0; l < N_S_VEC; l++) {
          for (short m = 0; m < N_P; m++) {
            lut_pq_temp[i*NUM_NS + k][m][j][l] = lut_pq[(i*WEIGHT_H_VEC + j)*N_S*N_P + (k*N_S_VEC + l)*N_P + m];
          }
        }
      }
    }
  }

  short lu_weight_h_counter = 0;
  short size = 0;
  nvec_ind_t index_local;
  nvec_out_t out_local;
  nvec_acc_t acc_local;
  bool valid = 0;

  #pragma ii 1
  do{
    
    if (!lu_weight_h_counter) {
      index_local = read_channel_nb_intel(indexChannel, &valid);
    }

    if ((lu_weight_h_counter) || (valid)) {
      #pragma unroll
      for (short i = 0; i < WEIGHT_H_VEC; i ++) {
        
        acc_local.new_col = index_local.n_s == 0 && valid;
        acc_local.last_data = index_local.n_s == NUM_NS-1 && lu_weight_h_counter == WEIGHT_H/WEIGHT_H_VEC - 1 && valid;
        acc_local.weight_h = lu_weight_h_counter;
        
        acc_t acc = 0;
        #pragma unroll      
        for (short j = 0; j < N_S_VEC; j ++) {
          ind_t min_idx = index_local.indices[j];
          acc_local.data[i][j] = lut_pq_temp[lu_weight_h_counter*NUM_NS + index_local.n_s + 0][min_idx][i][j];

        }
        
      }
      
      write_channel_intel(accChannel, acc_local);
      lu_weight_h_counter = (lu_weight_h_counter + 1 >= WEIGHT_H/WEIGHT_H_VEC) ? 0 : lu_weight_h_counter + 1;
      if (lu_weight_h_counter == 0) size+=N_S_VEC;
      
    }
    
  } while(size < IN_W*N_S);

}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
void kernel accumulate() {
  
  nvec_out_t __attribute__((register)) out_local;

  #pragma ii 1
  while(1) {
    
    nvec_acc_t acc_local = read_channel_intel(accChannel);

    #pragma unroll
    for (short i = 0; i < WEIGHT_H_VEC; i++) {
      acc_t acc = 0;

      #pragma unroll
      for (short j = 0; j < N_S_VEC; j++) {
        acc += acc_local.data[i][j];
      }

      out_local.data[acc_local.weight_h][i] = (acc_local.new_col) ? acc : out_local.data[acc_local.weight_h][i] + acc;
    }

    if (acc_local.last_data) {
      write_channel_intel(outChannel, out_local);
    }

  }
  
}


__attribute__((max_global_work_dim(0)))
__kernel
void write_out (__global acc_t * restrict out,
  short WEIGHT_H,
  short IN_W
  )
{
  bool valid;
  short count = 0;
  nvec_out_t out_local;

  #pragma ii 1
  do{  
    out_local = read_channel_nb_intel(outChannel, &valid);
    if (valid) {
      #pragma unroll
      for (short i = 0; i < WEIGHT_H_MAX/WEIGHT_H_VEC; i ++) {
        #pragma unroll
        for (short j = 0; j < WEIGHT_H_VEC; j ++) {
          if(i < WEIGHT_H/WEIGHT_H_VEC) {
            out[count+i*WEIGHT_H_VEC+j] = out_local.data[i][j];
            
          }
          
        }
      }
      count += WEIGHT_H;
    }

  } while(count<IN_W*WEIGHT_H);
}