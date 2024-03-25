#include <math.h>
#include "golden_model.h"
#include <cstdio>

int get_index(proto_t* in, proto_t* proto_table, int ns, int in_w, int L_S, int IN_W, int N_P) {
  // get first element
  dist_t min = 0;
  ind_t index = 0;
  for (int ls = 0; ls < L_S; ls ++) {
    dist_t element = in[ns*IN_W*L_S + in_w*L_S + ls] - proto_table[0*L_S + ls];
    #ifdef L1
    element = abs(element);
    #else 
    element = element * element;
    #endif
    min += element;
  }
  for (int np = 1; np < N_P; np ++) {
    dist_t comparator = 0;
    for (int ls = 0; ls < L_S; ls ++) {
      dist_t element = in[ns*IN_W*L_S + in_w*L_S + ls] - proto_table[np*L_S + ls];
      #ifdef L1
      element = abs(element);
      #else 
      element = element * element;
      #endif
      comparator += element;
    }
    if (comparator < min) {
      min = comparator;
      index = np;
    }
  }
  
  return index;
}

bool equal_flt(float correct, float test) {
  return (abs(test-correct) < correct / FLT_ERROR_FACTOR);
}

int validate(proto_t* in, proto_t* proto_table, lut_t* lut_pq, acc_t* out, ind_t* ind, int WEIGHT_H, 
              int IN_W, int N_S, int L_S, int N_P, bool test_ind) {
  
  int out_size = WEIGHT_H*IN_W;
  int ind_size = N_S*IN_W;
  acc_t* golden_out = (acc_t*) malloc(out_size*sizeof(acc_t));
  ind_t* golden_ind = (ind_t*) malloc(ind_size*sizeof(ind_t));

  #ifdef DEBUG
  printf("\n EXPECTED OUTPUT: \n");
  #endif

  acc_t element;
  // for each subsection in input column
  for (int weight_h = 0; weight_h < WEIGHT_H; weight_h ++) {
    for (int in_w = 0; in_w < IN_W; in_w ++ ) {
      element = 0;
      // calculate element
      for (int ns = 0; ns < N_S; ns ++) {
        int index = get_index(in, proto_table, ns, in_w, L_S, IN_W, N_P);
        golden_ind[ns * IN_W + in_w] = index;
        element += (acc_t) lut_pq[weight_h*N_S*N_P + ns*N_P + index];
      }
      golden_out[weight_h * IN_W + in_w] = element;
  #ifdef DEBUG
        printf("%d \t", element);
  #endif

    }
      #ifdef DEBUG
    printf("\n");
      #endif

  }

  #ifdef DEBUG
  printf("\n");

  for (int i = 0; i < ind_size; i ++) {
    printf("golden ind: %d\t", golden_ind[i]);
    if ((i+1)%(IN_W) == 0) printf("\n");
  }
  printf("\n");

  for (int i = 0; i < ind_size; i ++) {
    printf("result ind: %d\t", ind[i]);
    if ((i+1)%(IN_W) == 0) printf("\n");
  } 
  
  printf("\n");
  for (int i = 0; i < out_size; i ++) {
    printf("golden out: %d  ", golden_out[i]);
    if ((i+1)%(IN_W) == 0) printf("\n");
  }
 
  printf("\n");

  for (int i = 0; i < out_size; i ++) {
    printf("result out: %d  ", out[i]);
    if ((i+1)%(IN_W) == 0) printf("\n");
  }
  #endif

  if (test_ind) {
    for (int i = 0; i < ind_size; i ++) {
      if (golden_ind[i] != ind[i]) {
        printf("\nExpected index output: %d\n", golden_ind[i]);
        return i;
      }
    }
  } else {
    for (int i = 0; i < out_size; i++) {
      // if (!equal_flt(golden_out[i], out[i])) {
      if (golden_out[i] != out[i]) {
        printf("\nExpected output: %d\n", golden_out[i]);
        return i;
      }
    }
  }


  return -1;
}