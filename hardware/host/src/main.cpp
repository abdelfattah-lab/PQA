#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include <bits/stdc++.h>

#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "host.h"

using namespace aocl_utils;

#define STRING_BUFFER_LEN 1024
#define ACL_ALIGNMENT 64
#define NUM_KERNELS_TO_CREATE 3
#define NUM_QUEUES_TO_CREATE 3
#define NUM_QUEUES_TO_FINISH 3
#define NUM_INPUT_BUFFERS 3
#define KID_DIST_CALC 0
#define KID_PRODUCT_LOOKUP 1
#define KID_WRITE_OUT 2
#define KID_WRITE_IN 3
#define KID_READ_OUT 4
#define KID_READ_IND 5

#include "golden_model.h"

// Runtime constants
// Used to define the work set over which this kernel will execute.
static const size_t work_group_size = 1;  // 8 threads in the demo workgroup
// Defines kernel argument value, which is the workitem ID that will
// execute a printf call
static const int thread_id_to_output = 2;

// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;

cl_kernel kernel[NUM_KERNELS_TO_CREATE];
// extra queues for writing input and reading output and index buffers
#ifdef DEBUG
cl_command_queue cmdQueue[NUM_QUEUES_TO_CREATE+3]; 
#else
cl_command_queue cmdQueue[NUM_QUEUES_TO_CREATE+2];
#endif
cl_event kernel_exec_event[NUM_QUEUES_TO_CREATE];
cl_event kernel_write_event[NUM_INPUT_BUFFERS];


static cl_program program = NULL;

// Control whether the emulator should be used.
static bool use_emulator = false;

cl_mem d_in          = NULL;
cl_mem d_proto_table = NULL;
cl_mem d_lut_pq      = NULL;
cl_mem d_out         = NULL;
cl_mem d_ind         = NULL;

proto_t* in            = NULL;
proto_t* proto_table   = NULL;
lut_t* lut_pq        = NULL;
acc_t* out           = NULL;
proto_t* weight        = NULL;
ind_t* ind             = NULL;

short L_S = 0;
short N_S = 0;
short N_P = 0;
short WEIGHT_H = 0;
short IN_W = 0;

int in_size = 0;
int proto_table_size = 0;
int lut_pq_size = 0;
int out_size = 0;
int weight_size = 0;
int ind_size = 0;

const char *kernel_name[] = {
    "distance_calc",
    "product_lookup",
    "write_out",
};
const char* buffer_name[] = {
  "in",
  "prototable",
  "lutpq"
};


// Function prototypes
bool init();
void cleanup();
bool data_init();
void profile_buffer_write();
void profile_kernel();
void enqueue_kernel();
void validate_output();
double compute_kernel_execution_time(cl_event &event, double &start_d, double &end_d);
static void device_info_ulong( cl_device_id device, cl_device_info param, const char* name);
static void device_info_uint( cl_device_id device, cl_device_info param, const char* name);
static void device_info_bool( cl_device_id device, cl_device_info param, const char* name);
static void device_info_string( cl_device_id device, cl_device_info param, const char* name);
static void display_device_info( cl_device_id device );


void* acl_aligned_malloc (size_t size) {
  void *result = NULL;
  if (posix_memalign(&result, ACL_ALIGNMENT, size) != 0)
      printf("acl_aligned_malloc() failed.\n");
  return result;
}
void acl_aligned_free (void *ptr) {
  free (ptr);
}

// Entry point.
int main(int argc, char **argv) {
  Options options(argc, argv);

  // Optional argument to specify whether the emulator should be used.
  if(options.has("emulator")) {
    use_emulator = options.get<bool>("emulator");
  }

  cl_int status;

  if(!init()) {
    return -1;
  }

  if (!data_init()) {
    return -1;
  }

  //----------------------------------------------
  // Create device buffers
  //----------------------------------------------
  #ifdef DEV_INFO
  printf("\n===== Host-CPU transferring matrices A,B to the FPGA device global memory (DDR4) via PCIe ======\n\n");
  #endif 

  int in_size_max = N_S_MAX * IN_W_MAX * L_S_MAX;
  int proto_table_size_max = N_P_MAX * L_S_MAX;
  int lut_pq_size_max = WEIGHT_H_MAX * N_S_MAX * N_P_MAX;
  int out_size_max = WEIGHT_H_MAX * IN_W_MAX;
  int weight_size_max = WEIGHT_H_MAX * N_S_MAX * L_S_MAX;

  d_in = clCreateBuffer(
          context,
          CL_MEM_READ_ONLY,
          in_size_max*sizeof(proto_t),
          NULL,
          &status); 
  checkError(status, "Failed to create device buffer");
          
  d_proto_table = clCreateBuffer(
          context,
          CL_MEM_READ_ONLY,
          proto_table_size_max*sizeof(proto_t),
          NULL,
          &status); 
  checkError(status, "Failed to create device buffer");

  d_lut_pq = clCreateBuffer(
          context,
          CL_MEM_READ_ONLY,
          lut_pq_size_max*sizeof(lut_t),
          NULL,
          &status); 
  checkError(status, "Failed to create device buffer");

  d_out = clCreateBuffer(
          context,
          CL_MEM_WRITE_ONLY,
          out_size_max*sizeof(acc_t),
          NULL,
          &status); 
  checkError(status, "Failed to create device buffer");

  #ifdef DEBUG
  d_ind = clCreateBuffer(
          context,
          CL_MEM_WRITE_ONLY,
          ind_size*sizeof(ind_t),
          NULL,
          &status); 
  checkError(status, "Failed to create device buffer");
  #endif

  //----------------------------------------------
  // Write host data to device buffers
  //----------------------------------------------
  // blocking writes
  status = clEnqueueWriteBuffer(cmdQueue[KID_WRITE_IN], d_in, CL_TRUE, 0, sizeof(proto_t)*in_size, in, 0, NULL, &kernel_write_event[0]);
	checkError(status, "WriteBuffer");
	status = clEnqueueWriteBuffer(cmdQueue[KID_DIST_CALC], d_proto_table, CL_TRUE, 0, sizeof(proto_t)*proto_table_size, proto_table, 0, NULL, &kernel_write_event[1]);
	checkError(status, "WriteBuffer 2");
  status = clEnqueueWriteBuffer(cmdQueue[KID_PRODUCT_LOOKUP], d_lut_pq, CL_TRUE, 0, sizeof(lut_t)*lut_pq_size, lut_pq, 0, NULL, &kernel_write_event[2]);
	checkError(status, "WriteBuffer 3");

  //----------------------------------------------
  // Set the kernel argument
  //----------------------------------------------

  // DIST CAL KERNEL ARGS //

  status = clSetKernelArg(kernel[KID_DIST_CALC], 0, sizeof(cl_mem), (void*)&d_in);
  checkError(status, "Failed to set kernel arg 0");

  status = clSetKernelArg(kernel[KID_DIST_CALC], 1, sizeof(cl_mem), (void*)&d_proto_table);
  checkError(status, "Failed to set kernel arg 1");

  status = clSetKernelArg(kernel[KID_DIST_CALC], 2, sizeof(cl_short), (void*)&L_S);
  checkError(status, "Failed to set kernel arg 4");

  status = clSetKernelArg(kernel[KID_DIST_CALC], 3, sizeof(cl_short), (void*)&N_S);
  checkError(status, "Failed to set kernel arg 5");

  status = clSetKernelArg(kernel[KID_DIST_CALC], 4, sizeof(cl_short), (void*)&N_P);
  checkError(status, "Failed to set kernel arg 6");

  status = clSetKernelArg(kernel[KID_DIST_CALC], 5, sizeof(cl_short), (void*)&WEIGHT_H);
  checkError(status, "Failed to set kernel arg 7");

  status = clSetKernelArg(kernel[KID_DIST_CALC], 6, sizeof(cl_short), (void*)&IN_W);
  checkError(status, "Failed to set kernel arg 8");

  #ifdef DEBUG
  status = clSetKernelArg(kernel[KID_DIST_CALC], 7, sizeof(cl_mem), (void*)&d_ind);
  checkError(status, "Failed to set kernel arg 2");
  #endif

  // PRODUCT LOOKUP KERNEL ARGS //

  status = clSetKernelArg(kernel[KID_PRODUCT_LOOKUP], 0, sizeof(cl_mem), (void*)&d_lut_pq);
  checkError(status, "Failed to set kernel arg 1");

  status = clSetKernelArg(kernel[KID_PRODUCT_LOOKUP], 1, sizeof(cl_short), (void*)&L_S);
  checkError(status, "Failed to set kernel arg 3");

  status = clSetKernelArg(kernel[KID_PRODUCT_LOOKUP], 2, sizeof(cl_short), (void*)&N_S);
  checkError(status, "Failed to set kernel arg 4");

  status = clSetKernelArg(kernel[KID_PRODUCT_LOOKUP], 3, sizeof(cl_short), (void*)&N_P);
  checkError(status, "Failed to set kernel arg 5");

  status = clSetKernelArg(kernel[KID_PRODUCT_LOOKUP], 4, sizeof(cl_short), (void*)&WEIGHT_H);
  checkError(status, "Failed to set kernel arg 6");

  status = clSetKernelArg(kernel[KID_PRODUCT_LOOKUP], 5, sizeof(cl_short), (void*)&IN_W);
  checkError(status, "Failed to set kernel arg 7");

  // WRITE OUT KERNEL ARGS //

  status = clSetKernelArg(kernel[KID_WRITE_OUT], 0, sizeof(cl_mem), (void*)&d_out);
  checkError(status, "Failed to set kernel arg 1");

  status = clSetKernelArg(kernel[KID_WRITE_OUT], 1, sizeof(cl_short), (void*)&WEIGHT_H);
  checkError(status, "Failed to set kernel arg 6");

  status = clSetKernelArg(kernel[KID_WRITE_OUT], 2, sizeof(cl_short), (void*)&IN_W);
  checkError(status, "Failed to set kernel arg 7");

  #ifdef DEV_INFO
  printf("\nKernel initialization is complete.\n");
  printf("Launching the kernel...\n\n");
  #endif

  // Configure work set over which the kernel will execute
  size_t wgSize[3] = {work_group_size, 1, 1};
  size_t gSize[3] = {work_group_size, 1, 1};

  //----------------------------------------------
  // Launch the kernels
  //----------------------------------------------

  status = clEnqueueNDRangeKernel(
            cmdQueue[KID_DIST_CALC],
            kernel[KID_DIST_CALC],
            1,
            NULL,
            gSize,
            wgSize,
            2,
            kernel_write_event,
            &kernel_exec_event[KID_DIST_CALC]
            );
  checkError(status, "Failed to launch kernel");

  status = clEnqueueNDRangeKernel(
              cmdQueue[KID_PRODUCT_LOOKUP],
              kernel[KID_PRODUCT_LOOKUP],
              1,
              NULL,
              gSize,
              wgSize,
              1,
              &kernel_write_event[2],
              &kernel_exec_event[KID_PRODUCT_LOOKUP]
              );
  checkError(status, "Failed to launch kernel");

  status = clEnqueueNDRangeKernel(
              cmdQueue[KID_WRITE_OUT],
              kernel[KID_WRITE_OUT],
              1,
              NULL,
              gSize,
              wgSize,
              0,
              NULL,
              &kernel_exec_event[KID_WRITE_OUT]
              );
  checkError(status, "Failed to launch kernel");


  // Wait for command queue to complete pending events
  for(int i=NUM_KERNELS_TO_CREATE; i <= 0; i--) {
    status = clFlush(cmdQueue[i]);
    checkError(status, "Failed to flush");
  }

  for(int i=NUM_QUEUES_TO_FINISH-1; i >= 0; i--) {
    status = clFinish(cmdQueue[i]); 
    checkError(status, "Failed to finish");
  }
  
  //----------------------------------------------
  // Read and validate output
  //----------------------------------------------
  validate_output();

  //----------------------------------------------
  // Perform profiling
  //----------------------------------------------
  #ifdef PROFILE
  printf("\n===== Reporting measured throughput ======\n\n");
  profile_buffer_write();
  profile_kernel();
  #endif

  #ifdef DEV_INFO
  printf("\nKernel execution is complete.\n");
  #endif

  // Free the resources allocated
  cleanup();

  return 0;
}


/////// HELPER FUNCTIONS ///////

void validate_output() {
  cl_int status;
  #ifdef DEBUG
  status=clEnqueueReadBuffer(cmdQueue[KID_READ_IND], d_ind, CL_TRUE, 0, sizeof(ind_t)*ind_size, ind, 0, NULL, NULL);
	checkError(status, "Read Buffer");

  #endif

	status=clEnqueueReadBuffer(cmdQueue[KID_READ_OUT], d_out, CL_TRUE, 0, sizeof(acc_t)*out_size, out, 0, NULL, NULL);
	checkError(status, "Read Buffer");
  
  acc_t* transposed_out = (acc_t*) malloc(out_size*sizeof(acc_t));

  for (int i = 0; i < WEIGHT_H; i++) {
    for (int j = 0; j < IN_W; j++) {
      transposed_out[i*IN_W + j] = out[i+j*WEIGHT_H];
    }
  }

  int error_index = validate(in, proto_table, lut_pq, transposed_out, ind, WEIGHT_H, IN_W, N_S, L_S, N_P, 0) ;
  if (error_index != -1) {
    printf("INCORRECT OUT AT INDEX %d: %d\n", error_index, out[error_index]);
  } else {
    printf("\n CORRECT OUTPUT. \n\n");
  }
}

void enqueue_kernel() {
  cl_int status;
  size_t wgSize[3] = {work_group_size, 1, 1};
  size_t gSize[3] = {work_group_size, 1, 1};

  for (int i = 0; i < NUM_KERNELS_TO_CREATE; i++) {
    // printf("clEnqueueNDRangeKernel[%d]: %s!\n", i,kernel_name[i]);
    status = clEnqueueNDRangeKernel(
              cmdQueue[i],
              kernel[i],
              1,
              NULL,
              gSize,
              wgSize,
              0,
              NULL,
              &kernel_exec_event[i]
              );
    checkError(status, "Failed to launch kernel");
  }

  // Wait for command queue to complete pending events
  for(int i=NUM_KERNELS_TO_CREATE; i <= 0; i--) {
    status = clFlush(cmdQueue[i]);
    checkError(status, "Failed to flush");
  }

  for(int i=NUM_QUEUES_TO_FINISH-1; i >= 0; i--) {
    status = clFinish(cmdQueue[i]); 
    checkError(status, "Failed to finish");
  }
}


void profile_buffer_write() {
  // calculate the time
  double k_write_start_time[NUM_QUEUES_TO_FINISH];
  double k_write_end_time[NUM_QUEUES_TO_FINISH];
  double k_write_exec_time[NUM_QUEUES_TO_FINISH];

  for (int i=0; i<NUM_QUEUES_TO_FINISH; i++) {
      k_write_exec_time[i] = compute_kernel_execution_time(kernel_write_event[i], k_write_start_time[i], k_write_end_time[i]);
  }

  for(int i=0; i<NUM_INPUT_BUFFERS; i++) {
    printf("  Kernel execution time on FPGA: %s,\n\t\t\t\t\t\t\t\texec time = %.5f s, start=%.5f s, end=%.5f s\n", buffer_name[i], k_write_exec_time[i], k_write_start_time[i], k_write_end_time[i]);
  }
}

void profile_kernel() {
  // calculate the time
  double k_start_time[NUM_QUEUES_TO_FINISH];
  double k_end_time[NUM_QUEUES_TO_FINISH];
  double k_exec_time[NUM_QUEUES_TO_FINISH];

  for (int i=0; i<NUM_QUEUES_TO_FINISH; i++) {
      k_exec_time[i] = compute_kernel_execution_time(kernel_exec_event[i], k_start_time[i], k_end_time[i]);
  }
  printf("\n\n");
  
  double k_earliest_start_time = k_start_time[KID_DIST_CALC];
  double k_latest_end_time     = k_end_time[KID_WRITE_OUT];

  printf("N_S: %d      N_P: %d      L_S: %d      WEIGHT_H: %d      IN_W: %d\n",N_S,N_P,L_S,WEIGHT_H,IN_W);
  printf("N_S_VEC: %d  N_P_VEC: %d   L_S_VEC: %d  WEIGHT_H_VEC: %d\n",N_S_VEC,N_P_VEC,L_S_VEC,WEIGHT_H_VEC);
  printf("N_S_MAX: %d  N_P_MAX: %d  L_S_MAX: %d  WEIGHT_H_MAX: %d  IN_W_MAX: %d\n\n",N_S_MAX,N_P_MAX,L_S_MAX,WEIGHT_H_MAX,IN_W_MAX);


  for(int i=0; i<NUM_QUEUES_TO_FINISH; i++) {
    printf("  Kernel execution time on FPGA: %s,\n\t\t\t\t\t\t\t\texec time = %.5f s, start=%.5f s, end=%.5f s\n", kernel_name[i], k_exec_time[i], k_start_time[i], k_end_time[i]);
  }

  double k_overall_exec_time = k_latest_end_time - k_earliest_start_time;

  printf("\n");
  printf("  Loader kernels start time\t\t= %.5f s\n", k_earliest_start_time);
  printf("  Drainer kernels end time\t\t= %.5f s\n", k_latest_end_time);
  printf("  FPGA PQ exec time\t\t= %.5f s\n", k_overall_exec_time);

  // multiplied by 1.0e-9 to get G-FLOPs
  printf("\n");

  // L_S operations per prototype, repeated over each subspace, x2 for + and * operations
  double num_dist_calc_operations = (double)(2.0)* L_S * N_P * N_S * IN_W;
  // WEIGHT_H accumulates per out column, performed (N_S/N_S_VEC - 1) times for IN_W columns
  double num_acc_operations = (double) WEIGHT_H * (N_S/N_S_VEC - 1) * IN_W;
  double num_operations = num_dist_calc_operations + num_acc_operations;

  printf("  # operations = %.0f\n", num_operations );
  printf("  Throughput: %.5f GOPS\n", (double)1.0e-9 * num_operations / k_overall_exec_time);  
}


double compute_kernel_execution_time(cl_event &event, double &start_d, double &end_d)
{
    cl_ulong start, end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,      sizeof(cl_ulong), &end,     NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,    sizeof(cl_ulong), &start,   NULL);

    start_d = (double)1.0e-9 * start;
    end_d   = (double)1.0e-9 * end;

    return    (double)1.0e-9 * (end - start); // nanoseconds to seconds
}


bool init() {
  cl_int status;

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  if (use_emulator) {
    platform = findPlatform("Intel(R) FPGA Emulation Platform for OpenCL(TM)");
  } else {
    platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
  }
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
    return false;
  }

  // User-visible output - Platform information
  {
    char char_buffer[STRING_BUFFER_LEN]; 

    #ifdef DEV_INFO
    printf("Querying platform for info:\n");
    printf("==========================\n");
    #endif

    clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
    #ifdef DEV_INFO
    printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
    #endif
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
    #ifdef DEV_INFO
    printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
    #endif
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
    #ifdef DEV_INFO
    printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);
    #endif
  }

  // Query the available OpenCL devices.
  scoped_array<cl_device_id> devices;
  cl_uint num_devices;

  devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));

  // We'll just use the first device.
  device = devices[0];

  // Display some device information.
  #ifdef DEV_INFO
  display_device_info(device);
  #endif

  // Create the context.
  context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the command queue.
  for(int i=0; i<NUM_QUEUES_TO_CREATE; i++) {
    // fprintf(stdout,"cmdQueue i = %d, kernel name = %s\n", i, kernel_name[i]);
    cmdQueue[i] = clCreateCommandQueue(
            context,
            device,
            CL_QUEUE_PROFILING_ENABLE,
            &status);
            checkError(status, "Failed to create command queue");
  }

  // fprintf(stdout,"cmdQueue i = %d, a queue for reading the out buffer\n", NUM_QUEUES_TO_CREATE);
  cmdQueue[KID_READ_OUT] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status); 
  checkError(status, "Failed to create command queue");

  // fprintf(stdout,"cmdQueue i = %d, a queue for writing the in buffe\n", NUM_QUEUES_TO_CREATE);
  cmdQueue[KID_WRITE_IN] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status); 
  checkError(status, "Failed to create command queue");

  #ifdef DEBUG
  cmdQueue[KID_READ_IND] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status); 
  checkError(status, "Failed to create command queue");
  #endif

  // Create the program.
  std::string binary_file = getBoardBinaryFile("pq_basic", device);
  #ifdef DEV_INFO
  printf("Using AOCX: %s\n", binary_file.c_str());
  #endif
  program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);


  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  for(int j=0; j<NUM_KERNELS_TO_CREATE; j++) {
    // printf("Creating kernel[%d]: %s\n", j,kernel_name[j]);
    kernel[j] = clCreateKernel(program, (const char*)kernel_name[j], &status);
    checkError(status, "Failed to create kernel");
  }

  return true;
}

bool data_init() {
  
  FILE *fpIn = fopen("/home/ayc62/Documents/PQA/hardware/inc/in.bin", "rb");
  FILE *fpPrototable = fopen("/home/ayc62/Documents/PQA/hardware/inc/proto_table.bin", "rb");
  FILE *fpLutpq = fopen("/home/ayc62/Documents/PQA/hardware/inc/lut_pq.bin", "rb");
  FILE *fpSize = fopen("/home/ayc62/Documents/PQA/hardware/inc/size.bin", "rb");
  
  size_t result; 
  int size[5];
  result = fread(size, sizeof(int), 5, fpSize);
  

  N_S = size[0];
  N_P = size[1];
  L_S = size[2];
  WEIGHT_H = size[3];
  IN_W = size[4];
  
  in_size = N_S * IN_W * L_S;
  proto_table_size = N_P * L_S;
  lut_pq_size = WEIGHT_H * N_S * N_P;
  out_size = WEIGHT_H * IN_W;
  weight_size = WEIGHT_H * N_S * L_S;
  ind_size = N_S * IN_W;

  assert((L_S == L_S_VEC));

  if((in = (proto_t*)acl_aligned_malloc(in_size*sizeof(proto_t))) == NULL) {
    perror("Failed malloc of in");
    exit(1);
  }
  if((proto_table = (proto_t*)acl_aligned_malloc(proto_table_size*sizeof(proto_t))) == NULL) {
    perror("Failed malloc of proto_table");
    exit(1);
  }

  if((lut_pq = (lut_t*)acl_aligned_malloc(lut_pq_size*sizeof(lut_t))) == NULL) {
    perror("Failed malloc of proto_table");
    exit(1);
  }

  if((out = (acc_t*)acl_aligned_malloc(out_size*sizeof(acc_t))) == NULL) {
    perror("Failed malloc of proto_table");
    exit(1);
  }

  if((ind = (ind_t*)acl_aligned_malloc(ind_size*sizeof(ind_t))) == NULL) {
    perror("Failed malloc of proto_table");
    exit(1);
  }

  if((weight = (proto_t*)acl_aligned_malloc(weight_size*sizeof(proto_t))) == NULL) {
    perror("Failed malloc of proto_table");
    exit(1);
  }
  result = fread(in, sizeof(proto_t), in_size, fpIn);
  result = fread(proto_table, sizeof(proto_t), proto_table_size, fpPrototable);
  result = fread(lut_pq, sizeof(lut_t), lut_pq_size, fpLutpq);

  

  return true;

}



// Free the resources allocated during initialization
void cleanup() {

  if(program) {
    clReleaseProgram(program);
  }

  for(int i=0; i<NUM_KERNELS_TO_CREATE; i++) {
    clReleaseKernel(kernel[i]);
  }

  for(int i=0; i<NUM_QUEUES_TO_CREATE; i++) {
    clReleaseCommandQueue(cmdQueue[i]);
  }

  for(int i=0; i<NUM_QUEUES_TO_FINISH; i++) {
    clReleaseEvent(kernel_exec_event[i]);
  }

  for(int i=0; i<NUM_INPUT_BUFFERS; i++) {
    clReleaseEvent(kernel_write_event[i]);
  }

  if(context) {
    clReleaseContext(context);
  }

  acl_aligned_free(in);
  acl_aligned_free(proto_table);
  acl_aligned_free(lut_pq);
  acl_aligned_free(out);
  acl_aligned_free(weight);
  acl_aligned_free(ind);
}

// Helper functions to display parameters returned by OpenCL queries
static void device_info_ulong( cl_device_id device, cl_device_info param, const char* name) {
   cl_ulong a;
   clGetDeviceInfo(device, param, sizeof(cl_ulong), &a, NULL);
   #ifdef DEV_INFO 
   printf("%-40s = %llu\n", name, (unsigned long long)a);
   #endif
}
static void device_info_uint( cl_device_id device, cl_device_info param, const char* name) {
   cl_uint a;
   clGetDeviceInfo(device, param, sizeof(cl_uint), &a, NULL);
   #ifdef DEV_INFO 
   printf("%-40s = %u\n", name, a);
   #endif
}
static void device_info_bool( cl_device_id device, cl_device_info param, const char* name) {
   cl_bool a;
   clGetDeviceInfo(device, param, sizeof(cl_bool), &a, NULL);
   #ifdef DEV_INFO 
   printf("%-40s = %s\n", name, (a?"true":"false"));
   #endif
}
static void device_info_string( cl_device_id device, cl_device_info param, const char* name) {
   char a[STRING_BUFFER_LEN]; 
   clGetDeviceInfo(device, param, STRING_BUFFER_LEN, &a, NULL);
   #ifdef DEV_INFO 
   printf("%-40s = %s\n", name, a);
   #endif
}

// Query and display OpenCL information on device and runtime environment
static void display_device_info( cl_device_id device ) {

   printf("Querying device for info:\n");
   printf("========================\n");
   device_info_string(device, CL_DEVICE_NAME, "CL_DEVICE_NAME");
   device_info_string(device, CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
   device_info_uint(device, CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID");
   device_info_string(device, CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
   device_info_string(device, CL_DRIVER_VERSION, "CL_DRIVER_VERSION");
   device_info_uint(device, CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS");
   device_info_bool(device, CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
   device_info_bool(device, CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE");
   device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");
   device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE");
   device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
   device_info_bool(device, CL_DEVICE_IMAGE_SUPPORT, "CL_DEVICE_IMAGE_SUPPORT");
   device_info_ulong(device, CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");
   device_info_ulong(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY");
   device_info_ulong(device, CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
   device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");
   device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");
   device_info_uint(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
   device_info_uint(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MEM_BASE_ADDR_ALIGN");
   device_info_uint(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE");

   {
      cl_command_queue_properties ccp;
      clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &ccp, NULL);
      printf("%-40s = %s\n", "Command queue out of order? ", ((ccp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)?"true":"false"));
      printf("%-40s = %s\n", "Command queue profiling enabled? ", ((ccp & CL_QUEUE_PROFILING_ENABLE)?"true":"false"));
   }
}

