# PQA

Target device: DE10-Agilex board rev C
Quartus version: Intel Quartus Prime Pro Edition 21.2
OpenCL version: Intel FPGA SDK for OpenCL Pro Edition 21.2
Ubuntu version: Ubuntu 20.04.4 LTS

## Compile for hardware
To compile the design, first modify the design parameters in `hardware/inc/config.h` and `hardware/inc/device.h` to your desired values.

Run the following command: 
`aoc device/pq_basic.cl -o bin/pq_basic.aocx -board=B2E2_8GBx4 -v -clock=500MHz -parallel=16 -I $INTELFPGAOCLSDKROOT/include/kernel_headers`

Note that the compilation will take a few hours. 
