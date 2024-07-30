# QAOA Simulator

### Preliminary Setup

1. This repository uses AMD's AOCL to perform the AVX sincos function
    For AMD devices, you need to install the AOCL library first.  
    You can find it here: [https://www.amd.com/zh-tw/developer/aocl.html]()

2. After installation, modify Line 320 in [CMakeLists.txt](./QuEST/CMakeLists.txt) to set `AOCL_PATH` to your AOCL installation path.

### Getting started

1. Clone this repository and navigate to the corresponding directory.

2. Create the `build` directory and navigate to it.

    ```bash
    mkdir build
    cd build
    ```

3. Configure the build environment.

    ```bash
    # On CPU, no AVX
    cmake .. -DUSER_SOURCE=<target> -DPRECISION=1
    # On CPU, with AVX
    cmake .. -DUSER_SOURCE=<target> -DPRECISION=1 -DVECTORIZED
    # On GPU
    cmake .. -DUSER_SOURCE=<target> -DGPUACCELERATED=1 -DGPU_COMPUTE_CAPABILITY=<CC>
    ```

    where 
    * `<target>` refer to the source file to compile (`quest.c`, `weighted.c`, `unweighted.c`)
    * `<CC>` refer to the compute capability of your cuda device. You can find your device's compute capability on the this [website](https://developer.nvidia.com/cuda-gpus).

4. Build 

    ```bash
    make
    ```

5. Execute `./demo`. The output will show one of the stateVector and the eclapsed time for each qubits.


### Acknowledgements

The following repos are used in our simulator.

[QuEST Simulator](https://github.com/QuEST-Kit/QuEST): The base of our simulator, with version 3.5.0.

[sse-popcount](https://github.com/WojciechMula/sse-popcount/blob/master/popcnt-avx2-lookup.cpp): The method to implement AVX popcount.

[avx_mathfun](https://github.com/reyoung/avx_mathfun/blob/master/avx_mathfun.h): The implementation of AVX sincos function.