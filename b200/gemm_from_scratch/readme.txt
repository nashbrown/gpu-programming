huge thanks to https://github.com/gau-nernst/learn-cuda/tree/3b90ac9b/02e_matmul_sm100/
for the ptx reference

warmup iters 500
iters 100

cublas baseline
1024: 336 tflops
2048: 1237 tflops
4096: 1374 tflops
8192: 1419 tflops
16384: 1462 tflops

09_persistent_static
1024: 162 tflops
2048: 923 tflops
4096: 1429 tflops
8192: 1303 tflops
16384: 1219 tflops

my 09_persistent_static kernel lags a bit behind because
the thing i'm missing is that the epilogue warpgroup could load from tmem to registers to shared, 
signal to the mma warp that tmem is free after moving data out of tmem, 
and then issue a tma for smem->global. 
whereas currently my epilogue warpgroup signals to the mma warp after it finishes storing to global 
since my original implementation moves data straight from registers to global. 
may add this (and cluster launch control) eventually 
may also experiment with increasing and decreasing registers for specific roles (e.g. epilogue increases registers)

08_cluster_mma
1024: 174 tflops
2048: 930 tflops
4096: 1195 tflops
8192: 1191 tflops
16384: 1055 tflops




