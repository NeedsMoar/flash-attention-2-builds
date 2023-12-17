# flash-attention-2-builds
(Unofficial) Manual builds of wheels for https://github.com/Dao-AILab/flash-attention for Windows x64 and Torch 2.1.1+cu121
I recently tried running [https://github.com/comfyanonymous/ComfyUI](ComfyUI) for a brief while with --pytorch-cross-attention and forgot it was in my startup script, but starting getting OOM issues like crazy on an RTX4090.  Then I realized what was going on and went back to xformers + flash-attention where that hasn't happened yet regardless of what I throw at it.  Batches of 50 latents @ 960x768 followed by an upscale with 6 controlnets, 5 LoRAs, an AnimatedDiff model, the diffusers model I was using, and 2 motion loras loaded?  No problem.
Not having a wheel for it without digging through the comments on the flash attention page is a pain, whereas it builds on my computer in 2 minutes now with 64 jobs and reasonable memory use for a single GPU arch so I can crank these out pretty easily.  

## Important
Currently these builds _only_ include prebuilt code for the Ada Lovelace cards, as well as PTX code which should JIT on Hopper (sm_90), whatever sm_90a is, and future cards.  I figure anyone with a hopper won't be phased by the PTX JIT time.   Flash attention doesn't build fp8 kernels yet.  If I manage to get TransformerEngine to build on Windows, that'll be here too.  Currently there are wheels for python 3.10 and 3.11 (from the python website).  I plan on building wheels with support for sm_80 (Volta) and sm_86 (Ampere) in a short while, also with PTX kernels for sm_80.  The binary for sm_87 isn't being included because it's one of those bizarro Jetson type devices and unlike sm_80 wasn't a default in flash-attention's original build.

## Installing 
Download the wheel appropriate to your python version and CUDA version from releases, and run:
```cmd
pip install -U xformers<sup>1</sup>
pip install flash_attn-2.3.6-cp310-cp310-win_amd64-cuda12.3-sm89.whl or flash_attn-2.3.6-cp311-cp311-win_amd64-cuda12.3-sm89.whl
```
1. If you don't have it installed already, or just have an old version.  

## Verifying
Afterwards you can run 
```cmd
python -m xformers.info
```
Which should give you something like:
```cmd
E:\code\flash-attention>python -m xformers.info
A matching Triton is not available, some optimizations will not be enabled.
Error caught was: No module named 'triton'
xFormers 0.0.23
memory_efficient_attention.cutlassF:               available
memory_efficient_attention.cutlassB:               available
memory_efficient_attention.decoderF:               available
memory_efficient_attention.flshattF@v2.3.6:        available
memory_efficient_attention.flshattB@v2.3.6:        available
memory_efficient_attention.smallkF:                available
memory_efficient_attention.smallkB:                available
memory_efficient_attention.tritonflashattF:        unavailable
memory_efficient_attention.tritonflashattB:        unavailable
memory_efficient_attention.triton_splitKF:         unavailable
indexing.scaled_index_addF:                        unavailable
indexing.scaled_index_addB:                        unavailable
indexing.index_select:                             unavailable
swiglu.dual_gemm_silu:                             available
swiglu.gemm_fused_operand_sum:                     available
swiglu.fused.p.cpp:                                available
is_triton_available:                               False
pytorch.version:                                   2.1.1+cu121
pytorch.cuda:                                      available
gpu.compute_capability:                            8.9
gpu.name:                                          NVIDIA GeForce RTX 4090
build.info:                                        available
build.cuda_version:                                1201
build.python_version:                              3.10.11
build.torch_version:                               2.1.1+cu121
build.env.TORCH_CUDA_ARCH_LIST:                    5.0+PTX 6.0 6.1 7.0 7.5 8.0+PTX 9.0
build.env.XFORMERS_BUILD_TYPE:                     Release
build.env.XFORMERS_ENABLE_DEBUG_ASSERTIONS:        None
build.env.NVCC_FLAGS:                              None
build.env.XFORMERS_PACKAGE_FROM:                   wheel-v0.0.23
build.nvcc_version:                                12.1.66
source.privacy:                                    open source
```

To make sure that the flash attention kernels were picked up correctly by xformers.  The order of installation of the two doesn't matter in my experience.   The missing attention types are triton related;  afaik there's no Windows build for it and since it's an entire compiler / optimizer system I don't even want to try porting it.  

## Lazy Loading
These are built with MSVC 2022 v17.8.2 / CL.exe 19.38.33130 against the CUDA 12.3 toolchain, which is compatible with the Pytorch release and supports lazy loading officially on Windows;  
```cmd
set CUDA_MODULE_LOADING=lazy
```
To enable that, FWIW.  Diffusers generally loads so fast for me (or the models are in standby memory) that Lazy loading hasn't made much of a difference, but it's primarily intended to speed up PTX libraries and only JIT functions as they're needed I believe.  
