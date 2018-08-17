# test nvjpeg resize

This repo is main from NVIDIA [official website](https://developer.nvidia.com/nvjpeg-release-candidate-download)

File nvjpeg_npp.cpp/nvJpegnpp.hpp are copied from nvJpeg_example.cpp/nvJpeg_exmaple.hxx and modified for new features :
- 1.Resize imamges use Nppresize and write to specific path
- 2.Use cudastream for Parallelism


## Compile

```
g++ -O3 -m64 nvjpeg_npp.cpp -I../include -lnvjpeg -L../lib64 -I/usr/local/cuda-9.0/include -lcudart -lnppig -lnppisu -lnppc -L/usr/local/cuda-9.0/lib64  -I/usr/cv/include -L/usr/cv/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -Wl,-rpath=../lib64 -Wl,-rpath=/usr/local/cuda-9.0/lib64 -o nvjpeg_npp
```

## Use

Note: **Only support RGBI format to write image**
```
nvJpeg_npp -i /input/ -o /output/ -fmt bgri -batched -pipelined -b 12
```
Run the command `./nvjpeg_example -h `for the description of the parameters



