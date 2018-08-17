#include "cuda_runtime.h"
#include "nvjpeg.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

#include <npp.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

#include <sys/time.h> // timings
#include <string.h> // strcmpi


#include <dirent.h>  // linux dir traverse
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace cv;
using namespace std;

int cudaCheckError_str(const char* debugstr)
 {                                              
  cudaError_t e=cudaGetLastError();                                     
  if(e!=cudaSuccess) {                                                  
    std::cout << "Cuda failure: '"<<debugstr<<": " << cudaGetErrorString(e) << "'\n";   
    return 1;                                                           
  }                                                                     
}

#define cudaCheckError() {                                              \
  cudaError_t e=cudaGetLastError();                                     \
  if(e!=cudaSuccess) {                                                  \
    std::cout << "Cuda failure: " << cudaGetErrorString(e) << "'\n";   \
    return 1;                                                           \
  }                                                                     \
}


#define nvjpegCheckError(call) {                                                    \
    nvjpegStatus_t e = (call);                                                      \
    if(e != NVJPEG_STATUS_SUCCESS) {                                                \
        std::cout << "nvjpeg failure: error #" << e << "\n";                        \
        return 1;                                                                   \
    }                                                                               \
}


int readInput(const std::string &sInputPath, std::vector<std::string> &filelist)
{
    int error_code = 1;
    struct stat s;

    if( stat(sInputPath.c_str(), &s) == 0 )
    {
        if( s.st_mode & S_IFREG )
        {
            filelist.push_back(sInputPath);
        }
        else if( s.st_mode & S_IFDIR )
        {
            // processing each file in directory
            DIR *dir_handle;
            struct dirent *dir;
            dir_handle = opendir(sInputPath.c_str());
            std::vector<std::string> filenames;
            if (dir_handle) 
            {
                error_code = 0; 
                while ((dir = readdir(dir_handle)) != NULL) 
                {
                    if (dir->d_type == DT_REG)
                    {
                        std::string sFileName = sInputPath + dir->d_name;
                        filelist.push_back(sFileName);
                    }
                    else if (dir->d_type == DT_DIR)
                    {
                        std::string sname = dir->d_name;
                        if (sname != "." && sname != "..")
                        {
                            readInput(sInputPath + sname + "/", filelist);
                        }
                    }
                }
                closedir(dir_handle);    
            }
            else
            {
                std::cout << "Cannot open input directory: " << sInputPath << std::endl;
                return error_code;
            }
        }
        else
        {
            std::cout << "Cannot open input: " << sInputPath << std::endl;
            return error_code;
        }
    }
    else
    {
        std::cout << "Cannot find input path " << sInputPath << std::endl;
        return error_code;
    }
    
    return 0;
}

double walltime(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
}

static const char *nppErrorString(NppStatus error) {
  switch (error) {
  case NPP_NOT_SUPPORTED_MODE_ERROR:
    return "NPP_NOT_SUPPORTED_MODE_ERROR";

  case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:
    return "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR";

  case NPP_RESIZE_NO_OPERATION_ERROR:
    return "NPP_RESIZE_NO_OPERATION_ERROR";

  case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:
    return "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY";

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) <= 0x5000

  case NPP_BAD_ARG_ERROR:
    return "NPP_BAD_ARGUMENT_ERROR";

  case NPP_COEFF_ERROR:
    return "NPP_COEFFICIENT_ERROR";

  case NPP_RECT_ERROR:
    return "NPP_RECTANGLE_ERROR";

  case NPP_QUAD_ERROR:
    return "NPP_QUADRANGLE_ERROR";

  case NPP_MEM_ALLOC_ERR:
    return "NPP_MEMORY_ALLOCATION_ERROR";

  case NPP_HISTO_NUMBER_OF_LEVELS_ERROR:
    return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";

  case NPP_INVALID_INPUT:
    return "NPP_INVALID_INPUT";

  case NPP_POINTER_ERROR:
    return "NPP_POINTER_ERROR";

  case NPP_WARNING:
    return "NPP_WARNING";

  case NPP_ODD_ROI_WARNING:
    return "NPP_ODD_ROI_WARNING";
#else

    // These are for CUDA 5.5 or higher
  case NPP_BAD_ARGUMENT_ERROR:
    return "NPP_BAD_ARGUMENT_ERROR";

  case NPP_COEFFICIENT_ERROR:
    return "NPP_COEFFICIENT_ERROR";

  case NPP_RECTANGLE_ERROR:
    return "NPP_RECTANGLE_ERROR";

  case NPP_QUADRANGLE_ERROR:
    return "NPP_QUADRANGLE_ERROR";

  case NPP_MEMORY_ALLOCATION_ERR:
    return "NPP_MEMORY_ALLOCATION_ERROR";

  case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR:
    return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";

  case NPP_INVALID_HOST_POINTER_ERROR:
    return "NPP_INVALID_HOST_POINTER_ERROR";

  case NPP_INVALID_DEVICE_POINTER_ERROR:
    return "NPP_INVALID_DEVICE_POINTER_ERROR";
#endif

  case NPP_LUT_NUMBER_OF_LEVELS_ERROR:
    return "NPP_LUT_NUMBER_OF_LEVELS_ERROR";

  case NPP_TEXTURE_BIND_ERROR:
    return "NPP_TEXTURE_BIND_ERROR";

  case NPP_WRONG_INTERSECTION_ROI_ERROR:
    return "NPP_WRONG_INTERSECTION_ROI_ERROR";

  case NPP_NOT_EVEN_STEP_ERROR:
    return "NPP_NOT_EVEN_STEP_ERROR";

  case NPP_INTERPOLATION_ERROR:
    return "NPP_INTERPOLATION_ERROR";

  case NPP_RESIZE_FACTOR_ERROR:
    return "NPP_RESIZE_FACTOR_ERROR";

  case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR:
    return "NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR";


#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) <= 0x5000

  case NPP_MEMFREE_ERR:
    return "NPP_MEMFREE_ERR";

  case NPP_MEMSET_ERR:
    return "NPP_MEMSET_ERR";

  case NPP_MEMCPY_ERR:
    return "NPP_MEMCPY_ERROR";

  case NPP_MIRROR_FLIP_ERR:
    return "NPP_MIRROR_FLIP_ERR";
#else

  case NPP_MEMFREE_ERROR:
    return "NPP_MEMFREE_ERROR";

  case NPP_MEMSET_ERROR:
    return "NPP_MEMSET_ERROR";

  case NPP_MEMCPY_ERROR:
    return "NPP_MEMCPY_ERROR";

  case NPP_MIRROR_FLIP_ERROR:
    return "NPP_MIRROR_FLIP_ERROR";
#endif

  case NPP_ALIGNMENT_ERROR:
    return "NPP_ALIGNMENT_ERROR";

  case NPP_STEP_ERROR:
    return "NPP_STEP_ERROR";

  case NPP_SIZE_ERROR:
    return "NPP_SIZE_ERROR";

  case NPP_NULL_POINTER_ERROR:
    return "NPP_NULL_POINTER_ERROR";

  case NPP_CUDA_KERNEL_EXECUTION_ERROR:
    return "NPP_CUDA_KERNEL_EXECUTION_ERROR";

  case NPP_NOT_IMPLEMENTED_ERROR:
    return "NPP_NOT_IMPLEMENTED_ERROR";

  case NPP_ERROR:
    return "NPP_ERROR";

  case NPP_SUCCESS:
    return "NPP_SUCCESS";

  case NPP_WRONG_INTERSECTION_QUAD_WARNING:
    return "NPP_WRONG_INTERSECTION_QUAD_WARNING";

  case NPP_MISALIGNED_DST_ROI_WARNING:
    return "NPP_MISALIGNED_DST_ROI_WARNING";

  case NPP_AFFINE_QUAD_INCORRECT_WARNING:
    return "NPP_AFFINE_QUAD_INCORRECT_WARNING";

  case NPP_DOUBLE_SIZE_WARNING:
    return "NPP_DOUBLE_SIZE_WARNING";

  case NPP_WRONG_INTERSECTION_ROI_WARNING:
    return "NPP_WRONG_INTERSECTION_ROI_WARNING";

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) >= 0x6000
    /* These are 6.0 or higher */
  case NPP_LUT_PALETTE_BITSIZE_ERROR:
    return "NPP_LUT_PALETTE_BITSIZE_ERROR";

  case NPP_ZC_MODE_NOT_SUPPORTED_ERROR:
    return "NPP_ZC_MODE_NOT_SUPPORTED_ERROR";

  case NPP_QUALITY_INDEX_ERROR:
    return "NPP_QUALITY_INDEX_ERROR";

  case NPP_CHANNEL_ORDER_ERROR:
    return "NPP_CHANNEL_ORDER_ERROR";

  case NPP_ZERO_MASK_VALUE_ERROR:
    return "NPP_ZERO_MASK_VALUE_ERROR";

  case NPP_NUMBER_OF_CHANNELS_ERROR:
    return "NPP_NUMBER_OF_CHANNELS_ERROR";

  case NPP_COI_ERROR:
    return "NPP_COI_ERROR";

  case NPP_DIVISOR_ERROR:
    return "NPP_DIVISOR_ERROR";

  case NPP_CHANNEL_ERROR:
    return "NPP_CHANNEL_ERROR";

  case NPP_STRIDE_ERROR:
    return "NPP_STRIDE_ERROR";

  case NPP_ANCHOR_ERROR:
    return "NPP_ANCHOR_ERROR";

  case NPP_MASK_SIZE_ERROR:
    return "NPP_MASK_SIZE_ERROR";

  case NPP_MOMENT_00_ZERO_ERROR:
    return "NPP_MOMENT_00_ZERO_ERROR";

  case NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR:
    return "NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR";

  case NPP_THRESHOLD_ERROR:
    return "NPP_THRESHOLD_ERROR";

  case NPP_CONTEXT_MATCH_ERROR:
    return "NPP_CONTEXT_MATCH_ERROR";

  case NPP_FFT_FLAG_ERROR:
    return "NPP_FFT_FLAG_ERROR";

  case NPP_FFT_ORDER_ERROR:
    return "NPP_FFT_ORDER_ERROR";

  case NPP_SCALE_RANGE_ERROR:
    return "NPP_SCALE_RANGE_ERROR";

  case NPP_DATA_TYPE_ERROR:
    return "NPP_DATA_TYPE_ERROR";

  case NPP_OUT_OFF_RANGE_ERROR:
    return "NPP_OUT_OFF_RANGE_ERROR";

  case NPP_DIVIDE_BY_ZERO_ERROR:
    return "NPP_DIVIDE_BY_ZERO_ERROR";

  case NPP_RANGE_ERROR:
    return "NPP_RANGE_ERROR";

  case NPP_NO_MEMORY_ERROR:
    return "NPP_NO_MEMORY_ERROR";

  case NPP_ERROR_RESERVED:
    return "NPP_ERROR_RESERVED";

  case NPP_NO_OPERATION_WARNING:
    return "NPP_NO_OPERATION_WARNING";

  case NPP_DIVIDE_BY_ZERO_WARNING:
    return "NPP_DIVIDE_BY_ZERO_WARNING";
#endif

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) >= 0x7000
    /* These are 7.0 or higher */
  case NPP_OVERFLOW_ERROR:
    return "NPP_OVERFLOW_ERROR";

  case NPP_CORRUPTED_DATA_ERROR:
    return "NPP_CORRUPTED_DATA_ERROR";
#endif
  }

  return "<unknown>";
}