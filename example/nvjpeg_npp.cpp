
#include "nvjpeg_npp.hpp"

#define streamNum 4
#define MAX_THREAD 12

int dev_malloc(void** p, size_t s)
{
    return (int)cudaMalloc(p, s);
}

int dev_free(void* p)
{
    return (int)cudaFree(p);
}

typedef std::vector<std::string> FileNames;
typedef std::vector<std::vector <char> > FileData;

struct deocde_params_t
{
    std::string input_dir;
    int batch_size;
    int total_images;
    int dev;
    int warmup;

    nvjpegJpegState_t nvjpeg_state;
    nvjpegHandle_t nvjpeg_handle;
    cudaStream_t stream[streamNum];

    nvjpegOutputFormat_t fmt;
    bool write_decoded;
    std::string output_dir;

    bool pipelined;
    bool batched;
};

int read_next_batch(FileNames& image_names, int batch_size, FileNames::iterator& cur_iter, FileData& raw_data, 
    std::vector<size_t>& raw_len, FileNames &current_names)
{
    int counter = 0;

    while (counter < batch_size)
    {
        if (cur_iter == image_names.end())
        {
            std::cerr << "Image list is too short to fill the batch, adding files from the beginning of the image list" << std::endl;
            cur_iter = image_names.begin();
        }

        if (image_names.size() == 0)
        {
            std::cerr << "No valid images left in the input list, exit" << std::endl;
            return 1;   
        }

        // Read an image from disk.
        std::ifstream input(cur_iter->c_str(), std::ios::in | std::ios::binary | std::ios::ate);
        if(!(input.is_open()))
        {
            std::cerr << "Cannot open image: " << *cur_iter << ", removing it from image list" << std::endl;
            image_names.erase(cur_iter);
            continue;
        }   

        // Get the size
        std::streamsize file_size = input.tellg();
        input.seekg(0, std::ios::beg);
        // resize if buffer is too small
        if (raw_data[counter].size() < file_size)
        {
            raw_data[counter].resize(file_size);
        }
        if (!input.read(raw_data[counter].data(), file_size))
        {
            std::cerr << "Cannot read from file: " << *cur_iter << ", removing it from image list" << std::endl;
            image_names.erase(cur_iter);
            continue;
        }
        raw_len[counter] = file_size;

        current_names[counter] = *cur_iter;

        counter++;
        cur_iter++;
    }
    return 0;
}

//prepare buffers for RGBi output format
int prepare_buffers(FileData& file_data,
    std::vector<size_t>& file_len,
    std::vector<int>& img_width,
    std::vector<int>& img_height,
    std::vector<nvjpegImage_t> &ibuf,
    std::vector<nvjpegImage_t> &isz,
    FileNames &current_names,
    deocde_params_t &params,
    int stream_index)
{
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    int channels;
    nvjpegChromaSubsampling_t subsampling;

    for (int i = 0; i < file_data.size(); i++)
    {
        nvjpegStatus_t err;
      err = nvjpegGetImageInfo(params.nvjpeg_handle, (unsigned char*)file_data[i].data(), 
            file_len[i], &channels, &subsampling, widths, heights);
        if (err)
        {
            std::cout << "Cannot decode JPEG header: #" << i << std::endl;
            return 1;
        }
        img_width[i] = widths[0];
        img_height[i] = heights[0];

        // std::cout << "Processing: " << current_names[i] << std::endl;
        // std::cout << "Image is " << channels << " channels." << std::endl;
        // for (int c = 0; c < channels; c++)
        // {
        //     std::cout << "Channel #" << c << " size: "  << widths[c]  << " x " << heights[c] << std::endl;    
        // }
        
        // switch (subsampling)
        // {
        //     case NVJPEG_CSS_444:
        //         std::cout << "YUV 4:4:4 chroma subsampling" << std::endl;
        //         break;
        //     case NVJPEG_CSS_440:
        //         std::cout << "YUV 4:4:0 chroma subsampling" << std::endl;
        //         break;
        //     case NVJPEG_CSS_422:
        //         std::cout << "YUV 4:2:2 chroma subsampling" << std::endl;
        //         break;
        //     case NVJPEG_CSS_420:
        //         std::cout << "YUV 4:2:0 chroma subsampling" << std::endl;
        //         break;
        //     case NVJPEG_CSS_411:
        //         std::cout << "YUV 4:1:1 chroma subsampling" << std::endl;
        //         break;
        //     case NVJPEG_CSS_410:
        //         std::cout << "YUV 4:1:0 chroma subsampling" << std::endl;
        //         break;
        //     case NVJPEG_CSS_GRAY:
        //         std::cout << "Grayscale JPEG " << std::endl;
        //         break;
        //     case NVJPEG_CSS_UNKNOWN: 
        //         std::cout << "Unknown chroma subsampling" << std::endl;
        //         return 1;
        // }

        int mul = 1;
        // in the case of interleaved RGB output, write only to single channel, but 3 samples at once
        if (params.fmt == NVJPEG_OUTPUT_RGBI || params.fmt == NVJPEG_OUTPUT_BGRI)
        {
            channels = 1;
            mul = 3;
        }
        // in the case of rgb create 3 buffers with sizes of original image
        else if (params.fmt == NVJPEG_OUTPUT_RGB || params.fmt == NVJPEG_OUTPUT_BGR)
        {
            channels = 3;
            widths[1] = widths[2] = widths[0];
            heights[1] = heights[2] = heights[0];
        }

        // realloc output buffer if required
        for (int c = 0; c < channels; c++)
        {
            int aw = mul*widths[c];
            int ah = heights[c];
            int sz = aw*ah;
            ibuf[i].pitch[c] = aw;
            if (sz > isz[i].pitch[c])
            {
                if (ibuf[i].channel[c])
                {
                    cudaFree(ibuf[i].channel[c]);
                    cudaCheckError_str("===buffer prepare cuda free");
                }
                cudaMalloc(&ibuf[i].channel[c], sz);
                cudaCheckError_str("===buffer prepare cuda malloc");
                isz[i].pitch[c] = sz;
            }
        }        
    }
    return 0;
}

void release_buffers(std::vector<nvjpegImage_t> &ibuf)
{
    for (int i = 0; i < ibuf.size(); i++)
    {
        for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++)
            if (ibuf[i].channel[c])
                cudaFree(ibuf[i].channel[c]);
    }
}

int resizeImage(nvjpegImage_t& out, unsigned char* hDst, int width, int height, int  resize_w, int resize_h, cudaStream_t stream)
{
    cudaError_t eCopy;
    // cout<<width<<" "<<height<<endl;
    // hDst = Mat(Size(resize_w,resize_h), CV_8UC3, resize_w*3);
     
    int nDstPitchCUDA;
    nppSetStream(stream);
    // cudaStreamSynchronize(stream);
    Npp8u *dDst = nppiMalloc_8u_C3(resize_w, resize_h, &nDstPitchCUDA);

    NppiSize inSize = {width, height};
    NppiSize outSize = {resize_w, resize_h};
    NppiRect inRect = {0, 0, width-1, height-1};
    NppiRect outRect = {0, 0, resize_w-1, resize_h-1};

    NppStatus status = nppiResize_8u_C3R( out.channel[0], width*3, inSize,
                        inRect, dDst, nDstPitchCUDA,
                         outSize, outRect, NPPI_INTER_LINEAR);
    cudaCheckError_str("==resize ");
    // cudaStreamSynchronize(stream);
    // eCopy = cudaMemcpy2D(hDst, resize_w*3, dDst, nDstPitchCUDA, resize_w*3, resize_h, cudaMemcpyDeviceToHost);
    eCopy = cudaMemcpy2DAsync(hDst, resize_w*3, dDst, nDstPitchCUDA, resize_w*3, resize_h, cudaMemcpyDeviceToHost, stream);

    if(cudaSuccess != eCopy) 
    {
        std::cerr << "cudaMemcpy2D failed: " << cudaGetErrorString(eCopy) << std::endl;
        return 1;
    }
    nppiFree(dDst);
    // hDst.release();
    cudaCheckError_str("==cuda free");
    return 0;
}

int decode_images(const FileData& img_data, 
            const std::vector<size_t>& img_len, 
            std::vector<nvjpegImage_t> &out,
            deocde_params_t &params,
            vector<int>& width, vector<int>& height,
            int stream_index)
{
    // cudaStreamSynchronize(params.stream[stream_index]);
    // cudaCheckError_str("===decode stream");
    nvjpegStatus_t err;
    
    if (!params.batched)
    {
        if (!params.pipelined)
        {
            int thread_idx = 0;
            for (int i = 0; i < params.batch_size; i++)
            {
                if ((err = nvjpegDecode(params.nvjpeg_handle, params.nvjpeg_state, (const unsigned char *)img_data[i].data(), 
                    img_len[i], params.fmt, &out[i], params.stream[stream_index])) != NVJPEG_STATUS_SUCCESS)
                {
                    std::cerr << "Error in nvjpeg decode: #" << err << std::endl;
                    return 1;
                }
                // cudaStreamSynchronize(params.stream[stream_index]);//blocking and waiting works in stream done 
                // cudaCheckError();
            }
        }
        else
        {
            int thread_idx = 0;
            for (int i = 0; i < params.batch_size; i++)
            {
                if ((err = nvjpegDecodePhaseOne(params.nvjpeg_handle,params. nvjpeg_state, (const unsigned char *)img_data[i].data(), 
                    img_len[i], params.fmt, params.stream[stream_index])) != NVJPEG_STATUS_SUCCESS)
                {
                    std::cerr << "Error in nvjpeg decode pipelined: #" << err << std::endl;
                    return 1;
                }
                // cudaStreamSynchronize(params.stream[stream_index]);
                // cudaCheckError();
                if ((err = nvjpegDecodePhaseTwo(params.nvjpeg_handle, params.nvjpeg_state, params.stream[stream_index])) != NVJPEG_STATUS_SUCCESS)
                {
                    std::cerr << "Error in nvjpeg decode pipelined: #" << err << std::endl;
                    return 1;
                }
                cudaCheckError();
                if ((err = nvjpegDecodePhaseThree(params.nvjpeg_handle, params.nvjpeg_state, &out[i], params.stream[stream_index])) != NVJPEG_STATUS_SUCCESS)
                {
                    std::cerr << "Error in nvjpeg decode pipelined: #" << err << std::endl;
                    return 1;
                }
            }
            // cudaStreamSynchronize(params.stream[stream_index]);
            // cudaCheckError();
        }
    }
    else
    {
        std::vector<const unsigned char*> raw_inputs;
        for (int i = 0; i < params.batch_size; i++)
        {
            raw_inputs.push_back((const unsigned char*)img_data[i].data());
        }

        if (!params.pipelined)
        {

            double decode_time = walltime();

            if((err = nvjpegDecodeBatched(params.nvjpeg_handle, params.nvjpeg_state, 
                raw_inputs.data(), img_len.data(), out.data(), params.stream[stream_index]))!=NVJPEG_STATUS_SUCCESS)
            {
                    std::cerr << "Error in nvjpeg decode batched#" << err << std::endl;
                    return 1;                
            }
            double ttime = walltime() - decode_time;

            // cudaStreamSynchronize(params.stream[stream_index]);
            
            cout<<"time : "<<ttime<<endl;
            cudaError_t state = cudaStreamQuery(params.stream[stream_index]);
            switch(state)
            {
                case cudaSuccess: 
                        cout<<" have finished "<<stream_index<<endl;
                        break;
                case cudaErrorNotReady: 
                        cout<<" have not finished "<<stream_index<<endl;
                        break;
                default :
                        cout<<" error in stream query"<<endl;
            }
            cudaCheckError();
        }
        else
        {
            cudaStreamSynchronize(params.stream[stream_index]);            
            int thread_idx = -1;
            for (int i = 0; i < params.batch_size; i++)
            {
                thread_idx = (++thread_idx) % MAX_THREAD;
                if ((err = nvjpegDecodeBatchedPhaseOne(params.nvjpeg_handle, params.nvjpeg_state, raw_inputs[i], 
                            img_len[i], i, thread_idx, params.stream[stream_index])) != NVJPEG_STATUS_SUCCESS)
                {
                    std::cerr << "Error in nvjpegDecodeBatchedPhaseOne: #" << err << std::endl;
                    return 1;
                }

            }
            if ((err = nvjpegDecodeBatchedPhaseTwo(params.nvjpeg_handle, params.nvjpeg_state, params.stream[stream_index])) != NVJPEG_STATUS_SUCCESS)
            {
                std::cerr << "Error in nvjpegDecodeBatchedPhaseTwo: #" << err << std::endl;
                return 1;
            }
            if ((err = nvjpegDecodeBatchedPhaseThree(params.nvjpeg_handle, params.nvjpeg_state, out.data(), params.stream[stream_index])) != NVJPEG_STATUS_SUCCESS)
            {
                std::cerr << "Error in nvjpegDecodeBatchedPhaseThree: #" << err << std::endl;
                return 1;
            }

            // cudaError_t state = cudaStreamQuery(params.stream[stream_index]);
            // switch(state)
            // {
            //     case cudaSuccess: 
            //             cout<<" have finished "<<stream_index<<endl;
            //             break;
            //     case cudaErrorNotReady: 
            //             cout<<" have not finished "<<stream_index<<endl;
            //             break;
            //     default :
            //             cout<<" error in stream query"<<endl;
            // }
            // cudaStreamSynchronize(params.stream[stream_index]);
            // cudaCheckError();
        }
    }

    return 0;
}



int resize_images(std::vector<nvjpegImage_t> &iout, FileNames& filenames, std::vector<int> widths, std::vector<int> heights,
                     int resize_w, int resize_h, deocde_params_t &params, int stream_index)
{
    unsigned char  *dDst;
    for(int i = 0;i<params.batch_size; i++)
    {
        unsigned char *hDst;
        // hDst = (unsigned char*)malloc(resize_w*resize_h*3);
        //pinned memory for cudaMemcpy2DAsync
        cudaMallocHost((void **)&(hDst), (size_t)resize_w*resize_h*3);
        size_t position = filenames[i].rfind("/");
        std::string sFileName = (std::string::npos == position)? filenames[i] : filenames[i].substr(position + 1, filenames[i].size());
        position = sFileName.rfind(".");
        sFileName = (std::string::npos == position)? sFileName : sFileName.substr(0, position);
        std::string fname(params.output_dir + "/" + sFileName + ".jpg");
        // cudaStreamSynchronize(params.stream[stream_index]);

        resizeImage(iout[i], hDst, widths[i], heights[i], resize_w, resize_h, params.stream[stream_index]);
        // cudaStreamSynchronize(params.stream[stream_index]);
        cudaCheckError_str("===resizeImage");
        Mat image = Mat(Size(resize_w,resize_h),CV_8UC3, hDst, resize_w*3);

        imwrite(fname,image);
        cudaFreeHost(hDst);  
        // free(hDst);
        cudaCheckError_str("===cudaFreeHost");

    }
    return 0;   
}

double 
process_images(FileNames& image_names, deocde_params_t &params, double &total)
{
    // vector for storing raw files and file lengths
    // vector for storing raw files and file lengths
    FileData file_data(params.batch_size);
    std::vector<size_t> file_len(params.batch_size);
    FileNames current_names(params.batch_size);
    std::vector<int> widths(params.batch_size);
    std::vector<int> heights(params.batch_size);
    // we wrap over image files to process total_images of files
    FileNames::iterator file_iter = image_names.begin();

    // stream for decoding
    for(int i = 0; i<streamNum; i++){
        cudaStreamCreateWithFlags(&params.stream[i], cudaStreamNonBlocking);
    }
    cudaCheckError();

    int total_processed = 0;

    // output buffers
    std::vector<nvjpegImage_t> iout(params.batch_size);
    // output buffer sizes, for convenience
    std::vector<nvjpegImage_t> isz(params.batch_size);

    for (int i = 0; i < iout.size(); i++)
    {
        for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++)
        {
            iout[i].channel[c] = NULL;
            iout[i].pitch[c] = 0;
            isz[i].pitch[c] = 0;
        }
    }
    double test_time = walltime();
    int warmup = 0;
    int index = -1;
    while (total_processed < params.total_images)
    {    
        for(int index = 0; index < streamNum; index++){
            double   begin_time = walltime();        
            if (read_next_batch(image_names, params.batch_size, file_iter, file_data, file_len, current_names)) {return 1;}
            if (prepare_buffers(file_data, file_len, widths, heights, iout, isz, current_names, params, index)) { return 1;}
            if (decode_images(file_data, file_len, iout, params, widths, heights, index))  {return 1;}

            if (warmup < params.warmup) {
                warmup++;
            }
            else{
                total_processed += params.batch_size;
                cout<<"-"<<total_processed;
                cout.flush(); 
            }
        }
        for(int index = 0; index<streamNum; index++){
            resize_images(iout, current_names, widths, heights, 400, 400, params, index);
        }
    }
    total = walltime() - test_time;

    release_buffers(iout);

    for(int i = 0; i<streamNum; i++){
        cudaStreamDestroy(params.stream[i]);
    }
    cudaCheckError_str("===stream destroy 0");        

    return 0;
}

// parse parameters
int findParamIndex(const char **argv, int argc, const char *parm)
{
    int count = 0;
    int index = -1;

    for (int i = 0; i < argc; i++)
    {
        if (strncmp(argv[i], parm, 100) == 0)
        {
            index = i;
            count++;
        }
    }

    if (count == 0 || count == 1)
    {
        return index;
    }
    else
    {
        std::cout << "Error, parameter " << parm << " has been specified more than once, exiting\n" << std::endl;
        return -1;
    }

    return -1;
}

int main(int argc, const char * argv[])
{
    int pidx;

    if ((pidx = findParamIndex(argv, argc, "-h")) != -1 ||
        (pidx = findParamIndex(argv, argc, "--help")) != -1)
    {
        std::cout << "Usage: " << argv[0] << " -i images_dir [-b batch_size] [-t total_images] [-dev device_id] [-w warmup_iterations] [-o output_dir] [-pipelined] [-batched] [-fmt output_format]\n";
        std::cout << "Parameters: " << std::endl;
        std::cout << "\timages_dir\t:\tPath to single image or directory of images" << std::endl;
        std::cout << "\tbatch_size\t:\tDecode images from input by batches of specified size" << std::endl;
        std::cout << "\ttotal_images\t:\tDecode this much images, if there are less images \n" 
            <<"\t\t\t\t\tin the input than total images, decoder will loop over the input" << std::endl;
        std::cout << "\tdevice_id\t:\tWhich device to use for decoding" << std::endl;
        std::cout << "\twarmup_iterations\t:\tRun this amount of batches first without measuring performance" << std::endl;
        std::cout << "\toutput_dir\t:\tWrite decoded images as BMPs to this directory" << std::endl;
        std::cout << "\tpipelined\t:\tUse decoding in phases" << std::endl;
        std::cout << "\tbatched\t\t:\tUse batched interface" << std::endl;
        std::cout << "\toutput_format\t:\tnvJPEG output format for decoding. One of [rgb, rgbi, bgr, bgri, yuv, y, unchanged]" << std::endl;
        return 1;
    }

    deocde_params_t params;

    params.input_dir = "./";
    if ((pidx = findParamIndex(argv, argc, "-i")) != -1)
    {
        params.input_dir = argv[pidx + 1];
    }
    else
    {
        std::cerr << "Please specify input directory with encoded images" << std::endl;
        return 1;
    }

    params.batch_size = 1;
    if ((pidx = findParamIndex(argv, argc, "-b")) != -1)
    {
        params.batch_size = std::atoi(argv[pidx + 1]);
    }

    params.total_images = -1;
    if ((pidx = findParamIndex(argv, argc, "-t")) != -1)
    {
        params.total_images = std::atoi(argv[pidx + 1]);
    }

    params.dev = 0;
    if ((pidx = findParamIndex(argv, argc, "-dev")) != -1)
    {
        params.dev = std::atoi(argv[pidx + 1]);
    }

    params.warmup = 0;
    if ((pidx = findParamIndex(argv, argc, "-w")) != -1)
    {
        params.warmup = std::atoi(argv[pidx + 1]);
    }

    params.batched = false;
    if ((pidx = findParamIndex(argv, argc, "-batched")) != -1)
    {
        params.batched = true;
    }

    params.pipelined = false;
    if ((pidx = findParamIndex(argv, argc, "-pipelined")) != -1)
    {
        params.pipelined = true;
    }

    params.fmt = NVJPEG_OUTPUT_RGB;
    if ((pidx = findParamIndex(argv, argc, "-fmt")) != -1)
    {
        std::string sfmt = argv[pidx + 1];
        if (sfmt == "rgb")
            params.fmt = NVJPEG_OUTPUT_RGB;
        else if (sfmt == "bgr")
            params.fmt = NVJPEG_OUTPUT_BGR;
        else if (sfmt == "rgbi")
            params.fmt = NVJPEG_OUTPUT_RGBI;
        else if (sfmt == "bgri")
            params.fmt = NVJPEG_OUTPUT_BGRI;
        else if (sfmt == "yuv")
            params.fmt = NVJPEG_OUTPUT_YUV;
        else if (sfmt == "y")
            params.fmt = NVJPEG_OUTPUT_Y;
        else if (sfmt == "unchanged")
            params.fmt = NVJPEG_OUTPUT_UNCHANGED;
        else
        {
            std::cout << "Unknown format: " << sfmt << std::endl;
            return 1;
        }
        
    }

    params.write_decoded = false;
    if ((pidx = findParamIndex(argv, argc, "-o")) != -1)
    {
        params.output_dir = argv[pidx + 1];
        if (params.fmt != NVJPEG_OUTPUT_RGB && 
            params.fmt != NVJPEG_OUTPUT_BGR && 
            params.fmt != NVJPEG_OUTPUT_RGBI && 
            params.fmt != NVJPEG_OUTPUT_BGRI)
        {
            std::cout << "We can write ony BMPs, which require output format be either RGB/BGR or RGBi/BGRi" << std::endl;
            return 1;
        }
        params.write_decoded = true;
    }

    cudaDeviceProp props;
    cudaSetDevice(params.dev);
    cudaCheckError();
    cudaGetDeviceProperties(&props, params.dev);
    cudaCheckError();
    printf("Using GPU %d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s, overlap %s)\n",
        params.dev, props.name, props.multiProcessorCount,
        props.maxThreadsPerMultiProcessor,
        props.major, props.minor,
        props.ECCEnabled?"on":"off",
        props.deviceOverlap?"on":"off");
    
    cudaFree(0);
    cudaCheckError();

    nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
    for(int i = 0; i < streamNum; i++){    
        nvjpegCheckError(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &params.nvjpeg_handle));
        nvjpegCheckError(nvjpegJpegStateCreate(params.nvjpeg_handle, &params.nvjpeg_state)); 
        nvjpegCheckError(nvjpegDecodeBatchedInitialize(params.nvjpeg_handle, params.nvjpeg_state, params.batch_size, MAX_THREAD, params.fmt));
    }
    cudaCheckError();
  
    // read source images
    FileNames image_names;
    readInput(params.input_dir, image_names);

    if (params.total_images == -1)
    {
        params.total_images = image_names.size();
    }
    else if (params.total_images % params.batch_size)
    {
        params.total_images = ((params.total_images)/params.batch_size)*params.batch_size;
        std::cout << "Changing total_images number to " << params.total_images 
            << " to be multiple of batch_size - " << params.batch_size << std::endl;
    }

    std::cout << "Decoding images in directory: " << params.input_dir << ", total " << params.total_images 
        << ", batchsize " << params.batch_size << std::endl;


    double total; 
    if (process_images(image_names, params, total))
        return 1;
    std::cout << "Total decoding time: " << total << std::endl;
    std::cout << "Avg decoding time per image: " << total/params.total_images << std::endl;
    std::cout << "Avg images per sec: " << params.total_images/total << std::endl;
    std::cout << "Avg decoding time per batch: " << total/((params.total_images + params.batch_size - 1)/params.batch_size) << std::endl;


    nvjpegCheckError(nvjpegJpegStateDestroy(params.nvjpeg_state));
    nvjpegCheckError(nvjpegDestroy(params.nvjpeg_handle));

    cudaDeviceReset();
    cudaCheckError();
    return 0;
}
