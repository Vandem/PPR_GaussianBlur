#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <omp.h>
#include <numbers>
using namespace cv;
using namespace std;

//kernel generator from: https://gitlab.com/spook/Animator
shared_ptr<float[]> generateKernel(int diameter)
{
    float sigma = 1;
    shared_ptr<float[]> kernel(new float[diameter * diameter]);
    int mean = diameter / 2;
    float sum = 0.0; // For accumulating the kernel values
    for (int x = 0; x < diameter; ++x)
        for (int y = 0; y < diameter; ++y) {
            kernel[y * diameter + x] = (float)(exp(-0.5 * (pow((x - mean) / sigma, 2.0) + pow((y - mean) / sigma, 2.0))) / (2 * numbers::pi * sigma * sigma));

            // Accumulate the kernel values
            sum += kernel[y * diameter + x];
        }

    // Normalize the kernel
    for (int x = 0; x < diameter; ++x)
        for (int y = 0; y < diameter; ++y)
            kernel[y * diameter + x] /= sum;

    return kernel;
}


void gaussianBlur(unsigned char* inuputImage, unsigned char* outputImage, int width, int height, int radius)
{
    int diameter = 2 * radius + 1;
    shared_ptr<float[]> kernel = generateKernel(diameter);


    int row, col, c, sum, x, y, color;
    float sumKernel, kernelValue;

    //omp_set_num_threads(NUM_THREADS);
# pragma omp parallel \
  shared ( diameter, radius, width, height) \
  private ( row, col, x, y, c, sum, color, kernelValue, sumKernel)
# pragma omp for
    for (row = 0; row < height; row++)
    {
        for (col = 0; col < width; col++)
        {
            for (c = 0; c < 3; c++)
            {
                sum = 0;
                sumKernel = 0;

                for (y = -radius; y <= radius; y++)
                {
                    for (x = -radius; x <= radius; x++)
                    {
                        kernelValue = kernel[(x + radius) * diameter + y + radius];

                        if ((row + y) >= 0 && (row + y) < height && (col + x) >= 0 && (col + x) < width)
                        {
                            color = inuputImage[(row + y) * 3 * width + (col + x) * 3 + c];
                            sum += color * kernelValue;
                            sumKernel += kernelValue;
                        }
                    }
                }

                if (sumKernel > 0) {
                    outputImage[3 * row * width + 3 * col + c] = sum / sumKernel;
                }
            }
        }
    }
}

int main()
{
    Mat img = cv::imread("C:/Users/josch/FH/PPR/GaussianFilter/background.jpg");
    Mat out = img.clone();

    int radius = 9;

    for (int i = 0; i < 20; i++)
    {
    //double start_time = omp_get_wtime();
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();

    //gaussianBlur(img.data, out.data, img.cols, img.rows, radius);
    GaussianBlur(img, out, Size(radius, radius), 1.0);

    chrono::steady_clock::time_point end = chrono::steady_clock::now();

    long time = chrono::duration_cast<chrono::milliseconds>(end - begin).count();
    //double time = omp_get_wtime() - start_time;
    printf("Time: %ld milliseconds.\n", time);
    }

    //imshow("img", img);
    //imshow("out", out);
    waitKey(0);

    return 0;
}