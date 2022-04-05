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

const int NUM_THREADS = 16;

//kernel generator from: https://gitlab.com/spook/Animator
shared_ptr<float[]> generateKernel(int diameter, float sigma = 1)
{
    int x, y, mean;
    float sum;

    //sigma = 1;
    shared_ptr<float[]> kernel(new float[diameter * diameter]);
    mean = diameter / 2;
    sum = 0.0; // For accumulating the kernel values


# pragma omp parallel \
shared(sigma, mean, kernel) \
private (x, y)
# pragma omp for reduction(+:sum)
    for (x = 0; x < diameter; ++x)
        for (y = 0; y < diameter; ++y) {
            kernel[y * diameter + x] = (float)(exp(-0.5 * (pow((x - mean) / sigma, 2.0) + pow((y - mean) / sigma, 2.0))) / (2 * numbers::pi * sigma * sigma));

            // Accumulate the kernel values
            sum += kernel[y * diameter + x];
        }

    // Normalize the kernel
    for (x = 0; x < diameter; ++x)
        for (y = 0; y < diameter; ++y)
            kernel[y * diameter + x] /= sum;

    return kernel;
}


void gaussianBlur(unsigned char* inuputImage, unsigned char* outputImage, int width, int height, int radius, float sigma = 1)
{
    int diameter = 2 * radius + 1;
    shared_ptr<float[]> kernel = generateKernel(diameter, sigma);


    int row, col, c, x, y, color;
    float sumKernel, kernelValue, sum;

    omp_set_num_threads(NUM_THREADS);
# pragma omp parallel \
  shared ( diameter, radius, width, height) \
  private ( row, col, x, y, c, sum, color, kernelValue, sumKernel)
# pragma omp for schedule(dynamic)
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
                    // no race condition because every channel per pixel has a dedicated index in the array
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
    float sigma = 20;

    //for (int i = 0; i < 10; i++)
    //{
    double start_time = omp_get_wtime();

    gaussianBlur(img.data, out.data, img.cols, img.rows, radius, sigma);
    //GaussianBlur(img, out, Size(radius, radius), sigma);

    double end = omp_get_wtime();
    double time = end - start_time;
    printf("Time: %lf milliseconds.\n", time * 1000);
    //}

    imshow("img", img);
    imshow("out", out);
    waitKey(0);

    return 0;
}