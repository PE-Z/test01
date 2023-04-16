#include "opencv2/opencv.hpp"//
//#include <Eigen/Dense>
#include <opencv2/core/cvstd.hpp>
//#include<X11/Xlib.h>

//#include "camera.h"
//#include "armor_track.h"//装甲板预测
#include <opencv2/core/cvstd.hpp>
//#include "serialport.h"
// #include "energy_predict.h"
//#include "serial_main.h"
#include <thread>
#include <mutex>
#include <string>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "armor_detection.h"
#include <opencv2/highgui/highgui.hpp>


//#define DETECT
#define PREDICT

using namespace cv;
using namespace std;

//#include "camera.h"

//图片
int main()
{
    ArmorDetector Detect;
    std::vector<Armor> Targets;
    Mat src_copy;
    Mat src;
    // 从指定路径读取图片
    src = imread("C:/Users/86133/Desktop/视觉学习资料/视觉实战考核/蓝色灯条1.jpg");//背景亮度过高时，检测不出灯条
    if (src.empty())
    {
        printf("Fail to read image file!\n");
        return -1;
    }
    src.copyTo(src_copy);
    //imshow("src", src);

    Targets = Detect.autoAim(src_copy);

    //auto time_start = std::chrono::steady_clock::now();
    //// 进行装甲板自瞄，并获取自瞄目标
    //autoTarget = autoShoot.autoAim(src, 0);
    //// 显示原始图像
    //imshow("src", src);
    //// 如果检测到自瞄目标，则输出信息
    //if (!autoTarget.size.empty())
    //{
    //    printf("main get target!!!\n");
    //}
    
    // 等待按键事件
    waitKey(0);

    return 0;
}

////摄像头
//int main()
//{
//    ArmorDetector Detect;
//    std::vector<Armor> Targets;
//    Mat src_copy;
//
//    // 打开本地摄像头
//    VideoCapture cap(0);
//    if (!cap.isOpened())
//    {
//        cerr << "Failed to open camera!" << endl;
//        return -1;
//    }
//
//    while (true)
//    {
//        Mat src;
//        auto time_start = chrono::steady_clock::now();
//
//        // 从摄像头读取一帧图像
//        cap.read(src);
//        src.copyTo(src_copy);
//
//        if (!src.empty())
//        {
//            auto time_cap = chrono::steady_clock::now();
//            int time_stamp = (int)(chrono::duration<double, std::milli>(time_cap - time_start).count());
//
//
//
//            Targets = Detect.autoAim(src_copy);
//
//        }
//
//        // 等待按键或者一段时间后退出循环
//        if (waitKey(1) == 27) // ESC 键
//        {
//            break;
//        }
//    }
//
//    return 0;
//}
