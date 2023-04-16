#pragma once
#ifndef ARMOR_DETECTION_H
#define ARMOR_DETECTION_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
//#include "robot_status.h"
#include "number_dnn.h"
#include <iostream>

#define POINT_DIST(p1,p2) std::sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y))//两点间距

enum EnermyColor { RED = 0, BLUE = 1 };//颜色
enum EnermyType { SMALL = 0, BIG = 1 };//装甲板大还是小


struct Light : public cv::RotatedRect     //灯条结构体，继承RotatedRect
{
    Light() = default;
    explicit Light(cv::RotatedRect& box) : cv::RotatedRect(box)
    {
        cv::Point2f p[4];//点集存放端点
        box.points(p);//调用RotatedRect的四个点初始化点集
        std::sort(p, p + 4, [](const cv::Point2f& a, const cv::Point2f& b) {return a.y < b.y; });//按y轴坐标从小到大排序
        top = (p[0] + p[1]) / 2;//上端点坐标
        bottom = (p[2] + p[3]) / 2;//下端点坐标
        height = POINT_DIST(top, bottom);//灯条的长度
        width = POINT_DIST(p[0], p[1]);//灯条的宽度

        angle = top.x < bottom.x ? box.angle : 90 + box.angle;//灯条的角度，要考虑上下端点倒置
        if (fabs(bottom.x - top.x) <= 0.01) angle = 90;//如果上下端点x坐标差很小，则认为角度为90
    }
    int lightColor;//灯条的颜色
    cv::Point2f top;//上端点坐标
    cv::Point2f bottom;//下端点坐标
    double angle;//灯条的角度
    double height;//灯条的长度
    double width;//灯条的宽度

};

//装甲板结构体
struct Armor : public cv::RotatedRect    //装甲板结构体继承RotatedRect
{
    Armor() = default;
    explicit Armor(cv::RotatedRect& box) : cv::RotatedRect(box)
    {
        confidence = 0;  //置信度
        id = 0;  //装甲板数字
        type = SMALL;  //装甲板类型
        grade = 0;  //等级
    }

    cv::Point2f armor_pt4[4]; //左下角开始逆时针
    //        std::vector<cv::Point2f> armor_pt4; //左下角开始逆时针
    float confidence;//置信度
    int id;  // 装甲板类别
    int grade;//装甲板得分
    int type;  // 装甲板类型
    //Eigen::Vector3d world_position;  // 当前的真实坐标
    //Eigen::Vector3d camera_position;  // 当前的真实坐标
    //    int area;  // 装甲板面积
};



// 主类，实现目标检测
class ArmorDetector
{
public:
    ArmorDetector(); // 构造函数初始化

    std::vector<Armor> autoAim(const cv::Mat& src); // 将最终目标的坐标转换到摄像头原大小的
private:
    int save_num_cnt;//保存的序号【】【】【】【】【】【
    int binThresh; // 二值化阈值
    int enemy_color; // 敌方颜色
    int categories; // 分类数

    // 灯条判定条件
    double light_max_angle; //灯条最大偏转角
    double light_min_hw_ratio;//宽高比
    double light_max_hw_ratio;
    double light_min_area_ratio;// 旋转矩形和矩形比率
    double light_max_area_ratio;
    double light_max_area;//限制灯条面积条件

    // 装甲板判定条件
    double armor_big_max_wh_ratio;//大装甲板宽高比
    double armor_big_min_wh_ratio;
    double armor_small_max_wh_ratio;//小装甲板宽高比
    double armor_small_min_wh_ratio;
    double armor_max_angle;//装甲板最大偏转角
    double armor_height_offset;
    double armor_ij_min_ratio;//左右灯条的高度比
    double armor_ij_max_ratio;
    double armor_max_offset_angle;//左右灯条角度差

    // 装甲板分级统计所需的参数
    double near_standard;//靠近图像中心打分阈值
    int height_standard;//装甲板标准高度
    int grade_standard;//及格分
    double id_grade_ratio;//id等级比率（乘系数，算总分）
    double height_grade_ratio;//高度等级比率
    double near_grade_ratio;//接近等级比率

    float thresh_confidence; // 置信度阈值

    cv::Mat _src; // 裁剪src后的ROI
    cv::Mat _binary;
    std::vector<cv::Mat> temps;//【】【】【】【？？

    Armor lastArmor;

    std::vector<Light> candidateLights; // 筛选的灯条
    std::vector<Armor> candidateArmors; // 筛选的装甲板
    std::vector<Armor> finalArmors;//最终装甲板
    std::vector<cv::Mat> numROIs;//id的ROI区域
    Armor finalArmor; // 最终装甲板

    DNN_detect dnnDetect;//用于数字识别

    void setImage(const cv::Mat& src); // 对图像进行设置

    void findLights(); // 找灯条获取候选匹配的灯条

    void matchLights(); // 匹配灯条获取候选装甲板

    void chooseTarget(); // 找出优先级最高的装甲板

    bool isLight(Light& light, std::vector<cv::Point>& cnt);//是否是灯条

    // 判断两个灯条是否相邻或者其中一个灯条包含另一个灯条
    bool conTain(cv::RotatedRect& match_rect, std::vector<Light>& Lights, size_t& i, size_t& j);

    int armorGrade(const Armor& checkArmor); // 分级统计装甲板

    void preImplement(Armor& armor); // 预处理装甲板

    bool get_max(const float* data, float& confidence, int& id); // 为单任务网络服务【】【】【？？

    bool get_valid(const float* data, float& confidence, int& id); // 为多任务网络服务【】【】【？？
};

//}

#endif //ARMOR_DETECTION_H
