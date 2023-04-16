#pragma once
#ifndef ARMOR_DETECTION_H
#define ARMOR_DETECTION_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
//#include "robot_status.h"
#include "number_dnn.h"
#include <iostream>

#define POINT_DIST(p1,p2) std::sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y))//������

enum EnermyColor { RED = 0, BLUE = 1 };//��ɫ
enum EnermyType { SMALL = 0, BIG = 1 };//װ�װ����С


struct Light : public cv::RotatedRect     //�����ṹ�壬�̳�RotatedRect
{
    Light() = default;
    explicit Light(cv::RotatedRect& box) : cv::RotatedRect(box)
    {
        cv::Point2f p[4];//�㼯��Ŷ˵�
        box.points(p);//����RotatedRect���ĸ����ʼ���㼯
        std::sort(p, p + 4, [](const cv::Point2f& a, const cv::Point2f& b) {return a.y < b.y; });//��y�������С��������
        top = (p[0] + p[1]) / 2;//�϶˵�����
        bottom = (p[2] + p[3]) / 2;//�¶˵�����
        height = POINT_DIST(top, bottom);//�����ĳ���
        width = POINT_DIST(p[0], p[1]);//�����Ŀ��

        angle = top.x < bottom.x ? box.angle : 90 + box.angle;//�����ĽǶȣ�Ҫ�������¶˵㵹��
        if (fabs(bottom.x - top.x) <= 0.01) angle = 90;//������¶˵�x������С������Ϊ�Ƕ�Ϊ90
    }
    int lightColor;//��������ɫ
    cv::Point2f top;//�϶˵�����
    cv::Point2f bottom;//�¶˵�����
    double angle;//�����ĽǶ�
    double height;//�����ĳ���
    double width;//�����Ŀ��

};

//װ�װ�ṹ��
struct Armor : public cv::RotatedRect    //װ�װ�ṹ��̳�RotatedRect
{
    Armor() = default;
    explicit Armor(cv::RotatedRect& box) : cv::RotatedRect(box)
    {
        confidence = 0;  //���Ŷ�
        id = 0;  //װ�װ�����
        type = SMALL;  //װ�װ�����
        grade = 0;  //�ȼ�
    }

    cv::Point2f armor_pt4[4]; //���½ǿ�ʼ��ʱ��
    //        std::vector<cv::Point2f> armor_pt4; //���½ǿ�ʼ��ʱ��
    float confidence;//���Ŷ�
    int id;  // װ�װ����
    int grade;//װ�װ�÷�
    int type;  // װ�װ�����
    //Eigen::Vector3d world_position;  // ��ǰ����ʵ����
    //Eigen::Vector3d camera_position;  // ��ǰ����ʵ����
    //    int area;  // װ�װ����
};



// ���࣬ʵ��Ŀ����
class ArmorDetector
{
public:
    ArmorDetector(); // ���캯����ʼ��

    std::vector<Armor> autoAim(const cv::Mat& src); // ������Ŀ�������ת��������ͷԭ��С��
private:
    int save_num_cnt;//�������š���������������������
    int binThresh; // ��ֵ����ֵ
    int enemy_color; // �з���ɫ
    int categories; // ������

    // �����ж�����
    double light_max_angle; //�������ƫת��
    double light_min_hw_ratio;//��߱�
    double light_max_hw_ratio;
    double light_min_area_ratio;// ��ת���κ;��α���
    double light_max_area_ratio;
    double light_max_area;//���Ƶ����������

    // װ�װ��ж�����
    double armor_big_max_wh_ratio;//��װ�װ��߱�
    double armor_big_min_wh_ratio;
    double armor_small_max_wh_ratio;//Сװ�װ��߱�
    double armor_small_min_wh_ratio;
    double armor_max_angle;//װ�װ����ƫת��
    double armor_height_offset;
    double armor_ij_min_ratio;//���ҵ����ĸ߶ȱ�
    double armor_ij_max_ratio;
    double armor_max_offset_angle;//���ҵ����ǶȲ�

    // װ�װ�ּ�ͳ������Ĳ���
    double near_standard;//����ͼ�����Ĵ����ֵ
    int height_standard;//װ�װ��׼�߶�
    int grade_standard;//�����
    double id_grade_ratio;//id�ȼ����ʣ���ϵ�������ܷ֣�
    double height_grade_ratio;//�߶ȵȼ�����
    double near_grade_ratio;//�ӽ��ȼ�����

    float thresh_confidence; // ���Ŷ���ֵ

    cv::Mat _src; // �ü�src���ROI
    cv::Mat _binary;
    std::vector<cv::Mat> temps;//������������������

    Armor lastArmor;

    std::vector<Light> candidateLights; // ɸѡ�ĵ���
    std::vector<Armor> candidateArmors; // ɸѡ��װ�װ�
    std::vector<Armor> finalArmors;//����װ�װ�
    std::vector<cv::Mat> numROIs;//id��ROI����
    Armor finalArmor; // ����װ�װ�

    DNN_detect dnnDetect;//��������ʶ��

    void setImage(const cv::Mat& src); // ��ͼ���������

    void findLights(); // �ҵ�����ȡ��ѡƥ��ĵ���

    void matchLights(); // ƥ�������ȡ��ѡװ�װ�

    void chooseTarget(); // �ҳ����ȼ���ߵ�װ�װ�

    bool isLight(Light& light, std::vector<cv::Point>& cnt);//�Ƿ��ǵ���

    // �ж����������Ƿ����ڻ�������һ������������һ������
    bool conTain(cv::RotatedRect& match_rect, std::vector<Light>& Lights, size_t& i, size_t& j);

    int armorGrade(const Armor& checkArmor); // �ּ�ͳ��װ�װ�

    void preImplement(Armor& armor); // Ԥ����װ�װ�

    bool get_max(const float* data, float& confidence, int& id); // Ϊ������������񡾡�����������

    bool get_valid(const float* data, float& confidence, int& id); // Ϊ������������񡾡�����������
};

//}

#endif //ARMOR_DETECTION_H
