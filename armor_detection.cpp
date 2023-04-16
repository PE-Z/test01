#include "armor_detection.h"

//#define BINARY_SHOW
//#define DRAW_LIGHTS_CONTOURS
//#define DRAW_LIGHTS_RRT
#define SHOW_NUMROI
//#define DEBUG_DNN_PRINT
//#define DRAW_ARMORS_RRT
//#define DRAW_FINAL_ARMOR_S_CLASS
//#define SHOW_TIME

using namespace cv;
using namespace std;



//���ڻ�ͼ
Mat showSrc = imread("C:/Users/86133/Desktop/�Ӿ�ѧϰ����/�Ӿ�ʵս����/��ɫ����1.jpg");

// ���캯��
ArmorDetector::ArmorDetector()
{
    // ��ʼ����������Ϊ0
    save_num_cnt = 0;

    // ����һ��FileStorage���󣬶�ȡԤ��Ĳ����ļ�
    FileStorage fs("C:/Users/86133/Desktop/new_project/new_detection-main/other/detect_data.yaml", FileStorage::READ);

    // ��fs�ж�ȡ��ֵ����ֵ�������ж�������װ�װ��ж�����
    binThresh = (int)fs["binThresh"];
    light_max_angle = (double)fs["light_max_angle"];
    light_min_hw_ratio = (double)fs["light_min_hw_ratio"];
    light_max_hw_ratio = (double)fs["light_max_hw_ratio"];
    light_min_area_ratio = (double)fs["light_min_area_ratio"];
    light_max_area_ratio = (double)fs["light_max_area_ratio"];
    light_max_area = (double)fs["light_max_area"];
    armor_big_max_wh_ratio = (double)fs["armor_big_max_wh_ratio"];
    armor_big_min_wh_ratio = (double)fs["armor_big_min_wh_ratio"];
    armor_small_max_wh_ratio = (double)fs["armor_small_max_wh_ratio"];
    armor_small_min_wh_ratio = (double)fs["armor_small_min_wh_ratio"];
    armor_max_offset_angle = (double)fs["armor_max_offset_angle"];
    armor_height_offset = (double)fs["armor_height_offset"];
    armor_ij_min_ratio = (double)fs["armor_ij_min_ratio"];
    armor_ij_max_ratio = (double)fs["armor_ij_max_ratio"];
    armor_max_angle = (double)fs["armor_max_angle"];

    // ��fs�ж�ȡװ�װ�ּ�ͳ������Ĳ���
    near_standard = (double)fs["near_standard"];
    height_standard = (double)fs["height_standard"];
    id_grade_ratio = (double)fs["id_grade_ratio"];
    near_grade_ratio = (double)fs["near_grade_ratio"];
    height_grade_ratio = (double)fs["height_grade_ratio"];
    grade_standard = (int)fs["grade_standard"];

    // ��fs�ж�ȡ�����������Ŷ���ֵ�͵з���ɫ
    categories = (int)fs["categories"];
    thresh_confidence = (float)fs["thresh_confidence"];
    enemy_color = 1;

    // �ͷ��ļ���fs
    fs.release();
}

void ArmorDetector::setImage(const Mat& src)
{
    src.copyTo(_src);

    //��ֵ��
    Mat gray;
    cvtColor(_src, gray, COLOR_BGR2GRAY);
    threshold(gray, _binary, binThresh, 255, THRESH_BINARY);
    imshow("_binary", _binary);

#ifdef BINARY_SHOW
    imshow("_binary", _binary);
#endif //BINARY_SHOW
}

bool ArmorDetector::isLight(Light& light, vector<Point>& cnt)
{
    double height = light.height;
    double width = light.width;

    if (height <= 0 || width <= 0) //��͸�Ϊ����
        return false;

    //��һ��Ҫ���ڿ�
    bool standing_ok = height > width;

    //�߿������
    double hw_ratio = height / width;
    bool hw_ratio_ok = light_min_hw_ratio < hw_ratio&& hw_ratio < light_max_hw_ratio;

    //��Ӿ�����������ص����֮������
    double area_ratio = contourArea(cnt) / (height * width);
    //    std::cout<<area_ratio<<std::endl;
    bool area_ratio_ok = light_min_area_ratio < area_ratio&& area_ratio < light_max_area_ratio;

    //�����Ƕ�����
    bool angle_ok = fabs(90.0 - light.angle) < light_max_angle;
    // cout<<"angle: "<<light.angle<<endl;

    //�����������
    bool area_ok = contourArea(cnt) < light_max_area;
    //�����жϵ������ܼ�

    //�����жϵ������ܼ�
    bool is_light = hw_ratio_ok && area_ratio_ok && angle_ok && standing_ok && area_ok;

    if (!is_light)
    {
        //        cout<<hw_ratio<<"    "<<contourArea(cnt) / light_max_area<<"    "<<light.angle<<endl;
    }

    return is_light;
}

void ArmorDetector::findLights()
{
    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> hierarchy;
    cv::findContours(_binary, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);//Ѱ������

#ifdef DRAW_LIGHTS_CONTOURS
    for (int i = 0; i < contours.size(); i++)
        cv::drawContours(showSrc, contours, i, Scalar(255, 0, 0), 2, LINE_8);
    imshow("showSrc", showSrc);
#endif

    if (contours.size() < 2)//ͼ��������������
    {
        //        printf("no 2 contours\n");
        return;
    }

    for (auto& contour : contours)//��������
    {
        RotatedRect r_rect = minAreaRect(contour);
        Light light = Light(r_rect);

        if (isLight(light, contour))
        {
            //            cout<<"is_Light   "<<endl;
            cv::Rect rect = r_rect.boundingRect();//boundingRect���������Σ����ڱ�����ȡ��ɫ

            //ȷ�� ������װ��ͼ�� ������������
            if (0 <= rect.x && 0 <= rect.width && rect.x + rect.width <= _src.cols &&
                0 <= rect.y && 0 <= rect.height && rect.y + rect.height <= _src.rows)
            {
                int sum_r = 0, sum_b = 0;
                cv::Mat roi = _src(rect);//��ȡ�������������Σ�������ȡ��ɫ
                // Iterate through the ROI
                for (int i = 0; i < roi.rows; i++)
                {
                    for (int j = 0; j < roi.cols; j++)
                    {
                        if (cv::pointPolygonTest(contour, cv::Point2f(j + rect.x, i + rect.y), false) >= 0) // ֻ���������е�����������
                        {
                            sum_r += roi.at<cv::Vec3b>(i, j)[2];
                            sum_b += roi.at<cv::Vec3b>(i, j)[0];
                        }
                    }
                }
                //                 std::cout<<sum_r<<"           "<<sum_b<<std::endl;
                                // Sum of red pixels > sum of blue pixels ?
                light.lightColor = sum_r > sum_b ? RED : BLUE;//������ɫ

                // ��ɫ�����ϵ�ط��ľͲ�����

                if (light.lightColor == 1)//��ɫ1��ɫ0
                {
                    candidateLights.emplace_back(light);//�����ѡ����

                    //��������
                    Point2f vertice_lights[4];
                    light.points(vertice_lights);
                    for (int i = 0; i < 4; i++) {
                        line(showSrc, vertice_lights[i], vertice_lights[(i + 1) % 4], CV_RGB(255, 0, 0), 2, LINE_8);
                    }
                    //circle(showSrc,light.center,5,Scalar(0,0,0),-1);
                    imshow("showSrc1", showSrc);


#ifdef DRAW_LIGHTS_RRT
                    Point2f vertice_lights[4];
                    light.points(vertice_lights);
                    for (int i = 0; i < 4; i++) {
                        line(showSrc, vertice_lights[i], vertice_lights[(i + 1) % 4], CV_RGB(255, 0, 0), 2, LINE_8);
                    }
                    //circle(showSrc,light.center,5,Scalar(0,0,0),-1);
                    imshow("showSrc", showSrc);
#endif //DRAW_LIGHTS_RRT
                }
            }
        }
    }
    //cout<<"dengtiao  geshu:  "<<candidateLights.size()<<endl;
}

void ArmorDetector::matchLights()//ƥ�����
{
    if (candidateLights.size() < 2)//��ѡ������������
    {
        //        printf("no 2 lights\n");
        return;
    }

    // ����ת���δ���������
    sort(candidateLights.begin(), candidateLights.end(),
        [](RotatedRect& a1, RotatedRect& a2) {
            return a1.center.x < a2.center.x; });

    //��������I
    for (size_t i = 0; i < candidateLights.size() - 1; i++)
    {
        Light lightI = candidateLights[i];
        Point2f centerI = lightI.center;

        //��������J
        for (size_t j = i + 1; j < candidateLights.size(); j++)
        {
            Light lightJ = candidateLights[j];
            Point2f centerJ = lightJ.center;
            double armorWidth = POINT_DIST(centerI, centerJ) - (lightI.width + lightJ.width) / 2.0;//װ�װ��
            double armorHeight = (lightI.height + lightJ.height) / 2.0;//װ�װ��
            double armor_ij_ratio = lightI.height / lightJ.height;//�����߶ȱ�
            double armorAngle = atan2((centerI.y - centerJ.y), fabs(centerI.x - centerJ.x)) / CV_PI * 180.0;//װ�װ�Ƕ�

            //��߱�ɸѡ����
            bool small_wh_ratio_ok = armor_small_min_wh_ratio < armorWidth / armorHeight && armorWidth / armorHeight < armor_small_max_wh_ratio;
            bool big_wh_ratio_ok = armor_big_min_wh_ratio < armorWidth / armorHeight && armorWidth / armorHeight < armor_big_max_wh_ratio;
            bool wh_ratio_ok = small_wh_ratio_ok || big_wh_ratio_ok;

            //���ҵ����ǶȲ�ɸѡ����
            bool angle_offset_ok = fabs(lightI.angle - lightJ.angle) < armor_max_offset_angle;

            //�������������ĵ�߶Ȳ�ɸѡ����
            bool height_offset_ok = fabs(lightI.center.y - lightJ.center.y) / armorHeight < armor_height_offset;

            //���ҵ����ĸ߶ȱ�
            bool ij_ratio_ok = armor_ij_min_ratio < armor_ij_ratio&& armor_ij_ratio < armor_ij_max_ratio;

            //��ѡװ�װ�Ƕ�ɸѡ����
            bool angle_ok = fabs(armorAngle) < armor_max_angle;

            //��������
            bool is_like_Armor = wh_ratio_ok && angle_offset_ok && height_offset_ok && ij_ratio_ok && angle_ok;

            if (is_like_Armor)
            {
                //origin
                Point2f armorCenter = (centerI + centerJ) / 2.0;//װ�װ����ĵ�
                RotatedRect armor_rrect = RotatedRect(armorCenter,//����װ�װ����ת����
                    Size2f(armorWidth, armorHeight),
                    -armorAngle);

                Point2f pt4[4] = { lightI.bottom, lightJ.bottom, lightJ.top, lightI.top };//��˳�����װ�װ��ĸ���

                if (!conTain(armor_rrect, candidateLights, i, j))//�ж��Ƿ��� �����ҵ�����conTain�� �����
                {
                    Armor armor(armor_rrect);  //����װ�װ�

                    for (int index = 0; index < 4; index++)  //װ�װ���ĸ���
                        armor.armor_pt4[index] = pt4[index];

                    if (small_wh_ratio_ok)  //��߱��ж��Ǵ�װ�װ廹��Сװ�װ�
                        armor.type = SMALL;
                    else
                        armor.type = BIG;

                    preImplement(armor);// put mat into numROIs����������������������������������������������������������

                    candidateArmors.emplace_back(armor);//�����ѡװ�װ�

                    //����װ�װ�
                    Point2f vertice_armors[4];
                    armor.points(vertice_armors);
                    for (int m = 0; m < 4; m++)
                    {
                        line(showSrc, vertice_armors[m], vertice_armors[(m + 1) % 4], CV_RGB(0, 255, 255), 2, LINE_8);
                    }
                    //circle(showSrc,armorCenter,15,Scalar(0,255,255),-1);
                    imshow("showSrc2", showSrc);
                    putText(showSrc, to_string(armorAngle), armor.armor_pt4[3], FONT_HERSHEY_COMPLEX, 1.0, Scalar(0, 255, 255), 2, 8);

#ifdef DRAW_ARMORS_RRT
                    //cout<<"LightI_angle :   "<<lightI.angle<<"   LightJ_angle :   "<<lightJ.angle<<"     "<<fabs(lightI.angle - lightJ.angle)<<endl;
                    //cout<<"armorAngle   :   "<<armorAngle * 180 / CV_PI <<endl;
                    //cout<<"    w/h      :   "<<armorWidth/armorHeight<<endl;
                    //cout<<"height-offset:   "<<fabs(lightI.height - lightJ.height) / armorHeight<<endl;
                    //cout<<" height-ratio:   "<<armor_ij_ratio<<endl;

                    Point2f vertice_armors[4];
                    armor.points(vertice_armors);
                    for (int m = 0; m < 4; m++)
                    {
                        line(showSrc, vertice_armors[m], vertice_armors[(m + 1) % 4], CV_RGB(0, 255, 255), 2, LINE_8);
                    }
                    //circle(showSrc,armorCenter,15,Scalar(0,255,255),-1);
                    imshow("showSrc", showSrc);
                    putText(showSrc, to_string(armorAngle), armor.armor_pt4[3], FONT_HERSHEY_COMPLEX, 1.0, Scalar(0, 255, 255), 2, 8);
#endif //DRAW_ARMORS_RRT
                }
            }

        }
    }
}

void ArmorDetector::chooseTarget()//ѡ��Ŀ��
{

    if (candidateArmors.empty())//��û�к�ѡװ�װ�
    {
        //cout<<"no target!!"<<endl;
//        finalArmor = Armor();
        return;
    }
    else if (candidateArmors.size() == 1)//��1����ѡװ�װ�
    {
        cout << "get 1 target!!" << endl;
        Mat out_blobs = dnnDetect.net_forward(numROIs);//����������������������ʶ�𲿷�

        float* outs = (float*)out_blobs.data;
        if (get_max(outs, candidateArmors[0].confidence, candidateArmors[0].id))
        {
#ifdef SHOW_NUMROI
            cv::Mat numDst;//����ROI
            resize(numROIs[0], numDst, Size(200, 300));
            string name = to_string(candidateArmors[0].id) + ":" + to_string(candidateArmors[0].confidence * 100) + "%";
            //        printf("%d",armor.id);
            imshow("name", numDst);//����������������������ʶ�𲿷�

            //        std::cout<<"number:   "<<armor.id<<"   type:   "<<armor.type<<std::endl;
            //        string file_name = "../data/"+std::to_string(0)+"_"+std::to_string(cnt_count)+".jpg";
            //        cout<<file_name<<endl;
            //        imwrite(file_name,numDst);
            //        cnt_count++;
#endif
            candidateArmors[0].grade = 100;//�÷ֳ�ʼ��Ϊ100
            finalArmors.emplace_back(candidateArmors[0]);//��������װ�װ�
        }
    }
    else
    {
        //cout<<"get "<<candidateArmors.size()<<" target!!"<<endl;

        // dnn implement
        Mat out_blobs = dnnDetect.net_forward(numROIs);//����������������������ʶ�𲿷�
        float* outs = (float*)out_blobs.data;

        // ��ȡÿ����ѡװ�װ��id��type
        for (int i = 0; i < candidateArmors.size(); i++) {
            // numROIs has identical size as candidateArmors
            if (!get_max(outs, candidateArmors[i].confidence, candidateArmors[i].id))
            {
                outs += categories;
                continue;
            }
#ifdef SHOW_NUMROI
            cv::Mat numDst;//����ROI
            resize(numROIs[i], numDst, Size(200, 300));
            string name = to_string(candidateArmors[i].id) + ":" + to_string(candidateArmors[i].confidence * 100) + "%";
            //        printf("%d",armor.id);
            imshow("name", numDst);//����������������������ʶ�𲿷�
            //        std::cout<<"number:   "<<armor.id<<"   type:   "<<armor.type<<std::endl;
            //        string file_name = "../data/"+std::to_string(0)+"_"+std::to_string(cnt_count)+".jpg";
            //        cout<<file_name<<endl;
            //        imwrite(file_name,numDst);
            //        cnt_count++;
#endif
            // װ�װ����ĵ�����Ļ���Ĳ��֣������Ĳ�����������б��С�ģ�
            // ��α���Ƶ���л�Ŀ�꣺��С���ο���Ǹ��ٵ��ˣ�һ���������Ŀ�궪ʧ��
            // UI����������ѡ��ѡ�����Ǽ��ţ��������л����鷳���������飩

            //�����ɸѡװ�װ����ȼ�(�����������ֻ���������ȼ�����2��4��id���ȼ���������Щ���࣡������)
            /*������ȼ�����ʶ��Ӣ��1��װ�װ壬���3��4�ţ������ֵĻ�1��100��3��4��80������������
             *1����߱ȣ�ɸѡ����Ͳ���װ�װ壬����������װ�װ壩
             *2��װ�װ忿��ͼ������
             *3��װ�װ���б�Ƕ���С
             *4��װ�װ�����
             */
             //1����߱���һ����׼ֵ�͵�ǰֵ����ֵ�����ڱ�׼ֵĬ����Ϊ1�����ϱ�׼������Ϊ�÷�
             //2������Сroi�ھ͸��֣����ڲ����֣�����ռ�Ƚϵͣ�
             //3��90�ȼ�ȥװ�װ�ĽǶȳ���90�õ���ֵ���ϱ�׼����Ϊ�÷�
             //4����ǰ�������֮ǰ��װ�װ���и��ɴ�С���򣬻�ȡ�����СֵȻ���һ�����ù�һ���ĸ߶�ֵ���ϱ�׼����Ϊ�÷�

            candidateArmors[i].grade = armorGrade(candidateArmors[i]);//armorGrade����װ�װ����

            if (candidateArmors[i].grade > grade_standard)//�÷ִ������Ҫ��
            {
                finalArmors.emplace_back(candidateArmors[i]);//��������װ�װ�
            }
            outs += categories;
        }
    }

    Mat showSrc;
    _src.copyTo(showSrc);
    // std::cout<<"final_armors_size:   "<<finalArmors.size()<<std::endl;
    //����װ�װ����
    for (size_t i = 0; i < finalArmors.size(); i++)
    {
        //        Point2f armor_pts[4];
        //        finalArmors[i].points(armor_pts);
        for (int j = 0; j < 4; j++)//����װ�װ�
        {
            line(showSrc, finalArmors[i].armor_pt4[j], finalArmors[i].armor_pt4[(j + 1) % 4], CV_RGB(255, 255, 0), 2);
        }

        // ��ȡ��ǰװ�װ�����ֺ���Ϣ������Ϣ���ַ�����ʾ
        double ff = finalArmors[i].grade;
        string information = to_string(finalArmors[i].id) + ":" + to_string(finalArmors[i].confidence * 100) + "%";

        // ��װ�װ���Ϣ�����ֻ��Ƶ�ͼ����
        putText(showSrc, information, finalArmors[i].armor_pt4[3], FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 255));
    }

    // ���finalArmors�������п��õ�װ�װ壬���ڴ�������ʾ�����
    if (!finalArmors.empty())
        imshow("showSrc", showSrc);

#ifdef DRAW_FINAL_ARMOR_S_CLASS
    Mat showSrc;
    _src.copyTo(showSrc);
    // std::cout<<"final_armors_size:   "<<finalArmors.size()<<std::endl;

    for (size_t i = 0; i < finalArmors.size(); i++)
    {
        //        Point2f armor_pts[4];
        //        finalArmors[i].points(armor_pts);
        for (int j = 0; j < 4; j++)
        {
            line(showSrc, finalArmors[i].armor_pt4[j], finalArmors[i].armor_pt4[(j + 1) % 4], CV_RGB(255, 255, 0), 2);
        }

        double ff = finalArmors[i].grade;
        string information = to_string(finalArmors[i].id) + ":" + to_string(finalArmors[i].confidence * 100) + "%";
        //        putText(final_armors_src,ff,finalArmors[i].center,FONT_HERSHEY_COMPLEX, 1.0, Scalar(12, 23, 200), 1, 8);
        putText(showSrc, information, finalArmors[i].armor_pt4[3], FONT_HERSHEY_COMPLEX, 1, Scalar(255, 0, 255));
    }
    if (!finalArmors.empty())
        imshow("showSrc", showSrc);
#endif //DRAW_FINAL_ARMOR_S_CLASS
}

//���麯��
vector<Armor> ArmorDetector::autoAim(const cv::Mat& src)
{
    //��ʼ������ROI������װ�װ壬��ѡװ�װ壬��ѡ����
    if (!numROIs.empty())numROIs.clear();
    if (!finalArmors.empty())finalArmors.clear();
    if (!candidateArmors.empty())candidateArmors.clear();
    if (!candidateLights.empty())candidateLights.clear();

    //do autoaim task
#ifdef SHOW_TIME
    auto start = std::chrono::high_resolution_clock::now();
    setImage(src);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = seconds_duration(end - start).count();
    printf("set_time:%lf\n", duration);
    start = std::chrono::high_resolution_clock::now();
    findLights();
    end = std::chrono::high_resolution_clock::now();
    duration = seconds_duration(end - start).count();
    printf("light_time:%lf\n", duration);
    start = std::chrono::high_resolution_clock::now();
    matchLights();
    end = std::chrono::high_resolution_clock::now();
    duration = seconds_duration(end - start).count();
    printf("match_time:%lf\n", duration);
    start = std::chrono::high_resolution_clock::now();
    chooseTarget();
    end = std::chrono::high_resolution_clock::now();
    duration = seconds_duration(end - start).count();
    printf("choose_time:%lf\n", duration);
#else

    setImage(src); //ͼ����
    findLights(); //���ҵ���
    matchLights(); //ƥ�����
    chooseTarget(); //ѡ��Ŀ��װ�װ�
#endif

    return finalArmors; //���ؼ�⵽��Ŀ��װ�װ�
}

//�ж��Ƿ����װ�װ��л��б�ĵ��������
bool ArmorDetector::conTain(RotatedRect& match_rect, vector<Light>& Lights, size_t& i, size_t& j)
{
    Rect matchRoi = match_rect.boundingRect();
    //�������ҵ����м�ĵ���������У���װ�װ��Ƿ�����˵�����
    for (size_t k = i + 1; k < j; k++)
    {
        // ������ȷ�λ�õĵ�
        if (matchRoi.contains(Lights[k].top) ||
            matchRoi.contains(Lights[k].bottom) ||
            matchRoi.contains(Point2f(Lights[k].top.x + Lights[k].height * 0.25, Lights[k].top.y + Lights[k].height * 0.25)) ||
            matchRoi.contains(Point2f(Lights[k].bottom.x - Lights[k].height * 0.25, Lights[k].bottom.y - Lights[k].height * 0.25)) ||
            matchRoi.contains(Lights[k].center))
        {
            return true;//�����˾ͷ���true��װ�װ�������
        }
        else
        {
            continue;
        }
    }
    return false;
}

//����ʶ��Ԥ����
void ArmorDetector::preImplement(Armor& armor)
{
    Mat numDst; //�������򣨵����м䲿�֣���ͼ��

    // ��������
    const int light_length = 14;
    // ͸�ӱ任�ĸ߶ȺͿ��
    const int warp_height = 30;
    const int small_armor_width = 32;//Сװ�װ��� Ϊ48/3*2
    const int large_armor_width = 44;//��װ�װ��� Ϊ70/3*2
    // ����ROI����Ĵ�С
    const cv::Size roi_size(22, 30);

    // �������µ�����͸�ӱ任�������
    const int top_light_y = (warp_height - light_length) / 2;//��������y����
    const int bottom_light_y = top_light_y + light_length;//�����ײ�y����
    const int warp_width = armor.type == SMALL ? small_armor_width : large_armor_width;//������

    // Ŀ��װ�װ���ĸ��˵�
    cv::Point2f target_vertices[4] = {
    cv::Point(0, bottom_light_y),
    cv::Point(warp_width, bottom_light_y),
    cv::Point(warp_width, top_light_y),
    cv::Point(0, top_light_y),
    };
    // ����͸�ӱ任����
    const Mat& rotation_matrix = cv::getPerspectiveTransform(armor.armor_pt4, target_vertices);

    // ͸�ӱ任
    cv::warpPerspective(_src, numDst, rotation_matrix, cv::Size(warp_width, warp_height));

    // ��ȡ��������ROI
    numDst = numDst(cv::Rect(cv::Point((warp_width - roi_size.width) / 2, 0), roi_size));

    // ���������������������ʶ��
    dnnDetect.img_processing(numDst, numROIs);

    // save number roi
//     int c = waitKey(100);
//     cvtColor(numDst, numDst, cv::COLOR_BGR2GRAY);
//     threshold(numDst, numDst, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
//     string nn= std::to_string(save_num_cnt);
//     string path="/home/lmx/data_list/"+nn+".jpg";
//     if(c==113){
////
//         imwrite(path,numDst);
//         save_num_cnt++;
//     }

//    resize(numDst, numDst,Size(200,300));
//    cvtColor(numDst, numDst, cv::COLOR_BGR2GRAY);
//    threshold(numDst, numDst, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
//    string name = to_string(armor.id) + ":" + to_string(armor.confidence*100) + "%";
//    imshow("name", numDst);
    // std::cout<<"number:   "<<armor.id<<"   type:   "<<armor.type<<std::endl;

}

//����ʶ�𲿷�
bool ArmorDetector::get_max(const float* data, float& confidence, int& id)
{
    confidence = data[0];
    id = 0;
    for (int i = 0; i < categories; i++)
    {
        if (data[i] > confidence)
        {
            confidence = data[i];
            id = i;
        }
    }
    if (id == 0 || id == 2 || confidence < thresh_confidence)
        return false;
    else
        return true;
}

int ArmorDetector::armorGrade(const Armor& checkArmor)
{
    // ��������ϵͳ��ͨ�Ż��ƣ��״�������淶

    // ѡ����int�ض�double

    /////////id���ȼ������Ŀ////////////////////////
    int id_grade;
    int check_id = checkArmor.id;
    id_grade = check_id == 1 ? 100 : 80;
    ////////end///////////////////////////////////

    /////////���װ�װ������Ŀ/////////////////////
    // ���װ�װ壬���������һ����׼ֵ���̶����루����3/4�ף���װ�װ��С��Armor.area����Լ�Ƕ��٣��ִ�Сװ�װ壩
    // �ȱ�׼�����100��С����������������������С�ĵó�����ֵ���С
    int height_grade;
    double hRotation = checkArmor.size.height / height_standard;//װ�װ峤�����׼���ȵı�ֵ
    if (candidateArmors.size() == 1)  hRotation = 1;
    height_grade = hRotation * 60;
    //////////end/////////////////////////////////

    ////////����ͼ�����Ĵ����Ŀ//////////////////////
    // �������ģ������������룬�趨��׼ֵ����ͼ��������ͷ�����Ļ���Ĳ���
    int near_grade;
    double pts_distance = POINT_DIST(checkArmor.center, Point2f(_src.cols * 0.5, _src.rows * 0.5));//װ�װ�������ͼ�����ľ���
    near_grade = pts_distance / near_standard < 1 ? 100 : (near_standard / pts_distance) * 100;
    ////////end//////////////////////////////////

    // �����ϵ������ϸ���ڣ�
    int final_grade = id_grade * id_grade_ratio +
        height_grade * height_grade_ratio +
        near_grade * near_grade_ratio;

    //    std::cout<<id_grade<<"   "<<height_grade<<"   "<<near_grade<<"    "<<final_grade<<std::endl;
    //	std::cout<<"final_grade"<<std::endl;

    return final_grade;
}

//����ʶ�𲿷�
bool ArmorDetector::get_valid(const float* data, float& confidence, int& id)
{
    id = 1;
    int i = 2;
    confidence = data[i];
    for (; i < categories; i++)
    {
        if (data[i] > confidence)
        {
            confidence = data[i];
            id = i - 1;
        }
    }
    if (data[0] > data[1] || id == 2 || confidence < thresh_confidence)
        return false;
    else
        return true;
}

//}
