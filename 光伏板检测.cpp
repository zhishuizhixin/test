//解决tagetlist中漏识别后又重新识别到的问题。以Vy为主排序可解决。
//发现4中targetList传参有问题
//解决了从右往左飞行的识别问题
#include <algorithm>
#include<fstream>
#include<vector>
#include <opencv2/opencv.hpp>
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv2\imgproc\imgproc.hpp>
using namespace std;
using namespace cv;
typedef struct BLOCK
{
	int ID;
	cv::Rect LastLocal;
	bool flag;
	float score;
}BLOCK;

bool cmp_Score(const BLOCK &a, const BLOCK &b)
{
	return a.score > b.score;
}

//vector<BLOCK>::iterator find_target(vector<BLOCK> src, BLOCK dst)
//{
//	for (vector<BLOCK>::iterator it = src.begin(); it != src.end(); it++)
//	{
//		if (it->ID == dst.ID)
//		{
//			return it;
//		}
//		
//	}
//}
//void find_target(vector<BLOCK> src, BLOCK dst, vector<BLOCK>::iterator& it)
//{
//	for (it = src.begin(); it != src.end(); it++)
//	{
//		if (it->ID == dst.ID)
//		{
//			break;
//		}
//
//	}
//}


void Sort_Taget_By_ID(vector<BLOCK>& src)
{
	vector<BLOCK> temp = src;
	sort(temp.begin(), temp.end(), cmp_Score);
	//vector<BLOCK>::iterator result = find(src.begin(), src.end(), temp[0]);
	//vector<BLOCK>::iterator result = find_target(src, temp[0]);
	vector<BLOCK>::iterator result;
	BLOCK A_temp = *(temp.begin());
	for (result = src.begin(); result != src.end(); result++)
	{
		if (result->ID == A_temp.ID)
		{
			break;
		}

	}
	//find_target(src, A_temp, result);


	int ID_Increse = result->ID;
	int ID_Decrese = result->ID;

	if (result != src.begin())
	{
		for (vector<BLOCK>::iterator it = result - 1; it != src.begin()-1; it--)
		{
			it->ID = --ID_Decrese;
		}
	}

	if (result != src.end())
	{
		for (vector<BLOCK>::iterator it = result + 1; it != src.end()+1; it++)
		{
			it->ID = ++ID_Increse;
		}
	}


}

float GetOverlap(const Rect& box1, const Rect& box2, float &rate1, float &rate2)
{
	if (box1.x > box2.x + box2.width) { rate1 = 0; rate2 = 0; return 0.0; }
	if (box1.y > box2.y + box2.height) { rate1 = 0; rate2 = 0; return 0.0; }
	if (box1.x + box1.width < box2.x) { rate1 = 0; rate2 = 0; return 0.0; }
	if (box1.y + box1.height < box2.y) { rate1 = 0; rate2 = 0; return 0.0; }
	float colInt = min(box1.x + box1.width, box2.x + box2.width) - max(box1.x, box2.x);
	float rowInt = min(box1.y + box1.height, box2.y + box2.height) - max(box1.y, box2.y);
	float intersection = colInt * rowInt;
	rate1 = intersection / (box1.width*box1.height);
	rate2 = intersection / (box2.width*box2.height);
	return intersection;
}

bool cmp_y(const Rect &a, const Rect &b)
{
	return a.y> b.y;
}
bool cmp_x(const Rect &a, const Rect &b)
{
	return a.x> b.x;
}
bool cmp_target_x(const BLOCK &a, const BLOCK &b)
{
	return a.LastLocal.x> b.LastLocal.x;
}

int main()
{

	//VideoCapture capture("c:/DJI_0314.mp4");
	VideoCapture capture("D:/100MEDIA/DJI_0314.mp4");
	//VideoCapture capture("C:/Users/lenovo/AppData/Roaming/PotPlayerMini/Capture/example.mkv");

	int FrameNum = 0;
	vector<BLOCK> targetList0;
	vector<BLOCK> targetList1;

	vector<Rect> newTarget;
	vector<Rect> Vy[2];
	Mat preframe;
	int ID_0 = 0;
	int ID_1 = 0;
	int iFalg = 0;

	while (1)
	{
		iFalg++;
		if (iFalg>3)
		{
			iFalg = 3;
		}
		//frame存储每一帧图像
		Mat frame;
		//读取当前帧
		capture >> frame;
		//播放完退出
		if (frame.empty()) {

			break;
		}
		if (FrameNum == 0)
		{
			FrameNum++;
			frame.copyTo(preframe);
			continue;
		}
		FrameNum++;

		//匹配初始偏差量
		Mat preGray, Gray;

		cvtColor(frame, Gray, CV_BGR2GRAY);
		cvtColor(preframe, preGray, CV_BGR2GRAY);
		/*imshow("1", frame);
		imshow("2", preframe);*/

		Mat templateImage;
		Gray(Rect(80, 80, frame.cols - 160, frame.rows - 160)).copyTo(templateImage);
		Mat result;
		int result_cols = preframe.cols - templateImage.cols + 1;
		int result_rows = preframe.rows - templateImage.rows + 1;
		if (result_cols < 0 || result_rows < 0)
		{
			break;
		}
		result.create(result_cols, result_rows, CV_32FC1);

		matchTemplate(preGray, templateImage, result, TM_CCOEFF_NORMED);
		double minVal = -1;
		double maxVal;
		Point minLoc;
		Point maxLoc;
		Point matchLoc;
		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
		matchLoc = maxLoc;

		matchLoc.x -= 80;
		matchLoc.y -= 80;

		cout << matchLoc.x << endl;

		vector<Mat> RGB;
		split(frame, RGB);

		Mat temp_th;
		blur(RGB[1], temp_th, Size(51, 51));
		/*imshow("1", RGB[1]);
		imshow("2", temp_th);*/

		temp_th = temp_th - 130;
		temp_th = temp_th + 130;


		Mat temp = RGB[1] - temp_th;
		//imshow("3", temp);
		Mat bin;
		threshold(temp, bin, 10, 255, CV_THRESH_BINARY);
		//imshow("4", bin);
		Mat kel = getStructuringElement(MORPH_RECT, Size(3, 11));
		erode(bin, bin, kel);
		//imshow("5", bin);

		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(bin, contours, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0)); /// 计算矩

		Vy[0].clear();
		Vy[1].clear();
		newTarget.clear();

		for (int i = 0; i < contours.size(); i++)
		{
			if (contours[i].size() > 70)
			{
				Rect r0 = cv::boundingRect(Mat(contours[i]));
				if (r0.width > 20 && r0.height > 20 && r0.width < 50 && r0.height < 60)
				{
					newTarget.push_back(r0);

				}
			}

		}
		//
		sort(newTarget.begin(), newTarget.end(), cmp_y);
		int j = -1;
		for (int i = 0; i<2; i++)
		{
			if (j == newTarget.size() - 1)
				break;
			Vy[i].push_back(newTarget[++j]);
			for (j; j<newTarget.size(); j++)
			{
				if (abs(newTarget[j + 1].y - newTarget[j].y) < 40)
					Vy[i].push_back(newTarget[j + 1]);
				else

					break;
			}
		}
		sort(Vy[0].begin(), Vy[0].end(), cmp_x);
		sort(Vy[1].begin(), Vy[1].end(), cmp_x);
		//
		if (targetList0.size() == 0)
		{
			for (int i = 0; i < Vy[0].size(); i++)
			{
				BLOCK temp;
				temp.ID = ID_0;
				temp.LastLocal = Vy[0][i];
				temp.flag = 0;
				targetList0.push_back(temp);
				ID_0++;
			}
		}
		else
		{
			for (int i = 0; i < targetList0.size(); i++)
			{
				targetList0[i].LastLocal.x -= matchLoc.x;
				targetList0[i].LastLocal.y -= matchLoc.y;
			}
			for (int i = 0; i < targetList0.size(); i++)
			{
				targetList0[i].flag = 0;
				int j;
				for (j = 0; j < Vy[0].size(); j++)
				{
					float rate1, rate2;
					GetOverlap(targetList0[i].LastLocal, Vy[0][j], rate1, rate2);
					//锁定同一个Rect
					if (rate1 > 0.7&&rate2 > 0.7)
					{
						targetList0[i].LastLocal = Vy[0][j];
						//是同一个flag置1
						targetList0[i].flag = 1;
						targetList0[i].score = rate1 * rate2;
						Vy[0].erase(Vy[0].begin() + j);
						break;
					}
				}
			}

			for (int i = 0; i < targetList0.size(); i++)
			{
				if (targetList0[i].flag == 0)
				{
					targetList0.erase(targetList0.begin() + i);
				}
			}
			if (Vy[0].size() != 0)
			{
				for (int i = 0; i < Vy[0].size(); i++)
				{
					BLOCK temp;
					temp.ID = ID_0;
					temp.LastLocal = Vy[0][i];
					temp.flag = 0;
					targetList0.push_back(temp);
					ID_0++;
				}
			}
		}
		if (targetList0.size() == 19)
		{
			cout << "llllllllllll" << endl;
		}
		sort(targetList0.begin(), targetList0.end(), cmp_target_x);

		if (iFalg > 2)
		{
			Sort_Taget_By_ID(targetList0);
		}




		//////////////////////判断第二行//////////////////////////
		if (targetList1.size() == 0)
		{
			for (int i = 0; i < Vy[1].size(); i++)
			{
				BLOCK temp;
				temp.ID = ID_1;
				temp.LastLocal = Vy[1][i];
				temp.flag = 0;
				targetList1.push_back(temp);
				ID_1++;
			}
		}

		else
		{
			for (int i = 0; i < targetList1.size(); i++)
			{
				targetList1[i].LastLocal.x -= matchLoc.x;
				targetList1[i].LastLocal.y -= matchLoc.y;
			}
			for (int i = 0; i < targetList1.size(); i++)
			{
				targetList1[i].flag = 0;
				int j;
				for (j = 0; j < Vy[1].size(); j++)
				{
					float rate1, rate2;
					GetOverlap(targetList1[i].LastLocal, Vy[1][j], rate1, rate2);
					if (rate1 > 0.7&&rate2 > 0.7)
					{
						targetList1[i].LastLocal = Vy[1][j];
						targetList1[i].flag = 1;
						targetList1[i].score = rate1 * rate2;
						Vy[1].erase(Vy[1].begin() + j);
						break;
					}
				}
			}

			for (int i = 0; i < targetList1.size(); i++)
			{
				if (targetList1[i].flag == 0)
				{
					targetList1.erase(targetList1.begin() + i);
				}
			}
			if (Vy[1].size() != 0)
			{
				for (int i = 0; i < Vy[1].size(); i++)
				{
					BLOCK temp;
					temp.ID = ID_1;
					temp.LastLocal = Vy[1][i];
					temp.flag = 0;
					targetList1.push_back(temp);
					ID_1++;
				}
			}
		}
		sort(targetList1.begin(), targetList1.end(), cmp_target_x);
		if (iFalg>2)
		{
			Sort_Taget_By_ID(targetList1);
		}






		/*for (int i = 0; i < targetList.size(); i++)
		{
		rectangle(frame, targetList[i].LastLocal, Scalar(255, 255, 0), 2);
		putText(frame, to_string(targetList[i].ID), Point(targetList[i].LastLocal.x, targetList[i].LastLocal.y + targetList[i].LastLocal.height), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1, 8, false);
		}*/
		for (int i = 0; i < targetList0.size(); i++)
		{
			rectangle(frame, targetList0[i].LastLocal, Scalar(255, 255, 0), 2);
			putText(frame, to_string(targetList0[i].ID), Point(targetList0[i].LastLocal.x, targetList0[i].LastLocal.y + targetList0[i].LastLocal.height), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1, 8, false);
		}
		for (int i = 0; i < targetList1.size(); i++)
		{
			rectangle(frame, targetList1[i].LastLocal, Scalar(255, 255, 0), 2);
			putText(frame, to_string(targetList1[i].ID), Point(targetList1[i].LastLocal.x, targetList1[i].LastLocal.y + targetList1[i].LastLocal.height), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1, 8, false);
		}


		//putText(frame, to_string(matchLoc.x), Point(10, 400), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 0), 5, 8, false);
		putText(frame, to_string(FrameNum), Point(10, 400), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 0), 5, 8, false);


		imshow("读取视频", frame);
		imwrite("d:/temp/" + to_string(FrameNum) + ".bmp", frame);

		frame.copyTo(preframe);
		//延时30ms
		waitKey(20);
	}
	return 0;

}



