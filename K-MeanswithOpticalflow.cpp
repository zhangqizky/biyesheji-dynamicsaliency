#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include<opencv2/video/video.hpp>
#include<string>
#include<fstream>
#include<algorithm>

using namespace std;
using namespace cv;

//int main()
//{
//	VideoCapture cap("music_gummybear_880x720.mp4");
//	if (!cap.isOpened())
//	{
//			return -1;
//	}
//	Mat frame;
//	int i = 0;
//	while (true)
//	{
//		if (!cap.read(frame)) break;
//		imwrite("frame" + to_string(i) + ".jpg", frame);
//		i++;
//	}
//	return 0;
//}



// Farneback dense optical flow calculate and show in Munsell system of colors  
// Author : Zouxy  
// Date   : 2013-3-15  
// HomePage : http://blog.csdn.net/zouxy09  
// Email  : zouxy09@qq.com  

// API calcOpticalFlowFarneback() comes from OpenCV, and this  
// 2D dense optical flow algorithm from the following paper:  
// Gunnar Farneback. "Two-Frame Motion Estimation Based on Polynomial Expansion".  
// And the OpenCV source code locate in ..\opencv2.4.3\modules\video\src\optflowgf.cpp  

#include <iostream>  
#include "opencv2/opencv.hpp"  

using namespace cv;
using namespace std;

#define UNKNOWN_FLOW_THRESH 1e9  

// Color encoding of flow vectors from:  
// http://members.shaw.ca/quadibloc/other/colint.htm  
// This code is modified from:  
// http://vision.middlebury.edu/flow/data/  
void makecolorwheel(vector<Scalar> &colorwheel)
{
	int RY = 15;
	int YG = 6;
	int GC = 4;
	int CB = 11;
	int BM = 13;
	int MR = 6;

	int i;

	for (i = 0; i < RY; i++) colorwheel.push_back(Scalar(255, 255 * i / RY, 0));
	for (i = 0; i < YG; i++) colorwheel.push_back(Scalar(255 - 255 * i / YG, 255, 0));
	for (i = 0; i < GC; i++) colorwheel.push_back(Scalar(0, 255, 255 * i / GC));
	for (i = 0; i < CB; i++) colorwheel.push_back(Scalar(0, 255 - 255 * i / CB, 255));
	for (i = 0; i < BM; i++) colorwheel.push_back(Scalar(255 * i / BM, 0, 255));
	for (i = 0; i < MR; i++) colorwheel.push_back(Scalar(255, 0, 255 - 255 * i / MR));
}

void motionToColor(Mat flow, Mat &color)
{
	if (color.empty())
		color.create(flow.rows, flow.cols, CV_8UC3);

	static vector<Scalar> colorwheel; //Scalar r,g,b  
	if (colorwheel.empty())
		makecolorwheel(colorwheel);

	// determine motion range:  
	float maxrad = -1;

	// Find max flow to normalize fx and fy  
	for (int i = 0; i < flow.rows; ++i)
	{
		for (int j = 0; j < flow.cols; ++j)
		{
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);
			float fx = flow_at_point[0];
			float fy = flow_at_point[1];
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
				continue;
			float rad = sqrt(fx * fx + fy * fy);
			maxrad = maxrad > rad ? maxrad : rad;
		}
	}

	for (int i = 0; i < flow.rows; ++i)
	{
		for (int j = 0; j < flow.cols; ++j)
		{
			uchar *data = color.data + color.step[0] * i + color.step[1] * j;
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);

			float fx = flow_at_point[0] / maxrad;
			float fy = flow_at_point[1] / maxrad;
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))
			{
				data[0] = data[1] = data[2] = 0;
				continue;
			}
			float rad = sqrt(fx * fx + fy * fy);

			float angle = atan2(-fy, -fx) / CV_PI;
			float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);
			int k0 = (int)fk;
			int k1 = (k0 + 1) % colorwheel.size();
			float f = fk - k0;
			//f = 0; // uncomment to see original color wheel  

			for (int b = 0; b < 3; b++)
			{
				float col0 = colorwheel[k0][b] / 255.0;
				float col1 = colorwheel[k1][b] / 255.0;
				float col = (1 - f) * col0 + f * col1;
				if (rad <= 1)
					col = 1 - rad * (1 - col); // increase saturation with radius  
				else
					col *= .75; // out of range  
				data[2 - b] = (int)(255.0 * col);
			}
		}
	}
}
ofstream file("test.txt");
void invert(Mat &src, Mat &dst)
{
	//file << src << endl;
	if (src.cols != dst.cols || src.rows != dst.rows) return;
	for (int i = 0; i < src.rows; i++)
	{
		uchar*p1 = src.ptr<uchar>(i);
		uchar*p2 = dst.ptr<uchar>(i);
		for (int j = 0; j < src.cols; j++)
		{
			uchar pixel = p1[j];
			p2[j] = 255 - uchar(pixel);
		}

	}
}

int main(int, char**)
{
	VideoCapture cap;
	//cap.open(0);
	cap.open("human_chatting07/human_chatting07.mp4");  

	if (!cap.isOpened())
	   return -1;

	Mat prevgray, gray, flow, cflow, frame;
	//namedWindow("flow", 1);

	Mat motion2color;
	int i = 0;
	for (;;)
	{
		double t = (double)cvGetTickCount();

		cap >> frame;
		if (frame.empty()) break;
		//frame=imread("pingpong_closeup_rallys_960x720/" + to_string(i) + ".jpg");
		imwrite("test" + to_string(i) + ".jpg", frame);
		//frame = imread("212.jpg");
		//prevgray = imread("224.jpg");
		//cvtColor(prevgray, prevgray, CV_BGR2GRAY);
		cvtColor(frame, gray, CV_BGR2GRAY);
		//imshow("original", frame);

		if (prevgray.data)
		{
			calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
			motionToColor(flow, motion2color);
			imwrite("opticalflow" + to_string(i) + ".jpg", motion2color);
			Mat grayflow;
			cvtColor(motion2color, grayflow, CV_BGR2GRAY);
			Mat gray_th = Mat::zeros(Size(gray.cols, gray.rows), CV_8U);
			invert(grayflow, gray_th);
			//imwrite("temporal" + to_string(i) + ".jpg", gray_th);
		}
		if (waitKey(10) >= 0)
			break;
		std::swap(prevgray, gray);
		i++;
		t = (double)cvGetTickCount() - t;
		cout << "cost time: " << t / ((double)cvGetTickFrequency()*1000.) << endl;
	}

	//for (int index = 127; index <= 127; index++)
	//{
	//	Mat img = imread("opticalflow" + to_string(index) + ".jpg");
	//	//生成一维采样点,包括所有图像像素点,注意采样点格式为32bit浮点数。   
	//	Mat samples(img.cols*img.rows, 1, CV_32FC3);
	//	//标记矩阵，32位整形   
	//	Mat labels(img.cols*img.rows, 1, CV_32SC1);
	//	uchar* p;
	//	int i, j, k = 0;
	//	for (i = 0; i < img.rows; i++)
	//	{
	//		p = img.ptr<uchar>(i);
	//		for (j = 0; j < img.cols; j++)
	//		{
	//			samples.at<Vec3f>(k, 0)[0] = float(p[j * 3]);
	//			samples.at<Vec3f>(k, 0)[1] = float(p[j * 3 + 1]);
	//			samples.at<Vec3f>(k, 0)[2] = float(p[j * 3 + 2]);
	//			k++;
	//		}
	//	}

	//	int clusterCount = 3;
	//	Mat centers(clusterCount, 1, samples.type());
	//	kmeans(samples, clusterCount, labels,
	//		TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),
	//		3, KMEANS_PP_CENTERS, centers);
	//	//我们已知有3个聚类，用不同的灰度层表示。   
	//	Mat img1(img.rows, img.cols, CV_32FC3, Scalar(255, 255, 255));
	//	Mat img2(img.rows, img.cols, CV_32FC3, Scalar(255, 255, 255));
	//	Mat img3(img.rows, img.cols, CV_32FC3, Scalar(255, 255, 255));
	//	Mat img4(img.rows, img.cols, CV_32FC3, Scalar(255, 255, 255));
	//	float step = 255 / (clusterCount - 1);
	//	k = 0;

	//	for (i = 0; i < img1.rows; i++)
	//	{
	//		p = img.ptr<uchar>(i);
	//		for (j = 0; j < img1.cols; j++)
	//		{
	//			int tt = labels.at<int>(k, 0);
	//			if (tt == 1)
	//			{
	//				//p[j] = 255 - tt*step;
	//				img1.at<Vec3f>(i, j)[0] = samples.at<Vec3f>(k, 0)[0];
	//				img1.at<Vec3f>(i, j)[1] = samples.at<Vec3f>(k, 0)[1];
	//				img1.at<Vec3f>(i, j)[2] = samples.at<Vec3f>(k, 0)[2];
	//			}
	//			if (tt == 2)
	//			{
	//				//p[j] = 255 - tt*step;
	//				img2.at<Vec3f>(i, j)[0] = samples.at<Vec3f>(k, 0)[0];
	//				img2.at<Vec3f>(i, j)[1] = samples.at<Vec3f>(k, 0)[1];
	//				img2.at<Vec3f>(i, j)[2] = samples.at<Vec3f>(k, 0)[2];
	//			}
	//			if (tt == 3)
	//			{
	//				//p[j] = 255 - tt*step;
	//				img3.at<Vec3f>(i, j)[0] = samples.at<Vec3f>(k, 0)[0];
	//				img3.at<Vec3f>(i, j)[1] = samples.at<Vec3f>(k, 0)[1];
	//				img3.at<Vec3f>(i, j)[2] = samples.at<Vec3f>(k, 0)[2];
	//			}
	//			//if (tt == 4)
	//			//{
	//			//	//p[j] = 255 - tt*step;
	//			//	img4.at<Vec3f>(i, j)[0] = samples.at<Vec3f>(k, 0)[0];
	//			//	img4.at<Vec3f>(i, j)[1] = samples.at<Vec3f>(k, 0)[1];
	//			//	img4.at<Vec3f>(i, j)[2] = samples.at<Vec3f>(k, 0)[2];
	//			//}
	//			k++;
	//		}
	//	}
	//	imwrite("K-Means分割效果1" + to_string(index) + ".jpg", img1);
	//	imwrite("K-Means分割效果2" + to_string(index) + ".jpg", img2);
	//	imwrite("K-Means分割效果3" + to_string(index) + ".jpg", img3);
	//	imwrite("K-Means分割效果4" + to_string(index) + ".jpg", img4);
	//}
	//Mat img_2 = imread("K-Means分割效果"+to_string(42)+".jpg");
		//cout << img_2.type() <<"  "<<img_2.channels() << endl;
		//Mat gray;
		//cvtColor(img_2, gray, CV_BGR2GRAY);
		////imwrite("test.jpg", gray);
		//cout << gray.channels() << endl;
		////cout << gray << endl;
		//Mat res=Mat::zeros(Size(img_2.cols,img_2.rows),CV_8U);
		//invert(gray, res);
		////threshold(res,res,50,255,3);
		//GaussianBlur(res, res, Size(23, 23), 25.5, 25.5);
		//imwrite("K-Means分割效果gray"+to_string(42)+".jpg", res);
        //}


	//Mat frame;
	//for (int i = 42; i <= 42; i++)
	//{
	//	frame = imread("K-Means分割效果42.jpg");
	//	Mat gray1;
	//	Mat gray2=Mat::zeros(Size(frame.cols,frame.rows),CV_8U);
	//	cvtColor(frame, gray1, CV_BGR2GRAY);
	//	invert(gray1, gray2);
	//	GaussianBlur(gray2,gray2,Size(45,45),11.5,11.5);
	//	imwrite("temporal"+to_string(i)+".jpg",gray2);
	//}
	//frame = imread("temporal42.jpg",0);
	//threshold(frame,frame,240,255,4);
	//GaussianBlur(frame, frame, Size(15, 15), 4.5, 4.5);
	//imwrite("final42.jpg", frame);

	return 0;
}