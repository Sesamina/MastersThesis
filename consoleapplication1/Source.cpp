/*-------------------------------------------------------------------*/
/*                                                                   */
/*                    Zeiss Interventional Imaging                   */
/*                         Research Solution                         */
/*                   -----------------------------                   */
/*  Chair for Computer Aided Medical Procedures & Augmented Reality  */
/*                  Technische Universität München                   */
/*                                                                   */
/*-------------------------------------------------------------------*/

#pragma once

#include <fstream>
#include <memory>
#include <vector>
#include <algorithm>
#include <string>
#include <Windows.h>
#include "opencv2\opencv.hpp"
#include "OCTDefs.hpp"

#include <pcl/common/common_headers.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include<pcl/visualization/pcl_plotter.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/cvfh.h>
#include <boost/thread/thread.hpp>

boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis(pcl::PointCloud<pcl::PointXYZI>::ConstPtr cloud, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> rgb);
void MatToPoinXYZ(cv::Mat& OpencVPointCloud, cv::Mat& labelInfo, int z, pcl::PointCloud<pcl::PointXYZI>::Ptr& point_cloud_ptr, int height, int width);
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* viewer_void);


pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> rgb(point_cloud_ptr, 192, 192, 192);

bool waitKey = false;
int volumeBScans = 128;

int main(int argc, char** argv)
{
	int previousScanNumber = -1;
	int countIm = 0;
	std::string keys =
		"{help h ?    |      | Display the program help.}"
		"{@input      |      | Path to the directory containing recorded OCT frames.}"
		;
	cv::CommandLineParser parser(argc, argv, keys);
	parser.about("Zeiss Interventional Imaging Research Solution");
	if (parser.has("help"))
	{
		parser.printMessage();
		exit(0);
	}
	std::string inputDirectory = parser.get<std::string>(0);

	if (!parser.check())
	{
		parser.printErrors();
		exit(0);
	}

	//get the path to the images
	std::replace(inputDirectory.begin(), inputDirectory.end(), '\\', '/');
	int lastSlashIndex = inputDirectory.find_last_of('/', inputDirectory.size());
	if (lastSlashIndex < (int)inputDirectory.size() - 1)
		inputDirectory += "/";

	char search_path[300];
	WIN32_FIND_DATA fd;
	sprintf_s(search_path, "%s*.bmp", inputDirectory.c_str());
	HANDLE hFind = ::FindFirstFile(search_path, &fd);

	//count the number of pictures in the folder
	int count = 0;
	if (hFind != INVALID_HANDLE_VALUE)
	{
		do
		{
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
			{

				count++;
				
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}

	int minFrameNumber = 0;
	int maxFrameNumber = count;

	std::string filename = "";
	//	go through all frames
	for (int number = minFrameNumber; number < maxFrameNumber; number++)
	{
		//get the next frame
		std::stringstream filename;
		if (number < 100) {
			filename << "0";
		}
		if (number < 10) {
			filename << "0";
		}
		filename << number << ".bmp";
		//read the image in grayscale
		cv::Mat imageGray = cv::imread(inputDirectory.c_str() + filename.str(), CV_LOAD_IMAGE_GRAYSCALE);

		//flip and transpose the image
		cv::Mat transposedOCTimage;
		cv::flip(imageGray, imageGray, 0);
		cv::transpose(imageGray, transposedOCTimage);

		//set a threshold (0.26)
		cv::Mat thresholdedImage;
		cv::threshold(transposedOCTimage, thresholdedImage, 0.26 * 255, 1, 0);

		//use a median blur filter
		cv::Mat filteredImage;
		cv::medianBlur(thresholdedImage, filteredImage, 3);

		//label the image
		cv::Mat labelledImage;
		cv::Mat labelStats;
		cv::Mat labelCentroids;
		int numLabels = cv::connectedComponentsWithStats(filteredImage, labelledImage, labelStats, labelCentroids);

		//for every label with more than 400 points process it further for adding points to the cloud
		for (int i = 1; i < numLabels; i++) {
			if (labelStats.at<int>(i, cv::CC_STAT_AREA) > 400) {
				cv::Mat labelInfo = labelStats.row(i);
				MatToPoinXYZ(filteredImage, labelInfo, number, point_cloud_ptr, thresholdedImage.rows, thresholdedImage.cols);
			}
		}

		//show the images
		cv::imshow("OCT", transposedOCTimage);

		cv::waitKey(10);
		countIm = countIm + 1;

	}

	// Object for storing the normals.
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	// Object for storing the CVFH descriptors.
	pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptors(new pcl::PointCloud<pcl::VFHSignature308>);
	// Estimate the normals.
	pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> normalEstimation;
	normalEstimation.setInputCloud(point_cloud_ptr);
	normalEstimation.setRadiusSearch(0.03);
	pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZI>);
	normalEstimation.setSearchMethod(kdtree);
	normalEstimation.compute(*normals);
	// CVFH estimation object.
	pcl::CVFHEstimation<pcl::PointXYZI, pcl::Normal, pcl::VFHSignature308> cvfh;
	cvfh.setInputCloud(point_cloud_ptr);
	cvfh.setInputNormals(normals);
	cvfh.setSearchMethod(kdtree);
	// Set the maximum allowable deviation of the normals,
	// for the region segmentation step.
	cvfh.setEPSAngleThreshold(5.0 / 180.0 * M_PI); // 5 degrees.
												   // Set the curvature threshold (maximum disparity between curvatures),
												   // for the region segmentation step.
	cvfh.setCurvatureThreshold(1.0);
	// Set to true to normalize the bins of the resulting histogram,
	// using the total number of points. Note: enabling it will make CVFH
	// invariant to scale just like VFH, but the authors encourage the opposite.
	cvfh.setNormalizeBins(false);

	cvfh.compute(*descriptors);

	// Plotter object.
	pcl::visualization::PCLPlotter plotter;
	// We need to set the size of the descriptor beforehand.
	plotter.addFeatureHistogram(*descriptors, 308);

	plotter.plot();

	//open a viewer and show the cloud
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = simpleVis(point_cloud_ptr, rgb);
	viewer->updatePointCloud(point_cloud_ptr, rgb,  "sample cloud");
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
	return 0;
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis(pcl::PointCloud<pcl::PointXYZI>::ConstPtr cloud, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> rgb)
{
	// --------------------------------------------
	// -----Open 3D viewer and add point cloud-----
	// --------------------------------------------
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0); 
	viewer->addPointCloud<pcl::PointXYZI>(cloud, rgb, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	viewer->spinOnce();
	return (viewer);
}

void MatToPoinXYZ(cv::Mat& OpencVPointCloud, cv::Mat& labelInfo, int z, pcl::PointCloud<pcl::PointXYZI>::Ptr& point_cloud_ptr, int height, int width)
{
	//get the infos for the bounding box
	int x = labelInfo.at<int>(0, cv::CC_STAT_LEFT);
	int y = labelInfo.at<int>(0, cv::CC_STAT_TOP);
	int labelWidth = labelInfo.at<int>(0, cv::CC_STAT_WIDTH);
	int labelHeight = labelInfo.at<int>(0, cv::CC_STAT_HEIGHT);
	//go through points in bounding box
	for (int j = y; j < y+labelHeight; j++) {
		//indicate if first point with intensity = 1 in row has been found
		bool firstNotFound = true;
		//position of last point with intensity = 1 in row
		int lastPointPosition = 0;
		for (int i = x; i < x+labelWidth; i++)
		{
			if (OpencVPointCloud.at<uchar>(j,i) >= 1.0f){
				if (firstNotFound) {
					firstNotFound = false;
				}
				lastPointPosition = i;
			}
		}
		if (!firstNotFound) {
			//add the last point with intensity = 1 in row to the point cloud
			pcl::PointXYZI point;
			point.x = (float)z / volumeBScans * 2.6f;
			point.y = (float)j / height * 3.0f;
			point.z = (float)lastPointPosition / width * 2.0f;
			point.intensity = OpencVPointCloud.at<uchar>(j, lastPointPosition);
			point_cloud_ptr->points.push_back(point);
		}
	}
}
