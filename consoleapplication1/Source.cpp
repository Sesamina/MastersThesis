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
#include <pcl/common/centroid.h>
#include <pcl/point_cloud.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/cvfh.h>
#include <pcl/features/crh.h>
#include <pcl/recognition/crh_alignment.h>
#include <pcl/registration/icp.h>

#include <boost/thread/thread.hpp>

boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud);
void MatToPoinXYZ(cv::Mat& OpencVPointCloud, cv::Mat& labelInfo, int z, pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud_ptr, int height, int width);


pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr CAD_model_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);

bool waitKey = false;
int volumeBScans = 128;

int main(int argc, char** argv)
{
	int previousScanNumber = -1;
	int countIm = 0;
	std::string keys =
		"{help h ?    |      | Display the program help.}"
		"{@input      |      | Path to the directory containing recorded OCT frames.}"
		"{@model      |      | Path and filename of the model to load.}"
		;
	cv::CommandLineParser parser(argc, argv, keys);
	parser.about("Zeiss Interventional Imaging Research Solution");
	if (parser.has("help"))
	{
		parser.printMessage();
		exit(0);
	}
	std::string inputDirectory = parser.get<std::string>("@input");
	std::string modelPath = parser.get<std::string>("@model");

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

	//load CAD model (.stl)
	pcl::PolygonMesh::Ptr CAD_model(new pcl::PolygonMesh);
	pcl::io::loadPolygonFileSTL(modelPath, *CAD_model);
	pcl::fromPCLPointCloud2(CAD_model->cloud, *CAD_model_cloud_ptr);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> CADviewer = simpleVis(CAD_model_cloud_ptr);
	CADviewer->spin();


	//process OCT frames
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

	//open a viewer and show the generated point cloud after processing the frames
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = simpleVis(point_cloud_ptr);
	viewer->spin();

	// Estimate the normals.
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
	normalEstimation.setInputCloud(point_cloud_ptr);
	normalEstimation.setRadiusSearch(0.03);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
	normalEstimation.setSearchMethod(kdtree);
	normalEstimation.compute(*normals);

	//compute centroid
	Eigen::Vector4f centroid;
	pcl::compute3DCentroid(*point_cloud_ptr, centroid);

	// compute the clustered viewpoint feature histogram
	pcl::PointCloud<pcl::VFHSignature308>::Ptr cvfh_descriptors(new pcl::PointCloud<pcl::VFHSignature308>);
	pcl::CVFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> cvfh;
	cvfh.setInputCloud(point_cloud_ptr);
	cvfh.setInputNormals(normals);
	cvfh.setSearchMethod(kdtree);
	cvfh.setEPSAngleThreshold(5.0 / 180.0 * M_PI); // 5 degrees.
	cvfh.setCurvatureThreshold(1.0);
	cvfh.setNormalizeBins(false);
	cvfh.compute(*cvfh_descriptors);

	// compute the camera roll histogram
	pcl::PointCloud<pcl::Histogram<90>>::Ptr crh_descriptors(new pcl::PointCloud<pcl::Histogram<90>>);
	pcl::CRHEstimation<pcl::PointXYZ, pcl::Normal, pcl::Histogram<90>> crh;
	crh.setInputCloud(point_cloud_ptr);
	crh.setInputNormals(normals); 
	crh.setCentroid(centroid);
	crh.compute(*crh_descriptors);

	// compute the roll angle
	pcl::CRHAlignment<pcl::PointXYZ, 90> alignment;
	alignment.setInputAndTargetView(point_cloud_ptr, CAD_model_cloud_ptr);
	// CRHAlignment works with Vector3f, not Vector4f.
	//Eigen::Vector3f viewCentroid3f(CADCentroid[0], CADCentroid[1], CADCentroid[2]);
	//Eigen::Vector3f clusterCentroid3f(centroid[0], centroid[1], centroid[2]);
	//alignment.setInputAndTargetCentroids(clusterCentroid3f, viewCentroid3f);

	//// Compute the roll angle(s).
	//std::vector<float> angles;
	//alignment.computeRollAngle(*crh_descriptors, *CAD_crh_descriptors, angles);

	//if (angles.size() > 0)
	//{
	//	std::cout << "List of angles where the histograms correlate:" << std::endl;

	//	for (int i = 0; i < angles.size(); i++)
	//	{
	//		std::cout << "\t" << angles.at(i) << " degrees." << std::endl;
	//	}
	//}

	// plot the histograms
	pcl::visualization::PCLPlotter plotter;
	plotter.addFeatureHistogram(*cvfh_descriptors, 308);
	plotter.plot();
	pcl::visualization::PCLPlotter plotter2;
	plotter2.addFeatureHistogram(*crh_descriptors, 308);
	plotter2.plot();

	//perform iterative closest point
	pcl::PointCloud<pcl::PointXYZ>::Ptr finalCloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> registration;
	registration.setInputSource(CAD_model_cloud_ptr);
	registration.setInputTarget(point_cloud_ptr);

	registration.align(*finalCloud);
	if (registration.hasConverged())
	{
		std::cout << "ICP converged." << std::endl
			<< "The score is " << registration.getFitnessScore() << std::endl;
		std::cout << "Transformation matrix:" << std::endl;
		std::cout << registration.getFinalTransformation() << std::endl;
	}

	return 0;
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
	// --------------------------------------------
	// -----Open 3D viewer and add point cloud-----
	// --------------------------------------------
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_handler(cloud, 192, 192, 192);
	viewer->addPointCloud<pcl::PointXYZ>(cloud, rgb_handler, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	viewer->spinOnce();
	return (viewer);
}

void MatToPoinXYZ(cv::Mat& OpencVPointCloud, cv::Mat& labelInfo, int z, pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud_ptr, int height, int width)
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
			pcl::PointXYZ point;
			point.x = (float)z / volumeBScans * 2.6f;
			point.y = (float)j / height * 3.0f;
			point.z = (float)lastPointPosition / width * 2.0f;
			point_cloud_ptr->points.push_back(point);
		}
	}
}
