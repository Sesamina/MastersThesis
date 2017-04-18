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

#include "DistanceMetric.h"

boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud);
std::string getDirectoryPath(std::string path);
void MatToPoinXYZ(cv::Mat& OpencVPointCloud, cv::Mat& labelInfo, int z, pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud_ptr, int height, int width);
pcl::PointCloud<pcl::Normal>::Ptr computeNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr);
int countNumberOfFilesInDirectory(std::string inputDirectory, const char* fileExtension);
pcl::PointCloud<pcl::VFHSignature308>::Ptr computeCVFH(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr, pcl::PointCloud<pcl::Normal>::Ptr normals); pcl::PointCloud<pcl::Histogram<90>>::Ptr computeCRH(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr, pcl::PointCloud<pcl::Normal>::Ptr normals, Eigen::Vector4f centroid);


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
		"{@model      |      | Path to the models to load.}"
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
	inputDirectory = getDirectoryPath(inputDirectory);
	int count = countNumberOfFilesInDirectory(inputDirectory, "%s*.bmp");
	//get the path to the CAD model point clouds
	modelPath = getDirectoryPath(modelPath);
	int viewCount = countNumberOfFilesInDirectory(modelPath, "%s*.pcd");


	//load CAD models
	int minFrameNumber = 0;
	int maxFrameNumber = viewCount;
	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> CAD_model_views;
	std::vector<pcl::PointCloud<pcl::Normal>::Ptr> normals_of_views;
	std::vector<pcl::PointCloud<pcl::VFHSignature308>::Ptr> cvfh_descriptors_of_views;
	std::vector<pcl::PointCloud<pcl::Histogram<90>>::Ptr> crh_descriptors_of_views;
	std::vector<Eigen::Vector4f> centroids_of_views;
	for (int number = minFrameNumber; number < maxFrameNumber; number++)
	{
		//get the next frame
		std::stringstream filename;
		filename << number << ".pcd";
		if (pcl::io::loadPCDFile<pcl::PointXYZ>(modelPath, *CAD_model_cloud_ptr) != 0)
		{
			return -1;
		}
		CAD_model_views.push_back(CAD_model_cloud_ptr);

		//compute normals
		pcl::PointCloud<pcl::Normal>::Ptr normals = computeNormals(CAD_model_cloud_ptr);
		normals_of_views.push_back(normals);

		//compute centroid
		Eigen::Vector4f centroid;
		pcl::compute3DCentroid(*CAD_model_cloud_ptr, centroid);
		centroids_of_views.push_back(centroid);

		// compute the clustered viewpoint feature histogram
		pcl::PointCloud<pcl::VFHSignature308>::Ptr cvfh_descriptors = computeCVFH(CAD_model_cloud_ptr, normals);
		cvfh_descriptors_of_views.push_back(cvfh_descriptors);

		// compute the camera roll histogram
		pcl::PointCloud<pcl::Histogram<90>>::Ptr crh_descriptors = computeCRH(CAD_model_cloud_ptr, normals, centroid);
		crh_descriptors_of_views.push_back(crh_descriptors);

	}
	//create FLANN matrix for later use in matching cvfhs
	flann::Matrix<float> trainingData(new float[cvfh_descriptors_of_views.size() * cvfh_descriptors_of_views[0]->points[0].descriptorSize()],
		cvfh_descriptors_of_views.size(), cvfh_descriptors_of_views[0]->points[0].descriptorSize());

	for (size_t i = 0; i < trainingData.rows; ++i)
		for (size_t j = 0; j < trainingData.cols; ++j)
			trainingData[i][j] = cvfh_descriptors_of_views[i]->points[0].histogram[j];




	//show last loaded view in viewer
	boost::shared_ptr<pcl::visualization::PCLVisualizer> CADviewer = simpleVis(CAD_model_cloud_ptr);
	CADviewer->spin();







	//process OCT frames
	minFrameNumber = 0;
	maxFrameNumber = count;

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

	//compute normals
	pcl::PointCloud<pcl::Normal>::Ptr normals = computeNormals(point_cloud_ptr);

	//compute centroid
	Eigen::Vector4f centroid;
	pcl::compute3DCentroid(*point_cloud_ptr, centroid);

	// compute the clustered viewpoint feature histogram
	pcl::PointCloud<pcl::VFHSignature308>::Ptr cvfh_descriptors = computeCVFH(point_cloud_ptr, normals);

	// compute the camera roll histogram
	pcl::PointCloud<pcl::Histogram<90>>::Ptr crh_descriptors = computeCRH(point_cloud_ptr, normals, centroid);




	//get k nearest neighbours from cvfhs
	flann::LinearIndex<DistanceMetric<float>> index(trainingData, flann::LinearIndexParams());
	index.buildIndex();
	flann::Matrix<float> pointCloudData(new float[cvfh_descriptors->points[0].descriptorSize()], 1, cvfh_descriptors->points[0].descriptorSize());

	for (size_t i = 0; i < pointCloudData.rows; ++i)
		for (size_t j = 0; j < pointCloudData.cols; ++j)
			pointCloudData[i][j] = cvfh_descriptors->points[0].histogram[j];
	int k = 5;
	flann::Matrix<int> k_indices = flann::Matrix<int>(new int[k], 1, k);;
	flann::Matrix<float> k_distances = flann::Matrix<float>(new float[k], 1, k);
	index.knnSearch(pointCloudData, k_indices, k_distances, k, flann::SearchParams());

	//TODO: get the corresponding crhs


	for (int i = 0; i < k; i++) {
		// compute the roll angle between the computed cloud and the nearest neighbours
		pcl::CRHAlignment<pcl::PointXYZ, 90> alignment;
		//TODO: change to right pointer and right centroid
		alignment.setInputAndTargetView(point_cloud_ptr, CAD_model_views[i]);
		Eigen::Vector3f viewCentroid3f(centroids_of_views[i][0], centroids_of_views[i][1], centroids_of_views[i][2]);
		Eigen::Vector3f clusterCentroid3f(centroid[0], centroid[1], centroid[2]);
		alignment.setInputAndTargetCentroids(clusterCentroid3f, viewCentroid3f);

		// Compute the roll angle(s).
		std::vector<float> angles;
		alignment.computeRollAngle(*crh_descriptors, *crh_descriptors_of_views[i], angles);

		if (angles.size() > 0)
		{
			std::cout << "List of angles where the histograms correlate:" << std::endl;

			for (int i = 0; i < angles.size(); i++)
			{
				std::cout << "\t" << angles.at(i) << " degrees." << std::endl;
			}
		}

		// plot the histograms
		/*pcl::visualization::PCLPlotter plotter;
		plotter.addFeatureHistogram(*cvfh_descriptors, 308);
		plotter.plot();
		pcl::visualization::PCLPlotter plotter2;
		plotter2.addFeatureHistogram(*crh_descriptors, 308);
		plotter2.plot();*/

		//perform iterative closest point
		pcl::PointCloud<pcl::PointXYZ>::Ptr finalCloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> registration;
		registration.setInputSource(CAD_model_views[i]);
		registration.setInputTarget(point_cloud_ptr);

		registration.align(*finalCloud);
		if (registration.hasConverged())
		{
			std::cout << "ICP converged." << std::endl
				<< "The score is " << registration.getFitnessScore() << std::endl;
			std::cout << "Transformation matrix:" << std::endl;
			std::cout << registration.getFinalTransformation() << std::endl;
		}
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
	for (int j = y; j < y + labelHeight; j++) {
		//indicate if first point with intensity = 1 in row has been found
		bool firstNotFound = true;
		//position of last point with intensity = 1 in row
		int lastPointPosition = 0;
		for (int i = x; i < x + labelWidth; i++)
		{
			if (OpencVPointCloud.at<unsigned char>(j, i) >= 1.0f) {
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

// Estimate the normals.
pcl::PointCloud<pcl::Normal>::Ptr computeNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr) {
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
	normalEstimation.setInputCloud(point_cloud_ptr);
	normalEstimation.setRadiusSearch(0.03);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
	normalEstimation.setSearchMethod(kdtree);
	normalEstimation.compute(*normals);
	return normals;
}

// compute the clustered viewpoint feature histogram
pcl::PointCloud<pcl::VFHSignature308>::Ptr computeCVFH(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr, pcl::PointCloud<pcl::Normal>::Ptr normals) {
	pcl::PointCloud<pcl::VFHSignature308>::Ptr cvfh_descriptors(new pcl::PointCloud<pcl::VFHSignature308>);
	pcl::CVFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> cvfh;
	cvfh.setInputCloud(point_cloud_ptr);
	cvfh.setInputNormals(normals);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
	cvfh.setSearchMethod(kdtree);
	cvfh.setEPSAngleThreshold(5.0 / 180.0 * M_PI); // 5 degrees.
	cvfh.setCurvatureThreshold(1.0);
	cvfh.setNormalizeBins(false);
	cvfh.compute(*cvfh_descriptors);
	return cvfh_descriptors;
}

// compute the camera roll histogram
pcl::PointCloud<pcl::Histogram<90>>::Ptr computeCRH(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr, pcl::PointCloud<pcl::Normal>::Ptr normals, Eigen::Vector4f centroid) {
	pcl::PointCloud<pcl::Histogram<90>>::Ptr crh_descriptors(new pcl::PointCloud<pcl::Histogram<90>>);
	pcl::CRHEstimation<pcl::PointXYZ, pcl::Normal, pcl::Histogram<90>> crh;
	crh.setInputCloud(point_cloud_ptr);
	crh.setInputNormals(normals);
	crh.setCentroid(centroid);
	crh.compute(*crh_descriptors);
	return crh_descriptors;
}

// process the path to get the right format 
std::string getDirectoryPath(std::string path) {
	std::replace(path.begin(), path.end(), '\\', '/');
	int lastSlashIndex = path.find_last_of('/', path.size());
	if (lastSlashIndex < (int)path.size() - 1)
		path += "/";
	return path;
}

int countNumberOfFilesInDirectory(std::string inputDirectory, const char* fileExtension) {
	char search_path[300];
	WIN32_FIND_DATA fd;
	sprintf_s(search_path, fileExtension, inputDirectory.c_str());
	HANDLE hFind = ::FindFirstFile(search_path, &fd);

	//count the number of OCT frames in the folder
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
	return count;
}



