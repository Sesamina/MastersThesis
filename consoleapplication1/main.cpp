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

#include <memory>
#include "opencv2\opencv.hpp"
#include "OCTDefs.hpp"

#include <pcl/common/centroid.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/features/cvfh.h>
#include <pcl/recognition/crh_alignment.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>

#include <boost/thread/thread.hpp>
#include <boost/filesystem.hpp>

#include <flann/flann.h>
#include <flann/io/hdf5.h>

#include "DistanceMetric.h"
#include "Utility.h"
#include "TrainingData.h"
#include "Descriptors.h"


pcl::PointCloud<pcl::PointXYZ>::Ptr CAD_model_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);

bool waitKey = false;
int volumeBScans = 128;
bool notSaved = true;


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

	std::cout << "path input processed" << std::endl;

	std::vector<std::string> filenames;
	flann::Matrix<float> trainingData;

	//training data has to be generated
	if (notSaved)
	{
		if (sampleTrainingData(modelPath, filenames, CAD_model_cloud_ptr) != 0) {
			return -1;
		}		
	}
	//there exists saved training data
	else {
		trainingData = loadTrainingData(filenames);
	}

	std::cout << "finished processing CAD models." << std::endl;
	

	//process OCT frames
	int minFrameNumber = 0;
	int maxFrameNumber = count;

	pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
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

		processOCTFrame(imageGray, number, point_cloud_ptr);
		//show the images
		//cv::imshow("OCT", transposedOCTimage);

		cv::waitKey(10);
		countIm = countIm + 1;

	}
	std::cout << "finished processing OCT frames." << std::endl;

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

	std::cout << "starting nearest neighbour computation..." << std::endl;

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


	std::vector<pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>> icpResults;

	std::cout << "computing the roll angles..." << std::endl;
	for (int i = 0; i < k; i++) {

		std::cout << "distance: " << k_distances[0][i] << std::endl;

		int index = k_indices[i][0];
		//load point cloud at selected index
		pcl::PCLPointCloud2 cloud;
		if (pcl::io::loadPCDFile(filenames[index], cloud) == -1)
			break;
		pcl::PointCloud<pcl::PointXYZ> cloud2;
		pcl::fromPCLPointCloud2(cloud, cloud2);
		pcl::PointCloud<pcl::PointXYZ>::Ptr CAD_model_view(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::demeanPointCloud<pcl::PointXYZ>(cloud2, centroid, *CAD_model_view);
		//load crh at selected index
		pcl::PointCloud<pcl::Histogram<90>> cloud3;
		if (pcl::io::loadPCDFile("crh_"+filenames[index], cloud) == -1)
			break;
		pcl::fromPCLPointCloud2(cloud, cloud3);
		pcl::PointCloud<pcl::Histogram<90>>::Ptr crh_view(new pcl::PointCloud<pcl::Histogram<90>>);
		pcl::demeanPointCloud<pcl::Histogram<90>>(cloud3, centroid, *crh_view);
		//load normals at selected index
		pcl::PointCloud<pcl::Normal> cloud4;
		if (pcl::io::loadPCDFile("normals_" + filenames[index], cloud) == -1)
			break;
		pcl::fromPCLPointCloud2(cloud, cloud4);
		pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
		pcl::demeanPointCloud<pcl::Normal>(cloud4, centroid, *normals);

		// compute the roll angle between the computed cloud and the nearest neighbours
		pcl::CRHAlignment<pcl::PointXYZ, 90> alignment;
		alignment.setInputAndTargetView(point_cloud_ptr, CAD_model_view);

		Eigen::Vector4f centroid;
		pcl::compute3DCentroid(*CAD_model_view, centroid);
		Eigen::Vector3f viewCentroid3f(centroid[0], centroid[1], centroid[2]);
		Eigen::Vector3f clusterCentroid3f(centroid[0], centroid[1], centroid[2]);
		alignment.setInputAndTargetCentroids(clusterCentroid3f, viewCentroid3f);

		// Compute the roll angle(s).

		std::vector<float> angles;
		alignment.computeRollAngle(*crh_descriptors, *crh_view, angles);

		if (angles.size() > 0)
		{
			std::cout << "List of angles where the histograms correlate:" << std::endl;

			for (int i = 0; i < angles.size(); i++)
			{
				std::cout << "\t" << angles.at(i) << " degrees." << std::endl;
			}
		}
		else {
			std::cout << "no angles where histograms correlate..." << std::endl;
		}

		// plot the histograms
		/*pcl::visualization::PCLPlotter plotter1 = plotHistogram(*cvfh_descriptors);
		pcl::visualization::PCLPlotter plotter2 = plotHistogram(*crh_descriptors);*/

		std::cout << "starting ICP #" << i << std::endl;
		iterativeClosestPoint(CAD_model_view, point_cloud_ptr, icpResults);
	}
	//TODO: sort nearest neighbours using num inliers from last ICP iteration with dist thresh of twice voxel grid size
	std::sort(icpResults.begin(), icpResults.end());

	return 0;
}
















