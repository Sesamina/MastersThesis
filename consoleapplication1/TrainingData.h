#pragma once
#include <vector>
#include <string>
#include <fstream>
#include "Descriptors.h"
#include "Utility.h"

int sampleTrainingData(std::string modelPath, std::vector<std::string> filenames, pcl::PointCloud<pcl::PointXYZ>::Ptr CAD_model_cloud_ptr) {
	//count number of point cloud views
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
		std::cout << "processing CAD model view " << number << std::endl;
		//get the next frame
		std::stringstream filename;
		filename << number << ".pcd";
		if (pcl::io::loadPCDFile<pcl::PointXYZ>(modelPath + filename.str(), *CAD_model_cloud_ptr) != 0)
		{
			return -1;
		}
		filenames.push_back(filename.str());

		std::cout << "#points: " << CAD_model_cloud_ptr->points.size();
		pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>);
		// create passthrough filter instance
		pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;

		// set input cloud
		voxel_grid.setInputCloud(CAD_model_cloud_ptr);

		// set cell/voxel size to 0.1 meters in each dimension
		voxel_grid.setLeafSize(0.02, 0.02, 0.02);

		// do filtering
		voxel_grid.filter(*downsampled);

		std::cout << " after downsampling: " << downsampled->points.size() << std::endl;
		CAD_model_views.push_back(downsampled);
		pcl::io::savePCDFileASCII(filename.str(), *downsampled);

		//compute normals
		std::cout << "computing normals..." << std::endl;
		pcl::PointCloud<pcl::Normal>::Ptr normals = computeNormals(downsampled);
		std::cout << "finish normals..." << std::endl;
		normals_of_views.push_back(normals);
		pcl::io::savePCDFileASCII("normals_" + filename.str(), *normals);

		//compute centroid
		std::cout << "computing centroids..." << std::endl;
		Eigen::Vector4f centroid;
		pcl::compute3DCentroid(*downsampled, centroid);
		centroids_of_views.push_back(centroid);

		// compute the clustered viewpoint feature histogram
		std::cout << "computing cvfh..." << std::endl;
		pcl::PointCloud<pcl::VFHSignature308>::Ptr cvfh_descriptors = computeCVFH(downsampled, normals);
		cvfh_descriptors_of_views.push_back(cvfh_descriptors);
		pcl::io::savePCDFileASCII("cvfh_" + filename.str(), *cvfh_descriptors);

		// compute the camera roll histogram
		std::cout << "computing crh..." << std::endl;
		pcl::PointCloud<pcl::Histogram<90>>::Ptr crh_descriptors = computeCRH(downsampled, normals, centroid);
		crh_descriptors_of_views.push_back(crh_descriptors);
		pcl::io::savePCDFileASCII("crh_" + filename.str(), *crh_descriptors);
	}
	std::cout << "creating FLANN matrix..." << std::endl;

	//create FLANN matrix for later use in matching cvfhs
	flann::Matrix<float> trainingData(new float[cvfh_descriptors_of_views.size() * cvfh_descriptors_of_views[0]->points[0].descriptorSize()],
		cvfh_descriptors_of_views.size(), cvfh_descriptors_of_views[0]->points[0].descriptorSize());

	for (size_t i = 0; i < trainingData.rows; ++i)
		for (size_t j = 0; j < trainingData.cols; ++j)
			trainingData[i][j] = cvfh_descriptors_of_views[i]->points[0].histogram[j];

	// Save the training data matrix to file
	flann::save_to_file(trainingData, "training_data.h5", "training_data");
	std::ofstream fs;
	fs.open("training_data.list");
	for (size_t i = 0; i < filenames.size(); ++i)
		fs << filenames[i] << "\n";
	fs.close();

	return 0;
}

flann::Matrix<float> loadTrainingData(std::vector<std::string> filenames) {
	flann::Matrix<float> trainingData;
	//load list with filenames
	ifstream fs;
	fs.open("training_data.list");
	if (!fs.is_open() || fs.fail())
		std::cout << "error while loading training data" << std::endl;

	std::string line;
	while (!fs.eof())
	{
		getline(fs, line);
		if (line.empty())
			continue;
		filenames.push_back(line);
	}
	fs.close();

	//load the flann matrix of the training data
	flann::load_from_file(trainingData, "training_data.h5", "training_data");
	return trainingData;
}
