#pragma once
#include <string>
#include <Windows.h>
#include <algorithm>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/impl/point_types.hpp>

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

// process the path to get the right format 
std::string getDirectoryPath(std::string path) {
	std::replace(path.begin(), path.end(), '\\', '/');
	int lastSlashIndex = path.find_last_of('/', path.size());
	if (lastSlashIndex < (int)path.size() - 1)
		path += "/";
	return path;
}

//visualize point cloud
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

//template <typename PointT>
//pcl::visualization::PCLPlotter plotHistogram(pcl::PointCloud<PointT>& descriptors) {
//	pcl::visualization::PCLPlotter plotter;
//	plotter.addFeatureHistogram(descriptors, 308);
//	plotter.plot();
//	return plotter;
//}

//convert label in opencv matrix to point cloud
void MatToPoinXYZ(cv::Mat& OpencVPointCloud, cv::Mat& labelInfo, int z, pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud_ptr, int height, int width)
{
	int volumeBScans = 128;
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

//process the OCT frame
void processOCTFrame(cv::Mat imageGray, int number, pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud_ptr) {
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
}



