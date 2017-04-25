#pragma once
#include <pcl/features/crh.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/common_headers.h>

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

void iterativeClosestPoint(pcl::PointCloud<pcl::PointXYZ>::Ptr& CAD_model_view, pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud_ptr, std::vector<pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>>& icpResults) {
	//perform iterative closest point
	pcl::PointCloud<pcl::PointXYZ>::Ptr finalCloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> registration;
	registration.setInputSource(CAD_model_view);
	registration.setInputTarget(point_cloud_ptr);

	registration.align(*finalCloud);
	if (registration.hasConverged())
	{
		std::cout << "ICP converged." << std::endl
			<< "The score is " << registration.getFitnessScore() << std::endl;
		std::cout << "Transformation matrix:" << std::endl;
		std::cout << registration.getFinalTransformation() << std::endl;
		icpResults.push_back(registration);
	}
}
