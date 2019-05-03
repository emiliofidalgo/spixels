/**
* This file is part of spixels.
*
* Copyright (C) 2019 Emilio Garcia-Fidalgo <emilio.garcia@uib.es> (University of the Balearic Islands)
*
* spixels is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* spixels is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with spixels. If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <sys/time.h>

#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

#include "superpixels.h"

void getFilenames(const std::string &directory,
                  std::vector<std::string> *filenames)
{
  using namespace boost::filesystem;

  filenames->clear();
  path dir(directory);

  // Retrieving, sorting and filtering filenames.
  std::vector<path> entries;
  copy(directory_iterator(dir), directory_iterator(), back_inserter(entries));
  sort(entries.begin(), entries.end());
  for (auto it = entries.begin(); it != entries.end(); it++)
  {
    std::string ext = it->extension().c_str();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".png" || ext == ".jpg" ||
        ext == ".ppm" || ext == ".jpeg")
    {
      filenames->push_back(it->string());
    }
  }
}

bool fitsOnImage(const cv::Point2f& point, const cv::Mat& img) {
  return (point.x > 0 && point.x < img.cols) && (point.y > 0 && point.y < img.rows);
}

float euclideanDist(cv::Point2f& a, cv::Point2f& b) {
  cv::Point2f diff = a - b;
  return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

int main(int argc, char **argv) {

  // Loading image filenames
  std::vector<std::string> filenames;
  getFilenames(argv[1], &filenames);
  unsigned nimages = filenames.size();

  // Parameters
  int N = 500;
  unsigned min_tracks = 200;
  double compactness = 40;
  bool tracking_init = false;
  std::vector<cv::Point2f> points_prev, points;
  cv::Mat img_prev;
  cv::Mat seeds;  

  spixels::PreemptiveSLICSegmentation segm;

  // Processing the sequence of images
  for (unsigned i = 0; i < nimages; i++)
  {
    // Processing image i
    std::cout << "--- Processing image " << i << std::endl;

    // Loading and describing the image
    cv::Mat img = cv::imread(filenames[i]);
    
    if (i == 0) {
      img.copyTo(img_prev);
      continue;
    }

    // Deciding the number of superpixels to use
    int nspix = N;
    if (tracking_init) {
      nspix = seeds.cols;
    }

    std::cout << "Number of superpixels: " << nspix << std::endl;
    std::cout << "Seeds size: " << seeds.rows << " " << seeds.cols << std::endl;

    // std::cout << "Printing SEEDS: " << std::endl;
    // for (int j = 0; j < seeds.cols; j++) {
    //   std::cout << seeds.at<float>(0, j) << ", " << seeds.at<float>(1, j) << std::endl;
    // }

    // Superpixel segmentation
    // Segmentation
    segm.runSegmentation(img, nspix, compactness, seeds);

    // std::vector<spixels::Superpixel> superpixels;
    // segm.extractSuperpixels(superpixels);

    // Copying points    
    points.clear();
    points_prev.clear();
    segm.getCenters(points_prev);

    // Computing Optical Flow
    std::vector<uchar> features_found;
    cv::calcOpticalFlowPyrLK(
        img_prev,         // Previous image
        img,              // Next image
        points_prev,      // Previous set of points (from img_prev)
        points,           // Next set of points (from img)
        features_found,   // Output vector, each is 1 for tracked
        cv::noArray(),    // Output vector, lists errors (optional)
        cv::Size(21, 21), // Search window size
        3,                // Maximum pyramid level to construct
        cv::TermCriteria(
            cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
            20, // Maximum number of iterations
            0.3 // Minimum change per iteration
            ));
    
    // Computing Optical Flow
    std::vector<uchar> features_found_aux;
    std::vector<cv::Point2f> points_prev_aux;
    cv::calcOpticalFlowPyrLK(
        img,                  // Previous image
        img_prev,             // Next image
        points,               // Previous set of points (from img_prev)
        points_prev_aux,      // Next set of points (from img)
        features_found_aux,   // Output vector, each is 1 for tracked
        cv::noArray(),        // Output vector, lists errors (optional)
        cv::Size(21, 21),     // Search window size
        3,                    // Maximum pyramid level to construct
        cv::TermCriteria(
            cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
            20, // Maximum number of iterations
            0.3 // Minimum change per iteration
            ));

    // Filtering and postprocessing the resulting tracks
    // unsigned correct_tracks = 0;
    // for (unsigned j = 0; j < points.size(); j++) {
    //   if (!features_found[j] || !fitsOnImage(points[j], img)) {
    //     points[j] = points_prev[j];
    //   } else {
    //     correct_tracks++;
    //   }
    // }

    unsigned correct_tracks = 0;
    for (unsigned j = 0; j < points.size(); j++) {
      if (features_found[j] && 
          features_found_aux[j] &&
          fitsOnImage(points[j], img) &&
          euclideanDist(points_prev[j], points_prev_aux[j]) < 1.0f) {
            correct_tracks++;
          } else {
            points[j] = points_prev[j];
          }
    }

    std::cout << "Correct tracks: " << correct_tracks << std::endl;

    // Adjusting seeds
    if (!seeds.empty()) {
      seeds.release();
    }

    if (correct_tracks < min_tracks) {
      tracking_init = false;
      std::cout << "Reinitializing seeds" << std::endl;
    }
    else {
      tracking_init = true;
      seeds = cv::Mat::zeros(2, (int)points.size(), CV_32FC1);
      for (size_t j = 0; j < points.size(); j++) {
        seeds.at<float>(0, j) = points[j].y;
        seeds.at<float>(1, j) = points[j].x;
      }
    }

    // Describing superpixels
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    segm.describeSuperpixels(kps, desc);

    // --- Image Visualization ---
    // ---------------------------

    // overlay image
    cv::Mat overlay_img;
    segm.drawSegmentationImg(overlay_img);

    // Printing cluster centers
    // std::cout << cluster_x.size() << std::endl;
    // for (unsigned j = 0; j < cluster_x.size(); j++)
    // {
    //   cv::circle(R, cv::Point2f(cluster_x[j], cluster_y[j]), 1, cv::Scalar(0, 255, 0));
    // }

    for (unsigned j = 0; j < points_prev.size(); j++) {
      cv::circle(overlay_img, cv::Point2f(points_prev[j].x, points_prev[j].y), 2, cv::Scalar(255, 0, 0), -1);
      cv::circle(overlay_img, cv::Point2f(points[j].x, points[j].y), 2, cv::Scalar(0, 0, 255), -1);
      cv::line(overlay_img, cv::Point2f(points_prev[j].x, points_prev[j].y), cv::Point2f(points[j].x, points[j].y), cv::Scalar(0, 255, 255), 2);
    }

    cv::imshow("pSLIC", overlay_img);
    cv::waitKey(5);

    // std::swap(points, points_prev);
    cv::swap(img_prev, img);
  }

  return 0;
}
