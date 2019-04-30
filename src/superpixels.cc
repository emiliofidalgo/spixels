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

#include "superpixels.h"

namespace spixels {

SuperpixelCenter::SuperpixelCenter(
        const double& l_,
        const double& a_,
        const double& b_,
        const double& x_,
        const double& y_) :
    l(l_),
    a(a_),
    b(b_),
    x(x_),
    y(y_) {}

Superpixel::Superpixel(
        const double& l_,
        const double& a_,
        const double& b_,
        const double& x_,
        const double& y_) :
    center(l_, a_, b_, x_, y_) {}

void Superpixel::addPixel(const double& x, const double& y, bool boundary) {
  // Adding the point itself
  cv::Point2f p(x, y);
  pixels.push_back(p);

  // Adding a boundary if needed
  if (boundary) {
      is_boundary.push_back(1);
  } else {
      is_boundary.push_back(0);
  }
}

void PreemptiveSLICSegmentation::runSegmentation(
    const cv::Mat& I_rgb,
    const int& k,
    const double& compactness,
    cv::Mat& seeds) {

  // Clearing data structures
  if (!image_.empty()) image_.release();
  if (!labels_.empty()) labels_.release();
  if (!boundaries_.empty()) boundaries_.release();
  superpixels_.clear();

  // Copy the current frame for the segmentation
  I_rgb.copyTo(image_);

  // Superpixel segmentation
  int *labels_preemptiveSLIC;
  std::vector<double> cluster_x;
  std::vector<double> cluster_y;
  std::vector<double> cluster_l;
  std::vector<double> cluster_a;
  std::vector<double> cluster_b;
  PreemptiveSLIC preemptiveSLIC;
  preemptiveSLIC.preemptiveSLIC(I_rgb, k, compactness, labels_preemptiveSLIC, seeds, cluster_x, cluster_y, cluster_l, cluster_a, cluster_b);

  // Initializing the set of superpixels
  for (unsigned i = 0; i < cluster_x.size(); i++) {
    Superpixel sp(
      cluster_l[i],
      cluster_a[i],
      cluster_b[i],
      cluster_x[i],
      cluster_y[i]);

    superpixels_.push_back(sp);
  }

  // Postprocessing the segmentation
  int dimx = image_.cols;
  int dimy = image_.rows;
  labels_     = cv::Mat(dimy, dimx, CV_32SC1);
  boundaries_ = cv::Mat::zeros(dimy, dimx, CV_8UC1);
  for (int i = 0; i < dimy; i++) {
    for (int j = 0; j < dimx; j++) {
      int idx = dimx * i + j; // row major order
      int curr_lab = labels_preemptiveSLIC[idx];

      // Setting the label
      labels_.at<int>(i, j) = curr_lab;

      // Detecting if the pixel is a boundary
      bool is_boundary = false;
      if (i > 0 && i < dimy - 1 && j > 0 && j < dimx - 1) {
          int lab_nexti = labels_preemptiveSLIC[dimx * (i + 1) + j];
          int lab_nextj = labels_preemptiveSLIC[dimx * i + (j + 1)];
          if (curr_lab != lab_nexti || curr_lab != lab_nextj) {
              boundaries_.at<uchar>(i, j) = 1;
              is_boundary = true;
          }
      }

      // Adding this pixel to the corresponding Superpixel
      superpixels_[curr_lab].addPixel(j, i, is_boundary);
    }
  }

  delete[] labels_preemptiveSLIC;
}

void PreemptiveSLICSegmentation::drawSegmentationImg(cv::Mat& img) {
  image_.copyTo(img);
  int dimx = image_.cols;
  int dimy = image_.rows;

  for (int i = 1; i < dimy - 1; i++) {
    for (int j = 1; j < dimx - 1; j++) {
      if (boundaries_.at<uchar>(i, j)) {
        img.at<cv::Vec3b>(i, j)[0] = 0;
        img.at<cv::Vec3b>(i, j)[1] = 255;
        img.at<cv::Vec3b>(i, j)[2] = 0;
      }
    }
  }
}

void PreemptiveSLICSegmentation::getCenters(std::vector<cv::Point2f>& points) {
  points.clear();
  for (unsigned i = 0; i < superpixels_.size(); i++) {
    cv::Point2f p(superpixels_[i].center.x, superpixels_[i].center.y);
    points.push_back(p);
  }
}

}  // namespace spixels
