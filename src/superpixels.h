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

#ifndef SUPERPIXELS_H
#define SUPERPIXELS_H

#include <opencv2/opencv.hpp>

#include "preemptiveSLIC.h"

namespace spixels {

struct SuperpixelCenter {
  SuperpixelCenter(const double& l_,
                   const double& a_,
                   const double& b_,
                   const double& x_,
                   const double& y_);

  double l;
  double a;
  double b;
  double x;
  double y;
};

class Superpixel {
 public:
  Superpixel(const double& l_,
             const double& a_,
             const double& b_,
             const double& x_,
             const double& y_);
  ~Superpixel();
  void addPixel(const double x, const double y, bool boundary);
  void print();

  SuperpixelCenter center;
  std::vector<cv::Point2f> pixels;
  std::vector<uchar> is_boundary;
};

class PreemptiveSLICSegmentation {
 public:
  PreemptiveSLICSegmentation() {}

  void runSegmentation(
    const cv::Mat& I_rgb,
    const int& k,
    const double& compactness,
    cv::Mat& seeds);  

  void extractSuperpixels(std::vector<Superpixel>& spixs);
  void getCenters(std::vector<cv::Point2f>& points);  
  void drawSegmentationImg(cv::Mat& img);
  
  
 private:
  cv::Mat image_;
  cv::Mat labels_;
  cv::Mat boundaries_;
  std::vector<double> cluster_x_;
  std::vector<double> cluster_y_;
  std::vector<double> cluster_l_;
  std::vector<double> cluster_a_;
  std::vector<double> cluster_b_;
};

}  // namespace spixel

#endif  // SUPERPIXELS_H