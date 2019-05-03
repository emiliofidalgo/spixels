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

namespace spixels
{

SuperpixelCenter::SuperpixelCenter(
    const double &l_,
    const double &a_,
    const double &b_,
    const double &x_,
    const double &y_) : l(l_),
                        a(a_),
                        b(b_),
                        x(x_),
                        y(y_) {}

Superpixel::Superpixel(
    const double &l_,
    const double &a_,
    const double &b_,
    const double &x_,
    const double &y_) : center(l_, a_, b_, x_, y_) {}

Superpixel::~Superpixel()
{
  pixels.clear();
  is_boundary.clear();
}

void Superpixel::addPixel(const double x, const double y, bool boundary)
{
  // Adding the point itself
  cv::Point2f p(x, y);
  pixels.push_back(p);

  // // Adding a boundary if needed
  if (boundary)
  {
    is_boundary.push_back(1);
  }
  else
  {
    is_boundary.push_back(0);
  }
}

void Superpixel::print()
{
  std::cout << "C(" << center.l;
  std::cout << ", " << center.a;
  std::cout << ", " << center.b;
  std::cout << ", " << center.x;
  std::cout << ", " << center.y;
  std::cout << ") -- NPix: ";
  std::cout << pixels.size() << std::endl;
}

void PreemptiveSLICSegmentation::describeSuperpixels(std::vector<cv::KeyPoint> &kps, cv::Mat &desc, bool use_rgb)
{
  // Initializing keypoints structure
  kps.clear();
  for (unsigned i = 0; i < sp_count_; i++) {
    cv::KeyPoint kp;
    kp.pt.x = cluster_x_[i];
    kp.pt.y = cluster_y_[i];
    kps.push_back(kp);
  }

  // Convert to grayscale
  cv::Mat gray;
  cv::cvtColor(image_, gray, CV_BGR2GRAY);

  // LBP descriptor part
  lbp::LBP lbp(8, lbp::LBP::strToType("riu2"));
  lbp.calcLBP(gray, 1, true);
  cv::Mat lbpImg = lbp.getLBPImage();

  // Extracting Lab planes
  std::vector<cv::Mat> lab_planes;
  cv::split(image_lab_, lab_planes);

  // Creating the corresponding histograms
  int hist_size = 16;
  int curr_label;
  int lbp_val;
  int a_val;
  int b_val;
  cv::Mat lbp_h = cv::Mat::zeros(sp_count_, 10, CV_32F);
  cv::Mat a_h = cv::Mat::zeros(sp_count_, hist_size, CV_32F);
  cv::Mat b_h = cv::Mat::zeros(sp_count_, hist_size, CV_32F);

  // Generating histograms
  for (int i = 0; i < image_.rows; i++) {
    for (int j = 0; j < image_.cols; j++) {
      // Getting the current label of the superpixel
      curr_label = labels_.at<int>(i, j);

      // Getting values
      // LBP channel
      lbp_val = static_cast<int>(lbpImg.at<uchar>(i, j));
      lbp_h.at<float>(curr_label, lbp_val) += 1.0f;

      // A channel
      a_val = static_cast<int>(lab_planes[1].at<uchar>(i, j));      
      int a_pos = a_val % hist_size;
      a_h.at<float>(curr_label, a_pos) += 1.0f;

      // B channel
      b_val = static_cast<int>(lab_planes[2].at<uchar>(i, j));
      int b_pos = b_val % hist_size;
      b_h.at<float>(curr_label, b_pos) += 1.0f;
    }
  }

  // Normalizing histograms
  cv::Mat lbp_h_sum, a_h_sum, b_h_sum;
  cv::reduce(lbp_h, lbp_h_sum, 1, CV_REDUCE_SUM, -1);
  cv::reduce(a_h, a_h_sum, 1, CV_REDUCE_SUM, -1);
  cv::reduce(b_h, b_h_sum, 1, CV_REDUCE_SUM, -1);
  for (int i = 0; i < lbp_h.rows; i++) {
    lbp_h.row(i) /= lbp_h_sum.at<float>(i, 0);
    a_h.row(i) /= a_h_sum.at<float>(i, 0);
    b_h.row(i) /= b_h_sum.at<float>(i, 0);
  }

  // Conforming the final set of descriptors
  cv::hconcat(lbp_h, a_h, desc);
  cv::hconcat(desc, b_h, desc);
}

void PreemptiveSLICSegmentation::runSegmentation(
    const cv::Mat &I_rgb,
    const int &k,
    const double &compactness,
    cv::Mat &seeds)
{

  // Clearing data structures
  if (!image_.empty())
    image_.release();
  if (!image_lab_.empty())
    image_lab_.release();
  if (!labels_.empty())
    labels_.release();
  if (!boundaries_.empty())
    boundaries_.release();
  cluster_x_.clear();
  cluster_y_.clear();
  cluster_l_.clear();
  cluster_a_.clear();
  cluster_b_.clear();

  // Copy the current frame for the segmentation
  I_rgb.copyTo(image_);

  // Superpixel segmentation
  int *labels_preemptiveSLIC;
  PreemptiveSLIC preemptiveSLIC;
  preemptiveSLIC.preemptiveSLIC(I_rgb, k, compactness, labels_preemptiveSLIC, seeds, cluster_x_, cluster_y_, cluster_l_, cluster_a_, cluster_b_, image_lab_);
  sp_count_ = cluster_x_.size();

  // // Postprocessing the segmentation
  int dimx = image_.cols;
  int dimy = image_.rows;
  labels_ = cv::Mat(dimy, dimx, CV_32SC1);
  boundaries_ = cv::Mat::zeros(dimy, dimx, CV_8UC1);

  // Computing labels
  int idx;
  int curr_lab;
  for (int i = 0; i < dimy; i++)
  {
    for (int j = 0; j < dimx; j++)
    {
      idx = dimx * i + j; // row major order
      curr_lab = labels_preemptiveSLIC[idx];

      // Setting the label
      labels_.at<int>(i, j) = curr_lab;
    }
  }

  // Computing boundaries
  for (int i = 1; i < dimy - 1; i++)
  {
    for (int j = 1; j < dimx - 1; j++)
    {
      if (labels_.at<int>(i, j) != labels_.at<int>(i + 1, j) || labels_.at<int>(i, j) != labels_.at<int>(i, j + 1))
        boundaries_.at<uchar>(i, j) = 1;
    }
  }

  // // Adding pixels to superpixels
  // int label;
  // bool is_boundary;
  // for (int i = 0; i < dimy; i++) {
  //   for (int j = 0; j < dimx; j++) {
  //     label = labels_.at<int>(i, j);
  //     is_boundary = boundaries_.at<uchar>(i, j);
  //     superpixels_[label].addPixel(j, i, is_boundary);
  //   }
  // }

  //     // Detecting if the pixel is a boundary
  //     bool is_boundary = false;
  //     if (i > 0 && i < dimy - 1 && j > 0 && j < dimx - 1) {
  //         int lab_nexti = labels_preemptiveSLIC[dimx * (i + 1) + j];
  //         int lab_nextj = labels_preemptiveSLIC[dimx * i + (j + 1)];
  //         if (curr_lab != lab_nexti || curr_lab != lab_nextj) {
  //             boundaries_.at<uchar>(i, j) = 1;
  //             is_boundary = true;
  //         }
  //     }

  //     // Adding this pixel to the corresponding Superpixel
  //     superpixels_[curr_lab].addPixel(j, i, is_boundary);
  //   }
  // }
  delete[] labels_preemptiveSLIC;
}

void PreemptiveSLICSegmentation::getCenters(std::vector<cv::Point2f> &points)
{
  points.clear();
  for (unsigned i = 0; i < cluster_x_.size(); i++)
  {
    cv::Point2f p(cluster_x_[i], cluster_y_[i]);
    points.push_back(p);
  }
}

void PreemptiveSLICSegmentation::drawSegmentationImg(cv::Mat &img)
{
  image_.copyTo(img);
  int dimx = image_.cols;
  int dimy = image_.rows;

  for (int i = 1; i < dimy - 1; i++)
  {
    for (int j = 1; j < dimx - 1; j++)
    {
      if (boundaries_.at<uchar>(i, j))
      {
        img.at<cv::Vec3b>(i, j)[0] = 0;
        img.at<cv::Vec3b>(i, j)[1] = 255;
        img.at<cv::Vec3b>(i, j)[2] = 0;
      }
    }
  }
}

void PreemptiveSLICSegmentation::drawSuperpixels(cv::Mat &img)
{
  drawSegmentationImg(img);

  // Generating colours
  cv::RNG rng(12345);
  cv::Scalar colors[sp_count_];
  for (unsigned i = 0; i < sp_count_; i++)
  {
    colors[i] = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
  }

  int dimx = image_.cols;
  int dimy = image_.rows;
  int label;
  bool is_boundary;
  for (int i = 0; i < dimy; i++)
  {
    for (int j = 0; j < dimx; j++)
    {
      label = labels_.at<int>(i, j);
      is_boundary = boundaries_.at<uchar>(i, j);
      if (!is_boundary)
      {
        img.at<cv::Vec3b>(i, j)[0] = colors[label].val[0];
        img.at<cv::Vec3b>(i, j)[1] = colors[label].val[1];
        img.at<cv::Vec3b>(i, j)[2] = colors[label].val[2];
      }
    }
  }
}

} // namespace spixels
