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

#include "preemptiveSLIC.h"

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

// start a time measurement
double startTimeMeasure()
{
  timeval tim;
  gettimeofday(&tim, 0);
  return tim.tv_sec + (tim.tv_usec / 1000000.0);
}

// stop a time measurement
double stopTimeMeasure(double t1)
{
  timeval tim;
  gettimeofday(&tim, 0);
  double t2 = tim.tv_sec + (tim.tv_usec / 1000000.0);
  return t2 - t1;
}

/*
fill buffer
- each int32 is nothing-R-G-B (each 8 bit)
- row major order
*/
void fillBuffer(cv::Mat &I, unsigned int *buffer)
{
  int dimx = I.cols;
  int dimy = I.rows;

  unsigned int r, g, b;
  unsigned int idx;
  // unsigned int n=dimy*dimx;
  for (int i = 0; i < dimy; i++)
  {
    for (int j = 0; j < dimx; j++)
    {
      // read
      b = I.at<cv::Vec3b>(i, j)[0];
      g = I.at<cv::Vec3b>(i, j)[1];
      r = I.at<cv::Vec3b>(i, j)[2];

      // write in row major order
      idx = dimx * i + j;
      //buffer[idx] = R + G*256 + R*65536;
      buffer[idx] = b + (g << 8) + (r << 16);
    }
  }
}

void getLabelImage(cv::Mat &L, int *labels, int dimx, int dimy)
{
  // label image L
  L = cv::Mat(dimy, dimx, CV_32SC1);
  for (int i = 0; i < dimy; i++)
  {
    for (int j = 0; j < dimx; j++)
    {
      int idx = dimx * i + j; // row major order
      L.at<int>(i, j) = labels[idx];
    }
  }
}

void getBoundaryImage(cv::Mat &B, cv::Mat &L, int dimx, int dimy)
{
  // boundary image B
  B = cv::Mat::zeros(dimy, dimx, CV_8UC1);
  for (int i = 1; i < dimy - 1; i++)
  {
    for (int j = 1; j < dimx - 1; j++)
    {
      if (L.at<int>(i, j) != L.at<int>(i + 1, j) || L.at<int>(i, j) != L.at<int>(i, j + 1))
        B.at<uchar>(i, j) = 1;
    }
  }
}

void getOverlayedImage(cv::Mat &R, cv::Mat &B, cv::Mat &I)
{
  int dimx = I.cols;
  int dimy = I.rows;

  // overlayed image
  I.copyTo(R);
  for (int i = 1; i < dimy - 1; i++)
  {
    for (int j = 1; j < dimx - 1; j++)
    {
      if (B.at<uchar>(i, j))
      {
        R.at<cv::Vec3b>(i, j)[0] = 0;
        R.at<cv::Vec3b>(i, j)[1] = 0;
        R.at<cv::Vec3b>(i, j)[2] = 255;
      }
    }
  }
}

bool fitsOnImage(const cv::Point2f& point, const cv::Mat& img) {
  return (point.x > 0 && point.x < img.cols) && (point.y > 0 && point.y < img.rows);
}

int main(int argc, char **argv) {

  // Loading image filenames
  std::vector<std::string> filenames;
  getFilenames(argv[1], &filenames);
  unsigned nimages = filenames.size();

  // Parameters
  int N = 500;
  unsigned min_tracks = 300;
  double compactness = 40;
  bool tracking_init = false;
  std::vector<cv::Point2f> points_prev, points;
  cv::Mat img_prev;
  cv::Mat seeds;

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
    int *labels_preemptiveSLIC;
    std::vector<double> cluster_x;
    std::vector<double> cluster_y;
    double t = startTimeMeasure();
    PreemptiveSLIC preemptiveSLIC;
    preemptiveSLIC.preemptiveSLIC(img, nspix, compactness, labels_preemptiveSLIC, seeds, cluster_x, cluster_y);
    t = stopTimeMeasure(t);
    std::cout << "Time in ms: " << t * 1000 << std::endl;

    // Copying points
    points_prev.clear();
    points.clear();
    for (size_t j = 0; j < cluster_x.size(); j++) {
      cv::Point2f p(cluster_x[j], cluster_y[j]);
      points_prev.push_back(p);
    }

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

    // Filtering and postprocessing the resulting tracks
    size_t j, k;
    unsigned correct_tracks = 0;
    for (j = 0, k = 0; j < points.size(); j++) {
      // No tracking feature
      // if (!features_found[j] || !fitsOnImage(points[j], img))
      //   continue;
      if (!features_found[j] || !fitsOnImage(points[j], img))
        points[k] = points_prev[j];
      else {
        points[k] = points[j];
        correct_tracks++;
      }
      k += 1;
    }
    points.resize(k);

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

    // --- Image Visualization ---
    // ---------------------------

    // Label image L
    int dimx = img.cols;
    int dimy = img.rows;
    cv::Mat L;
    getLabelImage(L, labels_preemptiveSLIC, dimx, dimy);
    delete[] labels_preemptiveSLIC;

    // boundary image B
    cv::Mat B;
    getBoundaryImage(B, L, dimx, dimy);

    // overlay image
    cv::Mat R;
    getOverlayedImage(R, B, img);

    // Printing cluster centers
    // std::cout << cluster_x.size() << std::endl;
    // for (unsigned j = 0; j < cluster_x.size(); j++)
    // {
    //   cv::circle(R, cv::Point2f(cluster_x[j], cluster_y[j]), 1, cv::Scalar(0, 255, 0));
    // }

    for (unsigned j = 0; j < points_prev.size(); j++) {
      cv::circle(R, cv::Point2f(points_prev[j].x, points_prev[j].y), 1, cv::Scalar(255, 0, 0));
      cv::circle(R, cv::Point2f(points[j].x, points[j].y), 1, cv::Scalar(0, 255, 0));
    }

    cv::imshow("pSLIC", R);
    cv::waitKey(0);

    // std::swap(points, points_prev);
    cv::swap(img_prev, img);
  }

  return 0;
}
