/*
 * Preemptive SLIC
 * Copyright (C) 2014  Peer Neubert, peer.neubert@etit.tu-chemnitz.de
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * ----------------------------------------
 * 
 * This programm demonstrates the usage of the Preemptive SLIC implementation from:
 * 
 * "Compact Watershed and Preemptive SLIC: On improving trade-offs of superpixel 
 * segmentation algorithms" Peer Neubert and Peter Protzel, ICPR 2014 
 * 
 */ 

#include <math.h>
#include <stdio.h>

#include <vector>
#include <iostream>
#include "sys/time.h"

#include "opencv2/opencv.hpp"
#include "preemptiveSLIC.h"

using namespace std;
using namespace cv;


// start a time measurement
double startTimeMeasure(){
  timeval tim;
  gettimeofday(&tim, 0);
  return tim.tv_sec+(tim.tv_usec/1000000.0);
}

// stop a time measurement
double stopTimeMeasure(double t1){
  timeval tim;
  gettimeofday(&tim, 0);
  double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
  return t2-t1;
}

/*
fill buffer
- each int32 is nothing-R-G-B (each 8 bit)
- row major order
*/
void fillBuffer(Mat& I, unsigned int* buffer)
{
  int dimx = I.cols;
  int dimy = I.rows;
  
  unsigned int r, g, b;
  unsigned int idx;
  unsigned int n=dimy*dimx;
  for(int i=0; i<dimy; i++)
  {
    for(int j=0; j<dimx; j++)
    {
      // read 
      b = I.at<cv::Vec3b>(i,j)[0];
      g = I.at<cv::Vec3b>(i,j)[1];
      r = I.at<cv::Vec3b>(i,j)[2];
      
      // write in row major order      
      idx = dimx*i+j;
      //buffer[idx] = R + G*256 + R*65536;
      buffer[idx] = b + (g<<8) + (r<<16);      
    }
  }
}

void fillMat(Mat& L, int* labels, int dimx, int dimy)
{
  // label image L
  L = Mat(dimy, dimx, CV_32SC1);
  for(int i=0; i<dimy; i++)
  {
    for(int j=0; j<dimx; j++)
    {
      int idx = dimx*i+j; // row major order
      L.at<int>(i,j) = labels[idx];
    }
  }
}

void getBoundaryImage(Mat& B, Mat& L, int dimx, int dimy)
{
  // boundary image B
  B = Mat::zeros(dimy, dimx, CV_8UC1);
  for(int i=1; i<dimy-1; i++)
  {
    for(int j=1; j<dimx-1; j++)
    {
      if( L.at<int>(i,j)!=L.at<int>(i+1,j) || L.at<int>(i,j)!=L.at<int>(i,j+1))
        B.at<uchar>(i,j)=1;
    }
  }    
}

void getOverlayedImage(Mat& R, Mat& B, Mat& I)
{
  int dimx = I.cols;
  int dimy = I.rows;

  // overlayed image  
  I.copyTo(R);
  for(int i=1; i<dimy-1; i++)
  {
    for(int j=1; j<dimx-1; j++)
    {
      if( B.at<uchar>(i,j) )
      {
        R.at<cv::Vec3b>(i,j)[0] = 0;
        R.at<cv::Vec3b>(i,j)[1] = 0;
        R.at<cv::Vec3b>(i,j)[2] = 255;
      }
    }
  } 
}


int main()
{
  cout << "PreemptiveSLIC demo, Peer Neubert, 2014"<<endl;
  cout << "Going to call PreemptiveSLIC 300 times"<<endl;
  
  // parameters
  int N = 1000;// wished number of segments
  double compactness = 10;
  
  bool slic_flag = 1;
  bool watershed_flag = 1;
  
  // load image
  Mat I = imread("lena.bmp");
    
  for(float compactness=0; compactness<=20; compactness+=10)
  {
    // process
    double t=startTimeMeasure();
    Mat seeds;    
    int *labels_preemptiveSLIC;
    PreemptiveSLIC preemptiveSLIC;
    for(int i=0; i<100; i++)
      preemptiveSLIC.preemptiveSLIC(I, N, compactness*2.5, labels_preemptiveSLIC, seeds);      
    t = stopTimeMeasure(t)/100;   
    cout<<"PreemptiveSLIC: compactness="<<compactness*2.5 <<",  time in ms: "<<t*1000<<endl;
           
    // label image L
    int dimx = I.cols;
    int dimy = I.rows;
    Mat L;
    fillMat(L, labels_preemptiveSLIC, dimx, dimy);
    
    // boundary image B
    Mat B;
    getBoundaryImage(B, L, dimx, dimy);
    
    // overlay image          
    Mat R;
    getOverlayedImage(R, B, I);
    
    // show
    stringstream filename; 
    filename << "PreemptiveSLIC_compactness_"<<compactness<<".png";
    imshow(filename.str(), R);
  }

  // press any to exit
  int k = waitKey(0);
 
}
