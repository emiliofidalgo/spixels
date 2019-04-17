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
 */ 

#include "mex.h"
#include <math.h>
#include <stdio.h>

#include <vector>
#include <iostream>
#include "sys/time.h"
#include "preemptiveSLIC.h"
#include "opencv2/opencv.hpp"
#include "mex_helper.h"

using namespace std;
using namespace cv;
/*
 * call: [L] = mex_preemptiveSLIC(uint8(I), N, compactness)
 *
 * I                ... uint8, RGB image
 * N                ... int, wished number of segments
 * compactness      ... double, compactness parameter of SLIC
 * seeds            ... matrix of initial seeds, each row is [i,j], single values (can be empty)
 *                      currently, non grid ininialized seeds may cause incorrect caunting of changes
 *                      This it's redommended to ignore this parameter.
 * 
 * L ... labels 0...n
 *
 * compile with  
 *   mex mex_preemptiveSLIC.cpp ../preemptiveSLIC.cpp mex_helper.cpp -I.. $(pkg-config --cflags --libs opencv)
 */
void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[])
{
  // ============ parse input ==============

  if(nrhs!=3 && nrhs!=4)
  {
    cerr << "Error in mex_SLIC: number of input parameters is "<<nrhs<<" and should be 3 or 4\n";
    return;
  }

  // wished number of segments
  int N = (int)(mxGetScalar(prhs[1]));
  
  // wished number of segments
  double compactness = (int)(mxGetScalar(prhs[2]));
  
  Mat seeds;
  if(nrhs>=4)
    convertMx2Mat(prhs[3], seeds);
  
  // convert Matlab Matrix to uchar buffer
  //figure out dimensions
  const mwSize *dims;
  int dimx, dimy, dimc, numdims;
  dims = mxGetDimensions(prhs[0]);
  numdims = mxGetNumberOfDimensions(prhs[0]);
  if(numdims==2)
  {
    dimy = (int)dims[0]; 
    dimx = (int)dims[1];
  }
  else if(numdims==3)
  {
    dimy = (int)dims[0]; 
    dimx = (int)dims[1];
    dimc = (int)dims[2];
  }
  else
  {
    cerr << "Unsupported number of dimensions in mex_ownSLIC(): "<<numdims<<endl;
    return;
  }
  // get pointer to input data and fill buffer
  unsigned char* I = (unsigned char*) mxGetPr(prhs[0]);    
  unsigned int R, G, B;
  unsigned int idx;
  unsigned int n=dimy*dimx;
  unsigned int nn=n+n;
  Mat I_mat(dimy, dimx, CV_8UC3);
  for(int i=0; i<dimy; i++)
  {
    for(int j=0; j<dimx; j++)
    {
      // read in column major order
      idx = (dimy*j)+i;
      R = I[idx];
      G = I[idx + n];
      B = I[idx + nn];
      
      Vec3b pixel_color(B, G, R);
      I_mat.at<Vec3b>(i,j) = pixel_color;  
    }
  }
  
  // ============ process ==============
  // call SLIC function
  int* labels;
  PreemptiveSLIC preemptiveSLIC;
  preemptiveSLIC.preemptiveSLIC(I_mat, N, compactness *2.5, labels,seeds); 
  
  // ============ create output ==============  
  plhs[0] = mxCreateNumericMatrix(dimy, dimx, mxDOUBLE_CLASS, mxREAL);
  double* pointer = mxGetPr(plhs[0]);
    
  // copy data 
  int idx_cv, idx_matlab;
  for(int i=0; i<dimy; i++)
  {
    for(int j=0; j<dimx; j++)
    {
      idx_cv = dimx*i+j; // row major order index
      idx_matlab = dimy*j+i; // column major order index
      pointer[idx_matlab] = labels[idx_cv];
     }
  }
   

  // =========== garbage collection ===========
  if(labels) delete labels;
}
