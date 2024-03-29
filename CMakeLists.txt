cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(spixels LANGUAGES CXX)

# Setting build type to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

# Setting the flags for profiling information or not
if(CMAKE_BUILD_TYPE MATCHES Release)
  message(STATUS "Setting Release options")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  message(STATUS "Setting Debug options")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O1 -pg -march=native")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O1 -pg -march=native")
endif()

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support.
                         Please use a different C++ compiler.")
endif()

# Catkin dependencies
find_package(catkin REQUIRED COMPONENTS roscpp)

# Packages required to work with the library
find_package(OpenCV REQUIRED) # OpenCV
find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(OpenMP REQUIRED) # OpenMP
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_path(FFTW_INCLUDE_DIR fftw3.h  ${FFTW_INCLUDE_DIRS})
find_library(FFTW_LIBRARY fftw3 ${FFTW_LIBRARY_DIRS})

# Printing the compiling flags
message(STATUS "Compiler flags: ${CMAKE_CXX_FLAGS}")

# Defining the package
catkin_package(
    INCLUDE_DIRS pSLIC src
    LIBRARIES
    CATKIN_DEPENDS roscpp
    DEPENDS
)

include_directories(pSLIC
                    src
                    ${OpenCV_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS}
                    ${FFTW_INCLUDE_DIR})

link_directories(${OpenCV_LIBRARIES}
                 ${Boost_LIBRARIES})

###########
## Build ##
###########

# Preemptive SLIC
add_library(pSLIC
            pSLIC/preemptiveSLIC.cpp)
target_link_libraries(pSLIC
            ${OpenCV_LIBRARIES}
            )

# LBP
add_library(LBP
  STATIC
  src/LBP.cpp 
)
target_link_libraries(LBP
            ${OpenCV_LIBS} ${FFTW_LIBRARY})

# Main file
add_executable(demo_SLIC
               src/main.cc
               src/superpixels.cc
              )
target_link_libraries(demo_SLIC
                      pSLIC
                      LBP
                      ${OpenCV_LIBRARIES}
                      ${Boost_LIBRARIES}
)