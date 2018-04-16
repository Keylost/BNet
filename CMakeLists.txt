cmake_minimum_required(VERSION 2.8)

set(PROJECT BNet)
project(${PROJECT})

IF (WIN32)
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
set(OpenCV_DIR C:/NeuralNetworks/opencv/opencv/build)
set(OpenCV_INCLUDES ${OpenCV_DIR}/include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
endif(WIN32)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -std=c++11 -Wall")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
	set(LIBS ${LIBS}
		pthread
	)
endif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

find_package(OpenCV)

aux_source_directory(sources SOURCES)
set(SOURCES ${SOURCES}
	${PROJECT}.cpp
)

include_directories(
	include
	${OpenCV_INCLUDES}
)

add_executable(${PROJECT} ${SOURCES})

target_link_libraries(${PROJECT}
	${LIBS}
	${OpenCV_LIBS} 
)

IF (MSVC)
set(CMAKE_CXX_FLAGS_RELEASE "/MT")
set(CMAKE_CXX_FLAGS_DEBUG "/MTd")
endif(MSVC)
