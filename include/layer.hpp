#pragma once
#include <cstdlib> //rand()
#include <stdint.h>

#include "activation.hpp"
#include "link.hpp"

class Link;

enum LayerTypes
{
	INPUT = 0,
	HIDDEN = 1,
	OUTPUT = 2,
	CONVOLUTION = 3,
	SUB_SAMPLING = 4
};

class Layer
{
	public:
	LayerTypes layerType;
	ActivationFunctions AFType;
	int neuronsCount;
	double *signals; //сигналы нейронов
	double *errors; //хранит ошибки
	double *deltas; //суммарная ошибка нейрона слоя
	double *biases; //хранит биасы/смещения
	
	/*
	 * параметры только для сверточной сети 
	 */
	uint32_t kernelSize_x;
	uint32_t kernelSize_y;
	uint32_t kernelsCount;	
	uint32_t mapW;
	uint32_t mapH;
	double   **featuresMaps;
	
	Link *inLink;
	Link *outLink;
	Activation *activation;
	
	Layer(int _neuronsCount, ActivationFunctions _AFType);
	Layer(int inputW, int inputH, int _kernelSize_x, int _kernelSize_y, int _kernelsCount, ActivationFunctions _AFType);
};
