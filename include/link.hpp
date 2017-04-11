#pragma once
#include <cstdlib> //rand()

#include "layer.hpp"

class Layer;

class Link
{
	public:
	Layer *inLayer;
	Layer *outLayer;
	double **weights;
	
	Link(Layer *_inLayer, Layer *_outLayer);
};
