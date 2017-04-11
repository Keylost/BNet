#include "layer.hpp"

Layer::Layer(int _neuronsCount, ActivationFunctions _AFType)
{
	AFType = _AFType;
	neuronsCount = _neuronsCount;
	signals = new double[neuronsCount];
	errors  = new double[neuronsCount];
	deltas  = new double[neuronsCount];
	biases  = new double[neuronsCount];
	
	for(int i=0;i<neuronsCount;i++)
	{
		biases[i] = (double)(rand()%1000)/2500.0;
		if(biases[i]==0) biases[i] = 0.24;
	}
	
	switch(AFType)
	{
		case LOGISTIC:
		{
			activation = new ActivationLogistic();
			break;
		}
		case BINARY:
		{
			//activation = new ActivationLogistic();
			break;
		}
		default:
		{
			activation = new ActivationLogistic();
			break;
		}
	}
}

//для сверточных слоев
Layer::Layer(int inputW, int inputH, int _kernelSize_x, int _kernelSize_y, int _kernelsCount, ActivationFunctions _AFType)
{
	/*
	 * параметры только для сверточной сети 
	 */
	kernelSize_x = _kernelSize_x;
	kernelSize_y = _kernelSize_y;
	kernelsCount = _kernelsCount;
	mapW = inputW - _kernelSize_x - 1;
	mapH = inputH - _kernelSize_y - 1;
	int _neuronsCount =  mapW * mapH;
	featuresMaps = new double*[_kernelsCount];
	for(int i=0;i<_kernelsCount;i++)
	{
		featuresMaps[i] = new double[_neuronsCount];
	}
	
	
	AFType = _AFType;
	neuronsCount = _neuronsCount;
	signals = new double[neuronsCount];
	errors  = new double[neuronsCount];
	deltas  = new double[neuronsCount];
	biases  = new double[neuronsCount];
	
	for(int i=0;i<neuronsCount;i++)
	{
		biases[i] = (double)(rand()%1000)/2500.0;
		if(biases[i]==0) biases[i] = 0.24;
	}
	
	switch(AFType)
	{
		case LOGISTIC:
		{
			activation = new ActivationLogistic();
			break;
		}
		case BINARY:
		{
			//activation = new ActivationLogistic();
			break;
		}
		default:
		{
			activation = new ActivationLogistic();
			break;
		}
	}
}
