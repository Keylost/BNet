#include "link.hpp"

Link::Link(Layer *_inLayer, Layer *_outLayer)
{
	inLayer = _inLayer;
	outLayer = _outLayer;
	
	inLayer->outLink = this;
	outLayer->inLink = this;
	
	weights = new double*[inLayer->neuronsCount];
	for(int i=0;i<inLayer->neuronsCount;i++)
	{
		weights[i] = new double[outLayer->neuronsCount];
		for(int j = 0; j<outLayer->neuronsCount;j++)
		{
			weights[i][j] = (double)(rand()%1000)/2500.0;
			if(weights[i][j]==0) weights[i][j] = 0.07;
		}
	}
}
