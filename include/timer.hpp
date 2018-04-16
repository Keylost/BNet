#pragma once
#include <sys/timeb.h>
#include <cmath>

enum RTimerTicks
{
	rtimer_sec,
	rtimer_ms
};

class RTimer
{
	private:
	struct timeb tstart, tend;
	public:
	void start();
	void stop();
	long unsigned get(RTimerTicks tt); //returns time in ms
};
