#pragma once
#include <time.h>

enum RTimerTicks
{
	rtimer_sec,
	rtimer_ms,
	rtimer_us
};

class RTimer
{
	private:
	timespec tstart, tend;
	public:
	void start();
	void stop();
	size_t get(RTimerTicks tt); //returns time in ms
};
