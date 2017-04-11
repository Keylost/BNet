#include "timer.hpp"

/*
 * Функция start() отвечает за запукс таймера
 */
void RTimer::start()
{
	clock_gettime(CLOCK_REALTIME, &tstart);
}

/*
 * Функция stop() отвечает за остановку таймера
 */
void RTimer::stop()
{
	clock_gettime(CLOCK_REALTIME, &tend);
}

/*
 * Функция get() возвращает время в миллисекундах между вызовами start() и stop()
 */
size_t RTimer::get(RTimerTicks tt)
{
	switch(tt)
	{
		case rtimer_sec: //seconds
		{
			return ((tend.tv_sec-tstart.tv_sec) + (tend.tv_nsec-tstart.tv_nsec)/1000000000);
		}
		case rtimer_ms: //milliseconds
		{
			return ((tend.tv_sec-tstart.tv_sec)*1000 + (tend.tv_nsec-tstart.tv_nsec)/1000000);
		}
		case rtimer_us: //microseconds
		{
			return ((tend.tv_sec-tstart.tv_sec)*1000000 + (tend.tv_nsec-tstart.tv_nsec)/1000);
		}
		default: //return time in ms by default
		{
			return ((tend.tv_sec-tstart.tv_sec)*1000 + (tend.tv_nsec-tstart.tv_nsec)/1000000);
		}
	}	
}
