#include "timer.hpp"

/*
 * Функция start() отвечает за запукс таймера
 */
void RTimer::start()
{
	ftime(&tstart);
}

/*
 * Функция stop() отвечает за остановку таймера
 */
void RTimer::stop()
{
	ftime(&tend);
}

/*
 * Функция get() возвращает время в миллисекундах между вызовами start() и stop()
 */
long unsigned RTimer::get(RTimerTicks tt)
{
	switch(tt)
	{
		case rtimer_sec: //seconds
		{
			return ((tend.time - tstart.time) + round((tend.millitm - tstart.millitm)/1000.0));
		}
		case rtimer_ms: //milliseconds
		{
			return (1000.0 * (tend.time - tstart.time) + (tend.millitm - tstart.millitm));
		}
		default: //return time in ms by default
		{
			return (1000.0 * (tend.time - tstart.time) + (tend.millitm - tstart.millitm));
		}
	}	
}
