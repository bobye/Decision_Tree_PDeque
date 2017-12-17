#ifndef _TIMER_H_
#define _TIMER_H_

#define BILLION  1000000000L

// Timing, count in nano seconds.
#if defined (_WIN32)
#include <Windows.h>

inline double getRealTime() {
  FILETIME tm;
  ULONGLONG t;
#if defined(NTDDI_WIN8) && NTDDI_VERSION >= NTDDI_WIN8
	/* Windows 8, Windows Server 2012 and later. ---------------- */
	GetSystemTimePreciseAsFileTime( &tm );
#else
	/* Windows 2000 and later. ---------------------------------- */
	GetSystemTimeAsFileTime( &tm );
#endif  
  t = ((ULONGLONG)tm.dwHighDateTime << 32) | (ULONGLONG)tm.dwLowDateTime;
  return (double) t / (double) BILLION;
}


#else
#include <time.h>

#ifdef __MACH__
#include <sys/time.h>
//clock_gettime is not implemented on OSX
#include <mach/clock.h>
#include <mach/mach.h>
#define CLOCK_REALTIME 0
#define CLOCK_MONOTONIC 0
inline int clock_gettime(int clk_id, struct timespec* ts) {
  clock_serv_t cclock;
  mach_timespec_t mts;
  clk_id = 0; // something stupid to get ride of warnings
  host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
  clock_get_time(cclock, &mts);
  mach_port_deallocate(mach_task_self(), cclock);
  ts->tv_sec = mts.tv_sec;
  ts->tv_nsec = mts.tv_nsec;
  return 0;
}
#endif


inline double getRealTime() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);     
  return (double) ( ts.tv_sec ) + (double) ( ts.tv_nsec ) / (double) BILLION ;
}

#endif


#endif /* _TIMER_H_ */
