#if !defined(_POSIX_BARRIERS) || (_POSIX_BARRIERS < 0)

#ifndef PTHREAD_BARRIER_H_
#define PTHREAD_BARRIER_H_

#include <pthread.h>

// This file provides a simple (and slow) implementation of pthread_barriers
// for Mac OS, as Mac does not implement pthread barriers, and the Legate
// implementation utilizes them when MPI is disabled.

typedef int pthread_barrierattr_t;
typedef struct {
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  int count;
  int tripCount;
} pthread_barrier_t;

inline int pthread_barrier_init(pthread_barrier_t* barrier,
                                const pthread_barrierattr_t* attr,
                                unsigned int count)
{
  if (count == 0) { return -1; }
  if (pthread_mutex_init(&barrier->mutex, 0) < 0) { return -1; }
  if (pthread_cond_init(&barrier->cond, 0) < 0) {
    pthread_mutex_destroy(&barrier->mutex);
    return -1;
  }
  barrier->tripCount = count;
  barrier->count     = 0;
  return 0;
}

inline int pthread_barrier_destroy(pthread_barrier_t* barrier)
{
  pthread_cond_destroy(&barrier->cond);
  pthread_mutex_destroy(&barrier->mutex);
  return 0;
}

inline int pthread_barrier_wait(pthread_barrier_t* barrier)
{
  pthread_mutex_lock(&barrier->mutex);
  ++(barrier->count);
  if (barrier->count >= barrier->tripCount) {
    barrier->count = 0;
    pthread_cond_broadcast(&barrier->cond);
    pthread_mutex_unlock(&barrier->mutex);
    return 1;
  } else {
    pthread_cond_wait(&barrier->cond, &(barrier->mutex));
    pthread_mutex_unlock(&barrier->mutex);
    return 0;
  }
}

#endif  // PTHREAD_BARRIER_H_
#endif  // _POSIX_BARRIERS
