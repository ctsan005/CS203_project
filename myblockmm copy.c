#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>
#include <x86intrin.h>
#include <sys/time.h>
#include <pthread.h>
#include "myblockmm.h"

struct thread_info
{
    int tid;
    double **a, **b, **b_t, **c;
    int array_size;
    int number_of_threads;
    int n;
};
void *mythreaded_vector_blockmm(void *t);
void *tranpose_B(void *t);

char name[128];
char SID[128];
#define VECTOR_WIDTH 4
void my_threaded_vector_blockmm(double **a, double **b, double **c, int n, int ARRAY_SIZE, int number_of_threads)
{
  int i=0;
  pthread_t *thread;
  struct thread_info *tinfo;
  struct timeval time_start, time_end;
  strcpy(name,"Chi Chiu Tsang");
  strcpy(SID,"861265376");
  thread = (pthread_t *)malloc(sizeof(pthread_t)*number_of_threads);
  tinfo = (struct thread_info *)malloc(sizeof(struct thread_info)*number_of_threads);

  double **b_t;
  int j;

  //alloc a new array for transpose 
  b_t = (double **)malloc(ARRAY_SIZE*sizeof(double *));
  for(i = 0; i < ARRAY_SIZE; i++)
  {
    b_t[i] = (double *)_mm_malloc(ARRAY_SIZE*sizeof(double),256);
  }

  // gettimeofday(&time_start, NULL);
  for(i = 0; i < ARRAY_SIZE; i += ARRAY_SIZE/n){
    for(j = 0; j < ARRAY_SIZE; j+=ARRAY_SIZE/n){
      b_t[i][j] = b[j][i];
    }
  }
  // gettimeofday(&time_end, NULL);
  // double t = ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec))/1000000.0;
  // fprintf(stderr,"Time for matrix transpose: %lf\n",t);

  // for(i = 0; i < ARRAY_SIZE; i += ARRAY_SIZE/n){
  //   printf("test printing the value in matrix transpose: %f",b_t[i][i]);
  // }

  for(i = 0 ; i < number_of_threads ; i++)
  {
    tinfo[i].a = a;
    tinfo[i].b = b;
    tinfo[i].b_t = b_t;
    tinfo[i].c = c;
    tinfo[i].tid = i;
    tinfo[i].number_of_threads = number_of_threads;
    tinfo[i].array_size = ARRAY_SIZE;
    tinfo[i].n = n;
    pthread_create(&thread[i], NULL, mythreaded_vector_blockmm, &tinfo[i]);
  }  
  for(i = 0 ; i < number_of_threads ; i++)
    pthread_join(thread[i], NULL);

  return;
}

#define VECTOR_WIDTH 4

// void *tranpose_B(void *t){
//   struct thread_info tinfo = *(struct thread_info *)t;
//   int tid =  tinfo.tid;
//   int ARRAY_SIZE = tinfo.array_size;
//   int number_of_threads = tinfo.number_of_threads;
//   double **b = tinfo.b;
//   double **b_t = tinfo.b_t;
//   int n = tinfo.n;
//   int i,j;
//   for(i = 0; i < ARRAY_SIZE; i += ARRAY_SIZE/n){
//     for(j = (ARRAY_SIZE/number_of_threads)*(tid); j < (ARRAY_SIZE/number_of_threads)*(tid+1); j+=ARRAY_SIZE/n){
//       b_t[i][j] = b[j][i];
//     }
//   }
  
// }

void *mythreaded_vector_blockmm(void *t)
{
  int i,j,k, ii, jj, kk, x;
  __m256d va, vb0,vb1,vb2,vb3, vc0,vc1,vc2,vc3;
  struct thread_info tinfo = *(struct thread_info *)t;
  int number_of_threads = tinfo.number_of_threads;
  int tid =  tinfo.tid;
  double **a = tinfo.a;
  double **b = tinfo.b_t;
  double **c = tinfo.c;
  int ARRAY_SIZE = tinfo.array_size;
  int n = tinfo.n;
  for(i = (ARRAY_SIZE/number_of_threads)*(tid); i < (ARRAY_SIZE/number_of_threads)*(tid+1); i+=ARRAY_SIZE/n)
  {
    for(j = 0; j < ARRAY_SIZE; j+=(ARRAY_SIZE/n))
    {
      for(k = 0; k < ARRAY_SIZE; k+=(ARRAY_SIZE/n))
      {        
         for(ii = i; ii < i+(ARRAY_SIZE/n); ii++)
         {
            for(jj = j; jj < j+(ARRAY_SIZE/n); jj+=VECTOR_WIDTH)
            {
                    vc0 = _mm256_set_pd(0,0,0,0);
                    vc1 = _mm256_set_pd(0,0,0,0);
                    vc2 = _mm256_set_pd(0,0,0,0);
                    vc3 = _mm256_set_pd(0,0,0,0);
                    
                for(kk = k; kk < k+(ARRAY_SIZE/n); kk+=4)
                {
                        va = _mm256_broadcast_sd(&a[ii][kk]);
                        vb0 = _mm256_load_pd(&b[jj][kk]);
                        vb1 = _mm256_load_pd(&b[jj+1][kk]);
                        vb2 = _mm256_load_pd(&b[jj+2][kk]);
                        vb3 = _mm256_load_pd(&b[jj+3][kk]);
                        vc0 = _mm256_add_pd(vc0,_mm256_mul_pd(va,vb0));
                        vc1 = _mm256_add_pd(vc1,_mm256_mul_pd(va,vb1));
                        vc2 = _mm256_add_pd(vc2,_mm256_mul_pd(va,vb2));
                        vc3 = _mm256_add_pd(vc3,_mm256_mul_pd(va,vb3));
                 }

                 vc0 = _mm256_hadd_pd(vc0,vc1);
                 vc1 = _mm256_hadd_pd(vc2,vc3);
                //  double t = ((double*)&vc0)[0];

                     _mm256_store_pd(&c[ii][jj],_mm256_set_pd(((double*)&vc0)[0] + ((double*)&vc0)[2] +c[ii][jj],((double*)&vc0)[1] + ((double*)&vc0)[3] +c[ii][jj+1],((double*)&vc1)[0] + ((double*)&vc1)[2]+c[ii][jj+2],((double*)&vc1)[1] + ((double*)&vc1)[3] +c[ii][jj+3]));
            }
          }
      }
    }
  }  
}

