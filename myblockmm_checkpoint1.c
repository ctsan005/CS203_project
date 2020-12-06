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
    double **a, **b, **c, **b_t, **c_t;
    int array_size;
    int number_of_threads;
    int n;
};
void *mythreaded_vector_blockmm(void *t);

char name[128];
char SID[128];
#define VECTOR_WIDTH 4
void my_threaded_vector_blockmm(double **a, double **b, double **c, int n, int ARRAY_SIZE, int number_of_threads)
{
  int i=0;
  pthread_t *thread;
  struct thread_info *tinfo;
  strcpy(name,"Chi Chiu Tsang");
  strcpy(SID,"861265376");
  thread = (pthread_t *)malloc(sizeof(pthread_t)*number_of_threads);
  tinfo = (struct thread_info *)malloc(sizeof(struct thread_info)*number_of_threads);

  
  double **b_t, **c_t;
   //alloc a new array for transpose 
  b_t = (double **)malloc(ARRAY_SIZE*sizeof(double *));
  c_t = (double **)malloc(ARRAY_SIZE*sizeof(double *));
  for(i = 0; i < ARRAY_SIZE; i++)
  {
    b_t[i] = (double *)_mm_malloc(ARRAY_SIZE*sizeof(double),256);
    c_t[i] = (double *)_mm_malloc(ARRAY_SIZE*sizeof(double),256);
  }

  for(i = 0; i < ARRAY_SIZE; i += 1){
    for(int j = 0; j < ARRAY_SIZE; j+=1){
      b_t[i][j] = b[j][i];
      c_t[i][j] = 0;
    }
  }

  for(i = 0 ; i < number_of_threads ; i++)
  {
    tinfo[i].a = a;
    tinfo[i].b = b;
    tinfo[i].b_t = b_t;
    tinfo[i].c = c;
    tinfo[i].c_t = c_t;
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
void *mythreaded_vector_blockmm(void *t)
{
  int i,j,k, ii, jj, kk, x;
  __m256d va, vb,vb2,vb3,vb4, vc,vc2,vc3,vc4;
  __m256d temp;
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
            for(jj = j; jj < j+(ARRAY_SIZE/n); jj+=1)
            {
                    vc = _mm256_load_pd(&c[ii][jj]);
                    // vc2 = _mm256_load_pd(&c[ii][jj+4]);
                    double vc00 = c[ii][jj];
                    
                    
                for(kk = k; kk < k+(ARRAY_SIZE/n); kk++)
                {
                        double va0 = a[ii][kk];
                        double vb0 = b[jj][kk];
                        vc00 += va0 * vb0;
                                        
                 }
                    //  _mm256_store_pd(&c[ii][jj],vc);
                    //  _mm256_store_pd(&c[ii][jj+4],vc2);
                    // temp = _mm256_hadd_pd(temp,temp);
                    // c[ii][jj] = ((double*)&temp)[0] + ((double*)&temp)[2] + c_t;

                    c[ii][jj] = vc00;
            }
          }
      }
    }
  }  
}
