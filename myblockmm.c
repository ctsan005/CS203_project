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
    double **a, **b, **c;
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

  for(i = 0 ; i < number_of_threads ; i++)
  {
    tinfo[i].a = a;
    tinfo[i].b = b;
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
void *mythreaded_vector_blockmm(void *t)
{
  int i,j,k, ii, jj, kk, x;
  // register __m256d va,va2, vb,vb2, vc,vc2,vc3,vc4,vc5,vc6,vc7,vc8;
  register __m256d va,va2, vb,vb2, vc,vc2,vc3,vc4;
  struct thread_info tinfo = *(struct thread_info *)t;
  int number_of_threads = tinfo.number_of_threads;
  int tid =  tinfo.tid;
  double **a = tinfo.a;
  double **b = tinfo.b;
  double **c = tinfo.c;
  int ARRAY_SIZE = tinfo.array_size;
  int n = tinfo.n;
  //if the block size is too big, reduce it
  while(n >64 && n%8 == 0){
    n = n/2;
  }
  for(i = (ARRAY_SIZE/number_of_threads)*(tid); i < (ARRAY_SIZE/number_of_threads)*(tid+1); i+=ARRAY_SIZE/n)
  {
    for(j = 0; j < ARRAY_SIZE; j+=(ARRAY_SIZE/n))
    {
      for(k = 0; k < ARRAY_SIZE; k+=(ARRAY_SIZE/n))
      {        
         for(ii = i; ii < i+(ARRAY_SIZE/n); ii+=2)
         {
            for(jj = j; jj < j+(ARRAY_SIZE/n); jj+=VECTOR_WIDTH*2)
            {
                    vc = _mm256_load_pd(&c[ii][jj]);
                    vc2 = _mm256_load_pd(&c[ii][jj+4]);
                    vc3 = _mm256_load_pd(&c[ii+1][jj]);
                    vc4 = _mm256_load_pd(&c[ii+1][jj+4]);
                    // vc5 = _mm256_load_pd(&c[ii+2][jj]);
                    // vc6 = _mm256_load_pd(&c[ii+2][jj+4]);
                    // vc7 = _mm256_load_pd(&c[ii+3][jj]);
                    // vc8 = _mm256_load_pd(&c[ii+3][jj+4]);
                    // vc3 = _mm256_load_pd(&c[ii][jj+8]);
                    // vc4 = _mm256_load_pd(&c[ii][jj+12]);
                    
                for(kk = k; kk < k+(ARRAY_SIZE/n); kk++)
                {
                        va = _mm256_broadcast_sd(&a[ii][kk]);                        
                        vb = _mm256_load_pd(&b[kk][jj]);
                        vc = _mm256_add_pd(vc,_mm256_mul_pd(va,vb));
                        va2 = _mm256_broadcast_sd(&a[ii+1][kk]);
                        vb2 = _mm256_load_pd(&b[kk][jj+4]);
                        vc2 = _mm256_add_pd(vc2,_mm256_mul_pd(va,vb2));

                        vc3 = _mm256_add_pd(vc3,_mm256_mul_pd(va2,vb));
                        vc4 = _mm256_add_pd(vc4,_mm256_mul_pd(va2,vb2));

                        // va = _mm256_broadcast_sd(&a[ii+2][kk]);

                        // vc5 = _mm256_add_pd(vc5,_mm256_mul_pd(va,vb));
                        // vc6 = _mm256_add_pd(vc6,_mm256_mul_pd(va,vb2));     

                        // va2 = _mm256_broadcast_sd(&a[ii+3][kk]);

                        // vc7 = _mm256_add_pd(vc7,_mm256_mul_pd(va2,vb));
                        // vc8 = _mm256_add_pd(vc8,_mm256_mul_pd(va2,vb2));                    
                                                
                 }
                     _mm256_store_pd(&c[ii][jj],vc);
                     _mm256_store_pd(&c[ii][jj+4],vc2);
                     _mm256_store_pd(&c[ii+1][jj],vc3);
                     _mm256_store_pd(&c[ii+1][jj+4],vc4);
                    //  _mm256_store_pd(&c[ii+2][jj],vc5);
                    //  _mm256_store_pd(&c[ii+2][jj+4],vc6);
                    //  _mm256_store_pd(&c[ii+3][jj],vc7);
                    //  _mm256_store_pd(&c[ii+3][jj+4],vc8);
                    //  _mm256_store_pd(&c[ii][jj+8],vc3);
                    //  _mm256_store_pd(&c[ii][jj+12],vc4);
            }
          }
      }
    }
  }  
}
