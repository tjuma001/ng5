#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <emmintrin.h>
#include <sys/time.h>

//Used for dividing the intervals of the graph into segments
//and to define number of threads and number of cores.
long N = (2496/(192))*1024;

long doPrint = 0; 

struct timeval start, end;

void print(double* a, long N) {
   if (doPrint) {
   long i;
   for (i = 0; i < N; ++i)
      printf("%f ", a[i]);
   printf("\n");
   }
}  

//Method used to calculate start time when normal/gpu method is called
void starttime() {
  gettimeofday( &start, 0 );
}

//End time when normal/gpu method is finished
void endtime(const char* c) {
   gettimeofday( &end, 0 );
   double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
   printf("%s: %f ms\n", c, elapsed); 
}

//Start time, formatting
void init(double* a, long N, const char* c) {
  printf("***************** %s **********************\n", c); 
  print(a, N);
  starttime();
}

//Endtime, formatting
void finish(double* a, long N, const char* c) {
  endtime(c);
  print(a, N);
  printf("***************************************************\n");
}

//Host Continuous function over [a,b] 
double f (double x) {
    return 128 * sin(x/256);
}

//Device  Continuous function over [a,b] 
__device__ double ff (double x) {
    return 128 * sin(x/256);
}
//Device bisection method
__device__ void doBisectionGPU (double a, double b, double t1, long maxNumOfIterations){
	
    double p1 = 0.0;
//Step 1
    long i = 0;
	// Left y-axis point of current interval [a,b]
    double fOfA = ff(a);

//Step 2
    while (i <= maxNumOfIterations) {
//Step 3
        double p = ((a + b) / 2);
        double fOfP = ff(p);

        if (fabs((p - p1) / p) < t1 && (p != p1)){
            //printf("Iteration no. %3d X = %7.5f\n", i, p);
            break;
        }

//Step 5
        i = i + 1;

//Step 6
        if (fOfA * ff(p) > 0) {
            a = p;
            fOfA = ff(p);
        } 
		else
            b = p;

        // update p1
        p1 = p;

//Step 7
        if (i == maxNumOfIterations) {
            printf ("Method failed after %ld iterations\n",
                               maxNumOfIterations);
        }
    }
}	
// Device parallelized N-section method
__global__ void gpu_doBisection(long N, double aStart, double h) {

   long element = blockIdx.x*blockDim.x + threadIdx.x;
   
   // As long as element is less than number of threads, step into if statement
   if (element < N){
	   //f(X) times f(Xi+1) = result
	   double result = ff((aStart+element*h)) * ff((aStart+(element+1)*h));
	   
	   // Return only negative results indexes
	   if(result<0){
		   //Calculate left and right enpoint of negative results at which roots lay.
		  // Left Xi and Right Xi+1 endpoints for each N
		  double aSmaller = aStart + (element*h);
		  double bSmaller = aStart + ((element+1)*h);
		  doBisectionGPU (aSmaller, bSmaller, 0.0000001, 500); 
	   }
   }
}

// Host bisection method
void doBisection (double a, double b, double t1, long maxNumOfIterations1){
	
    double p1 = 0.0;
	
// Step 1
    long i = 0;
    double fOfA = f(a);

// Step 2
    while (i <= maxNumOfIterations1) {

// Step 3

        double p = ((a + b) / 2);
        double fOfP = f(p);

        if (fabs((p - p1) / p) < t1 && (p != p1)){
         //printf("Iteration no. %3ld X = %7.5f\n", i, p);
            break;
        }

// Step 5
		i = i + 1;

// Step 6
        if (fOfA * f(p) > 0) {
            a = p;
            fOfA = f(p);
        } else
            b = p;
		
        // update p1
        p1 = p;

// Step 7: Print the amount of iterations it took to find no root
        if (i == maxNumOfIterations1) {
            printf ("Method failed after %ld iterations\n",
                               maxNumOfIterations1);
        }
    }
}	
// Host CPU method
void gpu(long N) 
{
	//Declaring GPU threads and cores based on N
	long numThreads = 1024;
	long numCores = N / 1024 + 1;

	//Start and end of graph
	double aStart = 400;
	double b = 1200000000;
	
	//Brake the graph into smaller intervals
	double h = (b-aStart)/N;
	
	//Call GPU method
	gpu_doBisection<<<numCores, numThreads>>>(N, aStart, h);
   
   cudaDeviceSynchronize();
}




// Host main
int main()
{

  double* a;

  posix_memalign((void**)&a, 16,  N * sizeof(double));	
	
  // Test 1: Sequential For Loop
  
  //Start time for normal method
  init(a, N, "Normal");

  long i;
  long counter;
  
  //Normal method to find all roots given a fixed interval
  for(i = 400; i < 120000000; i=i+800){ 
  doBisection (i, i+800, 0.0000001, 500);
  counter++;
  }
  
  //Count roots
  printf("There are %ld roots\n",counter);

  //End time for normal method
  finish(a, N, "Normal"); 

  // Test 2: Vectorization
  
  //Start time for GPU method
  init(a, N, "GPU");
  
  //Call GPU
  gpu(N);  
  
  //End time for GPU method
  finish(a, N, "GPU");

  return 0;
}