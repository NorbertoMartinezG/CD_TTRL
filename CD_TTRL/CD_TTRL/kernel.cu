//////------------------------------ 800 EVENTS AND STREAMS  ---------------------------------------------------------------------------------  

/*
- Events
- Event API
- Event example
- Pinned memory
- Streams
- Stream API
- single stream
- multiple streams

802- Events - how to measure performance?

- use profiler (times only kernel duration + other invocations)
- Cuda events (marca de tiempo que se registra en la gpu)
	- event = timestamp
	- Timestamp recorded on the GPU
	- Invoked from the CPU side

803 - Event API
	- cudaEvent_t
	- cudaEventCreate(&e)
	- cudaEventRecord(e, 0)
	- cudaEventSynchronize(e)
	- cudaEventElapsedTime(&f, start, stop)




*/



//////------------------------------ 700 ATOMIC OPERATIONS  ---------------------------------------------------------------------------------  
/*
SUMMARY
- evita que las operaciones de los hilos sean interrumpidas por otros hilos que deben esperar
-CUDA supports several atomic operations
	-atomicAdd()
	-atomicOr()
	-atomicMin()... etc.

-Atomics incur a heavy performance penalty


x++ is a read-modify-write operation
- Read x into a register
- increment register value
- Write register back into x
- Effectively { temp 0 x; temp = temp + ; x = temp; }

if twoo threads do x++

- Each thread has its own temp (say t1 and t2)
- { t1 = x; t1 = t1+1; x = t1; }
- { t2 = x; t2 = t2+1; x = t2; }
(RACE CONDITION: THE THREAD THAT WRITES TO X FIRST WINS)

703 - atomic functions

Problema: muchos subprocesos acceden a la misma ubicación de memoria
Las operaciones atómicas garantizan que solo un hilo pueda acceder a la ubicación
Alcance de la cuadrícula!

atomicOp(x,y)
t1 = *x;		//read
t2 = t1 OP y;	//modify
*a = t2;		//write



*/

// 704 ATOMIC SUM
//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include "sm_20_atomic_functions.h"
//
//#include <stdio.h>
//#include <iostream>
//using namespace std;
//
//__device__ int dSum = 0;
//
//__global__ void sum(int* d)
//{
//	int tid = threadIdx.x;
//	//dSum += d[tid];
//	//IMPLEMENTANDO SUMA ATOMICA
//	atomicAdd(&dSum, d[tid]);
//}
//
//
//int main()
//{
//	const int count = 128;
//	const int size = sizeof(int) * count;
//
//	int h[count];
//	for (int i = 0; i < count; i++)
//	{
//		h[i] = i + 1;
//	}
//
//	int* d;
//	cudaMalloc(&d, size);
//	cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);
//	sum << <1, count >> > (d);
//
//	int hSum;
//	cudaMemcpyFromSymbol(&hSum, dSum, sizeof(int));
//	cout << "The sum of numbers from 1 to " << count << " is: " << hSum << endl;
//
//	cudaFree(d);
//	return 0;
//}

//705 Monte carlo Pi.
//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include "curand.h"
//
//#include <stdio.h>
//#include <iostream>
//#include <iomanip> // precision numerica
//
//using namespace std;
//
//__device__ int dCount = 0; //la cuenta se inicializa en 0
//
////kernel
//__global__ void countPoints(const float* xs, const float* ys) 
//{
//	int idx = blockIdx.x * blockDim.x + threadIdx.x; // obteniendo indice
//
//	float x = xs[idx] - 0.5f; 
//	float y = ys[idx] - 0.5f;
//	//Evaluacion de la posicion del circulo
//	int n = sqrtf(x * x + y * y) > 0.5f ? 0 : 1; // 0 fuera del circulo, 1 dentro del circulo
//
//	// operacion atomica
//	atomicAdd(&dCount, n);
//
//	//A VECES HAY QUE VERIFICAR QUE curand.lib EXISTA (LINKER-INPUT-ADDITIONAL DEPENDENCIES)
//
//}
//
//
//int main()
//{
//	const int count = 512 * 512;
//	const int size = count * sizeof(float);
//	cudaError_t cudaStatus;
//	curandStatus_t curandStatus;
//	curandGenerator_t gen; // generador
//
//	//generando matriz de 512 * 512
//	curandStatus = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
//	curandSetPseudoRandomGeneratorSeed(gen, time(0)); // generador de semilla
//
//	//generar 2 matrices
//	float* x;
//	float* y;
//	cudaStatus = cudaMalloc(&x, size);
//	cudaStatus = cudaMalloc(&y, size);
//
//	//generando datos aleatorios uniformes
//	curandStatus = curandGenerateUniform(gen, x, count);
//	curandStatus = curandGenerateUniform(gen, y, count);
//	
//	// contar puntos
//	countPoints << <512, 512 >> > (x, y);
//
//	int hCount;
//	cudaMemcpyFromSymbol(&hCount, dCount, sizeof(int));
//
//	cudaFree(x);
//	cudaFree(y);
//
//	cout << setprecision(12) << "pi is aproximately " << (4.0f * (float)hCount) / ((float)count) << endl;
//
//
//}
//


//////------------------------------ 600 THREAD COOPERATION AND SYNCHRONIZATION  ---------------------------------------------------------------------------------  
/*
-Interaccion entre hilos
Los subprocesos pueden tardar diferentes cantidades de tiempo en completar una parte de un cálculo.
A veces, desea que todos los hilos lleguen a un punto en particular antes de continuar con su trabajo.
Cuda ofrece una función de barrera de hilos __syncthreads ().
Un hilo que llama a __syncthreads () espera a que otros hilos lleguen a esta línea.



*/





//////------------------------------501-506 THE MANY TYPES OF MEMORY  ---------------------------------------------------------------------------------  

/*
GRPHICS PROCESSOR ARCHITECTURE

-SM-1
	SP-1
		-Texture cache
		-Constant cache
		-Shared Memory
		-Device Memory
	SP-2
		-Texture cache
		-Constant cache
		-Shared Memory
		-Device Memory
	SP-N
		-Texture cache
		-Constant cache
		-Shared Memory
		-Device Memory
-Device Memory

DEVICE MEMORY

-Grid scope ( available to all threads in all blocks in the grid )
-Aplication lifetime ( una vez que se asigna, existe hasta que se cierra la aplicacion
-Dynamic 
	-cudaMalloc() -- Asignar parte de la memoria del dispositivo y luego pasa el puntero
	-Pass pointer to kernel -- pasa el puntero a la memoria del kernel que desea ejecutar
	-cudaMemcpy() -- copia desde la memoria del host
	-cudaFree() -- Desasigna memoria

-Static
	-Declare global variable as device
		__device__ int sum = 0; -- automaticamente asigna memoria e incluso la inicializa para usarla dentro del kernel
	-Use freely within the kernel
	-Use cudaMemcpy[to/from] symbol() to copy to/from host memory
	-No need to explicity deallocate

503- Constant & texture memory
	-Memoria constante -- 64 kb
	-Declare as __constant__
	-cudaMemcpy [To/From] Symbol() to copy to/from host memory
	-Es muy util cuando todos los hilos leen la misma ubicacion

504 -Shared Memory
	-compartir solo entre hilos del mismo bloque
	-no se puede compartir entre bloques		
*/

//EJEMPLO SHARED MEMORY CON OPERACION REDUCE

 /*suma paralela
 Sumar todos los elementos en un vector.*/

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <iostream>
//
//using namespace std;
//
//__global__ void sumSingleBlock(int* d) // kernel 
//{
//	extern __shared__ int dcopy[]; //SE AGREGO PARA SHARED COPY
//	int tid = threadIdx.x; //256 sumSingleBlock << <1, count / 2 >> > (d) **count = 512
//	dcopy[tid * 2] = d[tid * 2];		//SE AGREGO PARA SHARED COPY
//	dcopy[tid * 2+1] = d[tid * 2+1];	//SE AGREGO PARA SHARED COPY
//
//	//recuento de hilos  tc - number of participating threads
//	//recuento de subprocesos
//	//blockDim.x = 256 // tc se va dividiendo entre 2 tc >>=1
//	//stepSize aumenta en potencia de 2 stepSize <<= 1 == 1,2,4,8,16,32,64
//	for (int tc = blockDim.x, stepSize = 1; tc > 0; stepSize <<= 1, tc >>=1)   // >>= el valor se desplaza hacia la derecha en uno
//	{
//		//treat must be all
//		if (tid < tc) // 256 < 256   // 
//		{
//			int pa = tid * stepSize * 2; // 256 * 1 * 2 
//			int pb = pa + stepSize;		 // 512 + 1
//			//d[pa] += d[pb];		// d[512] += d[512] + d[513]  
//			dcopy[pa] += dcopy[pb];		// d[512] += d[512] + d[513]  SE AGREGO PARA SHARED COPY
//		}
//
//		// SE AGREGO PARA SHARED COPY
//		if (tid == 0)
//		{
//			d[0] = dcopy[0];
//		}
//	}
//
//}
//
//int main()
//{
//	const int count = 32;
//	const int size = count * sizeof(int);
//
//	int h[count];
//	for (int i = 0; i < count; i++)
//	{
//		h[i] = i + 1;  // rellenar vector origen del 1 al 512
//	}
//
//	// asignar memoria
//	int* d;
//	cudaMalloc(&d, size);
//	cudaMemcpy(d, h, size, cudaMemcpyHostToDevice); // copiar de host a device(gpu)
//
//	// SE AUMENTA ",size" a la instruccion para SHARED MEMORY
//	sumSingleBlock << <1, count / 2, size >> > (d); // 1 = bloque 1 , count/2 = numero de hilos, d = conservar valores de d
//
//	int result;
//	cudaMemcpy(&result, d, sizeof(int), cudaMemcpyDeviceToHost);
//
//	cout << "Sum is " << result << endl;
//	cout << count << endl;
//	cudaFree(d);
//	return 0;
//
//}






//////------------------------------407 Scan  ---------------------------------------------------------------------------------  
// /*suma paralela
// Otra forma de Sumar todos los elementos en un vector.*/
// 
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <iostream>
//
//using namespace std;
//
//__global__ void runningSum(int* d)
//{
//	int threads = blockDim.x;
//	int tid = threadIdx.x;
//
//	//tc - total number of threads allowed
//	for (int tc = threads, step = 1; tc > 0; step *= 2)
//	{
//		// check if thread is allowed to do things
//		if (tid < tc)
//		{
//			d[tid + step] += d[tid];
//		}
//		tc -= step;
//	}
//
//
//
//}
//
//
//int main()
//{
//	const int count = 5;
//	const int size = count * sizeof(int); // esta valor es igual a 64
//
//	int h[count];
//	for (int i = 0; i < count; i++)
//	{
//		h[i] = i + 1;
//	}
//
//	int* d; // contenedor no inicializado
//	cudaMalloc(&d, size);
//	cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);
//
//	runningSum << <1, count - 1 >> > (d);
//
//	cudaMemcpy(h, d, size, cudaMemcpyDeviceToHost);
//
//	for(int i = 0; i < count; ++i)
//		cout << h[i] << '\t';
//
//	cudaFree(d);
//
//	return 0;
//	
//}



////------------------------------406 Reduce ---------------------------------------------------------------------------------  
// suma paralela
// Sumar todos los elementos en un vector.

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <iostream>
//
//using namespace std;
//
//
//__global__ void sumSingleBlock(int* d) // kernel 
//{
//	int tid = threadIdx.x; //256 sumSingleBlock << <1, count / 2 >> > (d) **count = 512
//
//	//recuento de hilos  tc - number of participating threads
//	//recuento de subprocesos
//	//blockDim.x = 256 // tc se va dividiendo entre 2 tc >>=1
//	//stepSize aumenta en potencia de 2 stepSize <<= 1 == 1,2,4,8,16,32,64
//	for (int tc = blockDim.x, stepSize = 1; tc > 0; stepSize <<= 1, tc >>=1)   // >>= el valor se desplaza hacia la derecha en uno
//	{
//		//treat must be all
//		if (tid < tc) // 256 < 256   // 
//		{
//			int pa = tid * stepSize * 2; // 256 * 1 * 2 
//			int pb = pa + stepSize;		 // 512 + 1
//			d[pa] += d[pb];		// d[512] += d[512] + d[513]  
//		}
//	}
//
//
//
//}
//
//
//int main()
//{
//	const int count = 32;
//	const int size = count * sizeof(int);
//
//	int h[count];
//	for (int i = 0; i < count; i++)
//	{
//		h[i] = i + 1;  // rellenar vector origen del 1 al 512
//	}
//
//	// asignar memoria
//	int* d;
//	cudaMalloc(&d, size);
//	cudaMemcpy(d, h, size, cudaMemcpyHostToDevice); // copiar de host a device(gpu)
//
//	sumSingleBlock << <1, count / 2 >> > (d); // 1 = bloque 1 , count/2 = numero de hilos, d = conservar valores de d
//
//	int result;
//	cudaMemcpy(&result, d, sizeof(int), cudaMemcpyDeviceToHost);
//
//	cout << "Sum is " << result << endl;
//	cout << count << endl;
//	cudaFree(d);
//	return 0;
//
//}


////------------------------------404 Gather--recopilar ---------------------------------------------------------------------------------  
// NO FUNCIONA
//BlackScholes

//#include "cuda_runtime.h" 
//#include "device_launch_parameters.h"
//#include "curand.h"
//
//#define _USE_MATH_DEFINES
//#include <iostream>
//#include <ctime>
//#include <cstdio>
//#include <math.h>
//
//using namespace std;
//
//__device__ __host__ __inline__ float N(float x)
//{
//	return 0.5 + 0.5 * erf(x * M_SQRT1_1);
//}
//
//__device__ __host__ void price(float k, float s, float t, float r, float v, float* c, float* p)
//{
//	float srt = v * sqrtf(t);
//	float d1 = (logf(s/k)+(r+0.5*v*v)*t) / srt;
//	float d2 = d1 - srt;
//	float kert = k * expf(-r * t);
//	*c = N(d1) * s - N(d2) * kert;
//	*p = kert - s + *c;
//}
//
//__global__ void price(float* k, float* s, float* t, float* r, float* v, float c, float* p)
//{
//	int idx = threadIdx.x;
//	price(k[idx], s[idx], t[idx], r[idx], v[idx], &c[idx], &p[idx], );
//}
//
//int main()
//{
//	const int count = 512; // numero de elementos a los que debemos poner precio
//	const int size = count * sizeof(float);
//
//	float* args[5];
//	curandGenerator_t gen; // Generador
//
//	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);// inicializador
//	
//	for (int i = 0; i < 5; i++)
//	{
//		cudaMalloc(&args[i], size);
//		curandGenerateUniform(gen, args[i], count);
//	}
//
//	float* dc, * dp;
//	cudaMalloc(&dc, size);
//	cudaMalloc(&dp, size);
//
//	price << <1, count >> > (args[0], args[1], args[2], args[3], args[4], dc, dp);
//
//	return 0;
//}


//505 - Resumen

/* - DECLARATION				MEMORY			SCOPE			LIFETIME		SLOWDOWN
	int foo;					register		Thread			kernel			1x
	int foo[10];				local			Thread			kernel			100x
	__shared__ int foo;			Shared			Block			kernel			1x
	__device__ int foo;			global			Grid			Application		100x
	__constant__ int foo;		constant		Grid			Application		1x


*/



////------------------------------401-403 patrones de computacion paralela------------------------------------------------------------------------
// AÑADIR A -- LINKER-INPUT-ADDITIONAL DEPENDECIES -- EN CONFIGURACION LA LIBRERIA CURAND.H
//- Element Addressing
//- Map
//- Gather
//- Scatter
//- Reduce
//- Scan

//Ejemplos
//
//- 1 block, N threads -> htreadldx.x
//- 1 block, MxN threads -> threadldx.y * blockDim.x + threadldx.x
//- N blocks, M threads -> blockldx.x * gridDim.x + threadldx.x

//--  MAP  --
//Aplicar una funcion a cada valor en la entrada

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include "curand.h"
//
//#include <iostream>
//#include <ctime>
//#include <cstdio>
//
//
//using namespace std;
//
//
//__global__ void addTen(float* d, int count)
//{
//	int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z; //calculando el indice de un elemento en un espacio de 6 dimensiones(calcular el numero de subprocesos por bloque que existen)
//	int treadPosInBlock = threadIdx.x +  // posicion del hilo (blckDim.x = bloque de dimension) // Tres dimensiones
//		blockDim.x * threadIdx.y +
//		blockDim.x * blockDim.y * threadIdx.z;
//	int blckPosInGrid = blockIdx.x +  // calculo de la posicion del bloque en una cuadricula
//		gridDim.x * blockIdx.y +
//		gridDim.x * gridDim.y * blockIdx.z;
//
//	int tid = blckPosInGrid * threadsPerBlock + treadPosInBlock; // posicion del hilo
//
//	if (tid < count)
//	{
//		d[tid] = d[tid] + 10;
//	}
//
//}	
//
//int main() {
//
//	GENERADOR NUMEROS ALEATORIOS
//	curandGenerator_t gen; // genera numeros aleatorios
//	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32); // inicializar generador
//	curandSetPseudoRandomGeneratorSeed(gen, time(0));// valor semilla
//	const int cantidad = 123456;//numero de valores a inicializar
//	const int size = cantidad * sizeof(float);
//	float *d; // puntero donde estara almacenado
//	float h[cantidad]; //matriz
//	cudaMalloc(&d, size);
//	curandGenerateUniform(gen, d, cantidad);
//
//	 dimensiones kernel
//	dim3 block(8, 8, 8); // bloque de 512 
//	dim3 cuadricula(16, 16);
//
//	addTen <<< cuadricula, block >>> (d, cantidad); //inicializamos el kernel
//	
//	cudaMemcpy(h, d, size, cudaMemcpyDeviceToHost); //copiar valores resultados desde kernel
//
//	cudaFree(d); //liberar memoria puntero
//
//	for (int i = 0; i < 100; i++)
//	{
//		cout << h[i] << endl;
//	}
//
//	return 0;
//}


////------------------------------308 Devide Introspection----------------------------------------------------

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <stdio.h>
//#include <iostream>
//using namespace std;
//
//int main() {
//	int count;
//	cudaGetDeviceCount(&count);  //numero de dispositivos o gpu's
//
//	cudaDeviceProp prop;	//informacion sobre el dispositivo
//
//	for (int i = 0; i < count; i++)
//	{
//		cudaGetDeviceProperties(&prop, i);
//		cout << "Device " << i << ": " << prop.name << endl; // nombre del dispositivo
//		cout << "Compute capability: " << prop.major << "." << prop.minor << endl; // capacidad de calculo
//
//		cout << "Maximum grid dimensions: (:" <<
//			prop.maxGridSize[0] << " x " <<
//			prop.maxGridSize[1] << " x " <<
//			prop.maxGridSize[2] << ") " << endl; // dimensiones maximas de cuadricula y bloque
//
//		cout << "Maximum block dimensions: (:" <<
//			prop.maxThreadsDim[0] << " x " <<
//			prop.maxThreadsDim[1] << " x " <<
//			prop.maxThreadsDim[2] << ") " << endl; // dimensiones maximas de cuadricula y bloque
//
//
//
//	}
//
//	return 0;
//}


////------------------------------306-7 Error handling----------------------------------------------------
/*
Para definir la ejecucion
<<<a,b>>>
a = blocks
b = threads
Realmente son 3 dimensiones (a x b x c)

dim3
conversion automatica de <<<a,b>>> = (a,1,1) por (b,1,1)

blockldx	=		donde estamos en la cuadrícula
gridDim		=		tamaño de la cuadricula
threadldmx	=		posición del hilo actual en el bloque de hilo
blockDim	=		Tamaño del bloque de hilo

max_threads_per_block =	 512
max_threads_per_multiprocessor = 1024

VERIFICAR ERRORES EN GPU. (regularmente no los marca a menos que se supervicen los status)
cudaSuccess
cudaGetErrorString()
cuRAND tiene curandStatus_t


*/


//------------------------------303-4 EXECUTION MODEL----------------------------------------------------
/*

LOCATION QUALIFIERS
___global___ === Define el Kernel, corre en la GPU, se llama desde CPU, recibe argumentos <<<dim3>>>
___device___ ===				 , corre en la GPU, se llama desde GPU, se pueden definir variables dentro de GPU
___host___   ===				 , corre en la CPU , se llama desde CPU

 -SE PUEDEN MEZCLAR

EXECUTION MODEL
 sumArrayGpu << <1, count >> > (da, db, dc); // el error señalado aqui no es un error realmente es fallo de analisis del VS
 //count hace referencia a los hilos EL 1 SE REFIERE A UN BLOQUE Y EL COUNT =5 SE REFIERE A 5 HILOS


*/


//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <stdio.h>
//#include <iostream>
//using namespace std;
//
//
//void sumArray(int* a1, int* b1, int* c1, int count1) {
//	for (int i = 0; i < count1; i++)
//	{
//		c1[i] = a1[i] + b1[i];
//	}
//}
//
////modificanco para usar hilos en gpu global se invoca en el CPU pero se ejecuta en GPU
//__global__ void sumArrayGpu(int* a, int* b, int* c) { //GLOBAL -- FUNCION HACIA GPU	
//	int i = threadIdx.x; // indice de hilo
//	c[i] = a[i] + b[i];
//}
//
//void main()
//{
//	const int count1 = 5;
//	int a1[] = { 1,2,3,4,5 };
//	int b1[] = { 10,20,30,40,50 };
//	int c1[count1];
//
//	sumArray(a1, b1, c1, count1);
//
//	//Imprimir
//	for (int i = 0; i < count1; i++)
//	{
//		cout << "posicion: " << i << " corresponde a: " << c1[i] << endl;
//	}
//
//	//----- LO MISMO PERO CON GPU ---
//
//	const int count = 5;
//	const int size = count * sizeof(int); //numero de elementos multiplicado por su tamaño para asignar memoria en GPU
//	//int a[] = { 1,2,3,4,5 };
//	int ha[] = { 1,2,3,4,5 }; // el cambio de nombre es para especificar que esta en el HOST
//	//int b[] = { 10,20,30,40,50 };
//	int hb[] = { 10,20,30,40,50 };
//	//int c[count];
//	int hc[count];
//
//	int* da, * dc, * db;//asignando memoria CPU
//	cudaMalloc(&da, size);//asignando memoria en GPU
//	cudaMalloc(&db, size);//asignando memoria en GPU
//	cudaMalloc(&dc, size);//asignando memoria en GPU
//
//	//copiando datos a GPU
//	//cudaMemcpy(da, ha, size, cudaMemcpyKind::cudaMemcpyHostToDevice); // instruccion completa
//	cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice); // instruccion corta
//	cudaMemcpy(db, hb, size, cudaMemcpyHostToDevice); // instruccion corta
//
//	//el 1 se refiere a un bloque
//	//count hace referencia a los hilos EL 1 SE REFIERE A UN BLOQUE Y EL COUNT =5 SE REFIERE A 5 HILOS
//	sumArrayGpu << <1, count >> > (da, db, dc); // el error señalado aqui no es un error realmente es fallo de analisis del VS
//
//	//Recuperando datos desde GPU
//	cudaMemcpy(hc, dc, size, cudaMemcpyDeviceToHost); //
//
//	//Imprimir
//	for (int i = 0; i < count1; i++)
//	{
//		cout << "posicion: " << i << " corresponde a: " << hc[i] << endl;
//	}
//
//}


////------------------------------303 HELLO CUDA----------------------------------------------------
//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <stdio.h>
//#include <iostream>
//using namespace std;
//
//
//void sumArray(int* a1, int* b1, int* c1, int count1) {
//	for (int i = 0; i < count1; i++)
//	{
//		c1[i] = a1[i] + b1[i];
//	}
//}
//
////modificanco para usar hilos en gpu global se invoca en el CPU pero se ejecuta en GPU
//__global__ void sumArrayGpu(int* a, int* b, int* c) { //GLOBAL -- FUNCION HACIA GPU	
//	int i = threadIdx.x; // indice de hilo
//	c[i] = a[i] + b[i];
//}
//
//void main()
//{
//	const int count1 = 5;
//	int a1[] = { 1,2,3,4,5 };
//	int b1[] = { 10,20,30,40,50 };
//	int c1[count1];
//
//	sumArray(a1, b1, c1, count1);
//
//	//Imprimir
//	for (int i = 0; i < count1; i++)
//	{
//		cout << "posicion: " << i << " corresponde a: " << c1[i] << endl;
//	}
//
//	//----- LO MISMO PERO CON GPU ---
//
//	const int count = 5;
//	const int size = count * sizeof(int); //numero de elementos multiplicado por su tamaño para asignar memoria en GPU
//	//int a[] = { 1,2,3,4,5 };
//	int ha[] = { 1,2,3,4,5 }; // el cambio de nombre es para especificar que esta en el HOST
//	//int b[] = { 10,20,30,40,50 };
//	int hb[] = { 10,20,30,40,50 };
//	//int c[count];
//	int hc[count];
//
//	int* da, * dc, * db;//asignando memoria CPU
//	cudaMalloc(&da, size);//asignando memoria en GPU
//	cudaMalloc(&db, size);//asignando memoria en GPU
//	cudaMalloc(&dc, size);//asignando memoria en GPU
//
//	//copiando datos a GPU
//	//cudaMemcpy(da, ha, size, cudaMemcpyKind::cudaMemcpyHostToDevice); // instruccion completa
//	cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice); // instruccion corta
//	cudaMemcpy(db, hb, size, cudaMemcpyHostToDevice); // instruccion corta
//
//	//el 1 se refiere a un bloque
//	//count hace referencia a los hilos
//	sumArrayGpu << <1, count >> > (da, db, dc); // el error señalado aqui no es un error realmente es fallo de analisis del VS
//
//	//Recuperando datos desde GPU
//	cudaMemcpy(hc, dc, size, cudaMemcpyDeviceToHost); //
//
//	//Imprimir
//	for (int i = 0; i < count1; i++)
//	{
//		cout << "posicion: " << i << " corresponde a: " << hc[i] << endl;
//	}
//
//}


//
////------------------------------PLANTILLA POR DEFECTO---------------------------------------------
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
//
//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//    int i = threadIdx.x;
//    c[i] = a[i] + b[i];
//}
//
//int main()
//{
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 1000, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
//}
//
//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}
