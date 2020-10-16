////------------------------------308 Devide Introspection----------------------------------------------------

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
using namespace std;

int main() {
	int count;
	cudaGetDeviceCount(&count);  //numero de dispositivos o gpu's

	cudaDeviceProp prop;	//informacion sobre el dispositivo

	for (int i = 0; i < count; i++)
	{
		cudaGetDeviceProperties(&prop, i);
		cout << "Device " << i << ": " << prop.name << endl; // nombre del dispositivo
		cout << "Compute capability: " << prop.major << "." << prop.minor << endl; // capacidad de calculo

		cout << "Maximum grid dimensions: (:" <<
			prop.maxGridSize[0] << " x " <<
			prop.maxGridSize[1] << " x " <<
			prop.maxGridSize[2] << ") " << endl; // dimensiones maximas de cuadricula y bloque

		cout << "Maximum block dimensions: (:" <<
			prop.maxThreadsDim[0] << " x " <<
			prop.maxThreadsDim[1] << " x " <<
			prop.maxThreadsDim[2] << ") " << endl; // dimensiones maximas de cuadricula y bloque



	}

	return 0;
}


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
