#include <cuda_runtime.h>
#include <cmath>
// params, grad, m, v are device pointers

__global__ void   AdamWkernel(float* params, const float* grad, float* m, float* v, int N, float lr,
                      float beta1, float beta2, float eps, float weight_decay, int t){

                        int idx = threadIdx.x + blockIdx.x *blockDim.x;

                        if(idx >= N){
                              return;
                        }

                        params[idx] = params[idx] * (1-lr*weight_decay);
                        m[idx] = beta1 *(m[idx]) + (1-beta1)*grad[idx];
                        v[idx] = beta2 * v[idx] + (1-beta2)*(powf(grad[idx],2));

                        float m_cap = m[idx]/(1-powf(beta1,t));
                        float v_cap = v[idx]/(1-powf(beta2,t));

                        params[idx] = params[idx] - lr*m_cap/(sqrtf(v_cap)+eps); 
                        

                      }

extern "C" void solve(float* params, const float* grad, float* m, float* v, int N, float lr,
                      float beta1, float beta2, float eps, float weight_decay, int t) {

                        dim3 threads_per_block(256);
                        dim3 blocks_per_grid((N+threads_per_block.x-1)/threads_per_block.x);

                        AdamWkernel<<<blocks_per_grid,threads_per_block>>>(params,grad,m,v,N,lr,beta1,beta2,eps,weight_decay,t);

                         cudaDeviceSynchronize();
                      }
