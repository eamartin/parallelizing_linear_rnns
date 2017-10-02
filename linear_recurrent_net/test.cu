#include <inttypes.h>
#include <stdio.h>
#include <time.h>
#include "linear_recurrence.h"

#define gpuErrChk(ans) { gpuAssert2((ans), __FILE__, __LINE__); }
void gpuAssert2(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
  }
}

uint64_t nanotime(void) {
  uint64_t billion = 1000 * 1000 * 1000;
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return t.tv_nsec + billion * t.tv_sec;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "Must pass n_steps and n_dims as args.\n");
    return 1;
  }

  int n_steps = atoi(argv[1]);
  int n_dims = atoi(argv[2]);
  int n_elements = n_dims * n_steps;
  printf("Running on n_steps=%d n_dims=%d\n", n_steps, n_dims);

  float *decays = (float *) calloc(n_elements, sizeof(float));
  for (int i = 0; i < n_elements; i++) {
    decays[i] = .999;
  }
  float *d_decays;
  gpuErrChk(cudaMalloc(&d_decays, n_elements * sizeof(float)));
  gpuErrChk(cudaMemcpy(d_decays, decays, n_elements * sizeof(float),
		       cudaMemcpyHostToDevice));

  float *impulses = (float *) calloc(n_elements, sizeof(float));
  for (int i = 0; i < n_dims; i++) {
    impulses[i + 0 * n_dims] = 2.0;
  }
  float *d_impulses;
  gpuErrChk(cudaMalloc(&d_impulses, n_elements * sizeof(float)));
  gpuErrChk(cudaMemcpy(d_impulses, impulses,
		       n_elements * sizeof(float), cudaMemcpyHostToDevice));

  float *out = (float *) calloc(n_elements, sizeof(float));
  float *d_out;
  gpuErrChk(cudaMalloc(&d_out, n_elements * sizeof(float)));
  gpuErrChk(cudaMemset(d_out, 0, n_elements * sizeof(float)));

  gpuErrChk(cudaDeviceSynchronize());
  uint64_t plr_start = nanotime();
  compute_linear_recurrence(d_decays, d_impulses, NULL, d_out, n_dims, n_steps);
  gpuErrChk(cudaDeviceSynchronize());
  uint64_t plr_ns = nanotime() - plr_start;
  printf("PLR: %lu ns\n", plr_ns);

  gpuErrChk(cudaDeviceSynchronize());
  uint64_t slr_start = nanotime();
  compute_serial_linear_recurrence(d_decays, d_impulses, NULL, d_out, n_dims, n_steps);
  gpuErrChk(cudaDeviceSynchronize());
  uint64_t slr_ns = nanotime() - slr_start;
  printf("PLR: %lu ns\n", plr_ns);

  gpuErrChk(cudaMemcpy(out, d_out, n_elements * sizeof(float),
		       cudaMemcpyDeviceToHost));

  gpuErrChk(cudaFree(d_decays));
  gpuErrChk(cudaFree(d_impulses));
  gpuErrChk(cudaFree(d_out));
}
