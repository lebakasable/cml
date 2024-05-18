#include <time.h>

#define ML_IMPLEMENTATION
#include "ml.h"

#define BITS 2

int main(void)
{
  srand(time(NULL));

  size_t n = 1<<BITS;
  size_t rows = n*n;
  Mat ti = mat_alloc(rows, 2*BITS);
  Mat to = mat_alloc(rows, BITS + 1);
  for (size_t i = 0; i < ti.rows; ++i) {
    size_t x = i/n;
    size_t y = i%n;
    size_t z = x + y;
    for (size_t j = 0; j < BITS; ++j) {
      MAT_AT(ti, i, j) = (x>>j)&1;
      MAT_AT(ti, i, j + BITS) = (y>>j)&1;
      MAT_AT(to, i, j) = (z>>j)&1;
    }
    MAT_AT(to, i, BITS) = z >= n;
  }

	size_t arch[] = {2*BITS, 2*BITS + 1, BITS + 1};
  Model m = model_alloc(arch, ML_ARRAY_LEN(arch));
  Model g = model_alloc(arch, ML_ARRAY_LEN(arch));
  model_rand(m, 0, 1);

  float rate = 1;

  for (size_t i = 0; i < 10*1000; ++i) {
    model_backprop(m, g, ti, to);
  	model_learn(m, g, rate);
  }

	size_t fails = 0;
  for (size_t x = 0; x < n; ++x) {
    for (size_t y = 0; y < n; ++y) {
      size_t z = x + y;
      for (size_t j = 0; j < BITS; ++j) {
        MAT_AT(MODEL_IN(m), 0, j) = (x>>j)&1;
        MAT_AT(MODEL_IN(m), 0, j + BITS) = (y>>j)&1;
      }
      model_forward(m);
      if (MAT_AT(MODEL_OUT(m), 0, BITS) > 0.5f) {
        if (z < n) {
					printf("%zu + %zu - (OVERFLOW<>%zu)\n", x, y, z);
					fails += 1;
        }
      } else {
        size_t a = 0;
        for (size_t j = 0; j < BITS; ++j) {
  				size_t bit = MAT_AT(MODEL_OUT(m), 0, j) > 0.5f;
  				a |= bit<<j;
        }
        if (z != a) {
					printf("%zu + %zu - (OVERFLOW|%zu<>%zu)\n", x, y, z, a);
					fails += 1;
        }
        printf("%zu\n", a);
      }
    }
  }

  if (fails == 0) {
    printf("OK\n");
  }

  return 0;
}
