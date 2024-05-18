#include <time.h>

#define ML_IMPLEMENTATION
#include "ml.h"

// https://en.wikipedia.org/wiki/Adder_(electronics)
float td_sum[] = {
  0, 0, 0, 0, 0, 0,
  0, 0, 0, 1, 0, 1,
  0, 1, 0, 1, 0, 1,
  0, 1, 1, 0, 1, 1,
};

// https://en.wikipedia.org/wiki/XOR_gate
float td_xor[] = {
  0, 0, 0,
  0, 1, 1,
  1, 0, 1,
  1, 1, 0,
};

// https://en.wikipedia.org/wiki/OR_gate
float td_or[] = {
  0, 0, 0,
  0, 1, 1,
  1, 0, 1,
  1, 1, 1,
};

int main(void)
{
  srand(time(NULL));

  float *td = td_xor;

  size_t stride = 3;
  size_t n = 4;

  Mat ti = {
    .rows = n,
    .cols = 2,
    .stride = stride,
    .es = td,
  };

  Mat to = {
    .rows = n,
    .cols = 1,
    .stride = stride,
    .es = td + 2,
  };

  size_t desc[] = {2, 2, 1};
  Model m = model_alloc(desc, ML_ARRAY_LEN(desc));
  Model g = model_alloc(desc, ML_ARRAY_LEN(desc));
  model_rand(m, 0, 1);

  float rate = 1;

  for (size_t i = 0; i < 100000; ++i) {
#if 0
    float eps = 1e-1;
    model_fdiff(m, g, eps, ti, to);
#else
    model_backprop(m, g, ti, to);
#endif
    model_learn(m, g, rate);
  }

  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      MAT_AT(MODEL_IN(m), 0, 0) = i;
      MAT_AT(MODEL_IN(m), 0, 1) = j;
      model_forward(m);
      printf("%zu ^ %zu = %f\n", i, j, MAT_AT(MODEL_OUT(m), 0, 0));
    }
  }

  return 0;
}
