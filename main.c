#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct {
  float or_w1;
  float or_w2;
  float or_b;
  float nand_w1;
  float nand_w2;
  float nand_b;
  float and_w1;
  float and_w2;
  float and_b;
} Model;

// https://en.wikipedia.org/wiki/Sigmoid_function
float sigmoidf(float x)
{
  return 1/(1 + expf(-x));
}

float forward(Model m, float x1, float x2)
{
  float a = sigmoidf(m.or_w1*x1 + m.or_w2*x2 + m.or_b);
  float b = sigmoidf(m.nand_w1*x1 + m.nand_w2*x2 + m.nand_b);
  return sigmoidf(a*m.and_w1 + b*m.and_w2 + m.and_b);
}

typedef float sample[3];

// https://en.wikipedia.org/wiki/OR_gate
sample or_data[] = {
  {0, 0, 0},
  {1, 0, 1},
  {0, 1, 1},
  {1, 1, 1},
};

// https://en.wikipedia.org/wiki/NOR_gate
sample nor_data[] = {
  {0, 0, 1},
  {1, 0, 0},
  {0, 1, 0},
  {1, 1, 0},
};

// https://en.wikipedia.org/wiki/XOR_gate
sample xor_data[] = {
  {0, 0, 0},
  {1, 0, 1},
  {0, 1, 1},
  {1, 1, 0},
};

// https://en.wikipedia.org/wiki/AND_gate
sample and_data[] = {
  {0, 0, 0},
  {1, 0, 0},
  {0, 1, 0},
  {1, 1, 1},
};

// https://en.wikipedia.org/wiki/NAND_gate
sample nand_data[] = {
  {0, 0, 1},
  {1, 0, 1},
  {0, 1, 1},
  {1, 1, 0},
};

sample *data = xor_data;
size_t data_count = 4;

// https://en.wikipedia.org/wiki/Loss_function
float cost(Model m)
{
  float result = 0.0f;
  for (size_t i = 0; i < data_count; ++i) {
    float x1 = data[i][0];
    float x2 = data[i][1];
    float y = forward(m, x1, x2);
    float d = y - data[i][2];
    result += d*d;
  }
  return result/data_count;
}

float rand_float(void)
{
  return (float)rand()/(float)RAND_MAX;
}

Model rand_model(void)
{
  return (Model) {
    .or_w1 = rand_float(),
    .or_w2 = rand_float(),
    .or_b = rand_float(),
    .nand_w1 = rand_float(),
    .nand_w2 = rand_float(),
    .nand_b = rand_float(),
    .and_w1 = rand_float(),
    .and_w2 = rand_float(),
    .and_b = rand_float(),
  };
}

void print_model(Model m)
{
  printf("or_w1 = %f\n", m.or_w1);
  printf("or_w2 = %f\n", m.or_w2);
  printf("or_b = %f\n", m.or_b);
  printf("nand_w1 = %f\n", m.nand_w1);
  printf("nand_w2 = %f\n", m.nand_w2);
  printf("nand_b = %f\n", m.nand_b);
  printf("and_w1 = %f\n", m.and_w1);
  printf("and_w2 = %f\n", m.and_w2);
  printf("and_b = %f\n", m.and_b);
}

Model finite_diff(Model m, float eps)
{
  Model g;
  float c = cost(m);
  float saved;

  saved = m.or_w1;
  m.or_w1 += eps;
  g.or_w1 = (cost(m) - c)/eps;
  m.or_w1 = saved;

  saved = m.or_w2;
  m.or_w2 += eps;
  g.or_w2 = (cost(m) - c)/eps;
  m.or_w2 = saved;

  saved = m.or_b;
  m.or_b += eps;
  g.or_b = (cost(m) - c)/eps;
  m.or_b = saved;

  saved = m.nand_w1;
  m.nand_w1 += eps;
  g.nand_w1 = (cost(m) - c)/eps;
  m.nand_w1 = saved;

  saved = m.nand_w2;
  m.nand_w2 += eps;
  g.nand_w2 = (cost(m) - c)/eps;
  m.nand_w2 = saved;

  saved = m.nand_b;
  m.nand_b += eps;
  g.nand_b = (cost(m) - c)/eps;
  m.nand_b = saved;

  saved = m.and_w1;
  m.and_w1 += eps;
  g.and_w1 = (cost(m) - c)/eps;
  m.and_w1 = saved;

  saved = m.and_w2;
  m.and_w2 += eps;
  g.and_w2 = (cost(m) - c)/eps;
  m.and_w2 = saved;

  saved = m.and_b;
  m.and_b += eps;
  g.and_b = (cost(m) - c)/eps;
  m.and_b = saved;

  return g;
}

Model train(Model m, Model g, float rate)
{
  m.or_w1 -= rate*g.or_w1;
  m.or_w2 -= rate*g.or_w2;
  m.or_b -= rate*g.or_b;
  m.nand_w1 -= rate*g.nand_w1;
  m.nand_w2 -= rate*g.nand_w2;
  m.nand_b -= rate*g.nand_b;
  m.and_w1 -= rate*g.and_w1;
  m.and_w2 -= rate*g.and_w2;
  m.and_b -= rate*g.and_b;
  return m;
}

int main(void)
{
  srand(time(NULL));
  
  Model m = rand_model();

  float eps = 1e-1;
  float rate = 1e-1;

  for (size_t i = 0; i < 100000; ++i) {
    Model g = finite_diff(m, eps);
    m = train(m, g, rate);
  }

  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      printf("%zu | %zu = %f\n", i, j, forward(m, i, j));
    }
  }

  return 0;
}
