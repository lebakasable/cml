#ifndef ML_H_
#define ML_H_

#include <stdio.h>
#include <stddef.h>
#include <math.h>

#ifndef ML_MALLOC
#include <stdlib.h>
#define ML_MALLOC malloc
#endif // ML_MALLOC

#ifndef ML_ASSERT
#include <assert.h>
#define ML_ASSERT assert
#endif // ML_ASSERT

typedef struct {
  size_t rows;
  size_t cols;
  size_t stride;
  float *es;
} Mat;

float ml_randf(void);
float ml_sigmoidf(float x);

Mat mat_alloc(size_t rows, size_t cols);
void mat_fill(Mat m, float x);
void mat_rand(Mat m, float low, float high);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
void mat_sum(Mat dst, Mat a);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sig(Mat m);
void mat_print(Mat m, const char *name, size_t pad);

// https://en.wikipedia.org/wiki/Machine_learning
typedef struct {
  size_t len;
  Mat *ws;
  Mat *bs;
  Mat *as;
} Model;

Model model_alloc(size_t *desc, size_t desc_len);
void model_zero(Model m);
void model_rand(Model m, float low, float high);
void model_print(Model m, const char *name);
void model_forward(Model m);
float model_cost(Model m, Mat ti, Mat to);
void model_fdiff(Model m, Model g, float eps, Mat ti, Mat to);
void model_backprop(Model m, Model g, Mat ti, Mat to);
void model_learn(Model m, Model g, float rate);

#endif // ML_H_

#ifdef ML_IMPLEMENTATION

#define ML_ARRAY_LEN(xs) (sizeof(xs)/sizeof((xs)[0]))

float ml_randf(void)
{
  return (float)rand()/(float)RAND_MAX;
}

// https://en.wikipedia.org/wiki/Sigmoid_function
float ml_sigmoidf(float x)
{
  return 1.0f/(1.0f + expf(-x));
}

#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]

Mat mat_alloc(size_t rows, size_t cols)
{
  Mat m;
  m.rows = rows;
  m.cols = cols;
  m.stride = cols;
  m.es = ML_MALLOC(sizeof(*m.es)*rows*cols);
  ML_ASSERT(m.es != NULL);
  return m;
}

void mat_fill(Mat m, float x)
{
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      MAT_AT(m, i, j) = x;
    }
  }
}

void mat_rand(Mat m, float low, float high)
{
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      MAT_AT(m, i, j) = ml_randf()*(high - low) + low;
    }
  }
}

Mat mat_row(Mat m, size_t row)
{
  return (Mat) {
    .rows = 1,
    .cols = m.cols,
    .stride = m.stride,
    .es = &MAT_AT(m, row, 0),
  };
}

void mat_copy(Mat dst, Mat src)
{
  ML_ASSERT(dst.rows == src.rows);
  ML_ASSERT(dst.cols == src.cols);

  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MAT_AT(dst, i, j) = MAT_AT(src, i, j);
    }
  }
}

void mat_sum(Mat dst, Mat a)
{
  ML_ASSERT(dst.rows == a.rows);
  ML_ASSERT(dst.cols == a.cols);

  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MAT_AT(dst, i, j) += MAT_AT(a, i, j);
    }
  }
}

void mat_dot(Mat dst, Mat a, Mat b)
{
  ML_ASSERT(a.cols == b.rows);
  size_t n = a.cols;
  ML_ASSERT(dst.rows == a.rows);
  ML_ASSERT(dst.cols == b.cols);

  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MAT_AT(dst, i, j) = 0;
      for (size_t k = 0; k < n; ++k) {
        MAT_AT(dst, i, j) += MAT_AT(a, i, k)*MAT_AT(b, k, j);
      }
    }
  }
}

void mat_sig(Mat m)
{
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      MAT_AT(m, i, j) = ml_sigmoidf(MAT_AT(m, i, j));
    }
  }
}

void mat_print(Mat m, const char *name, size_t pad)
{
  printf("%*s%s = [\n", (int)pad, "", name);
  for (size_t i = 0; i < m.rows; ++i) {
    printf("%*s  ", (int)pad, "");
    for (size_t j = 0; j < m.cols; ++j) {
      printf("%f ", MAT_AT(m, i, j));
    }
    printf("\n");
  }
  printf("%*s]\n", (int)pad, "");
}

#define MAT_PRINT(m) mat_print(m, #m, 0)

#define MODEL_IN(m) (m).as[0]
#define MODEL_OUT(m) (m).as[(m).len]

Model model_alloc(size_t *desc, size_t desc_len)
{
  ML_ASSERT(desc_len > 0);

  Model m;
  m.len = desc_len - 1;

  m.ws = ML_MALLOC(sizeof(*m.ws)*m.len);
  ML_ASSERT(m.ws != NULL);
  m.bs = ML_MALLOC(sizeof(*m.bs)*m.len);
  ML_ASSERT(m.bs != NULL);
  m.as = ML_MALLOC(sizeof(*m.as)*(m.len + 1));
  ML_ASSERT(m.as != NULL);

  m.as[0] = mat_alloc(1, desc[0]);
  for (size_t i = 1; i < desc_len; ++i) {
    m.ws[i - 1] = mat_alloc(m.as[i - 1].cols, desc[i]);
    m.bs[i - 1] = mat_alloc(1, desc[i]);
    m.as[i]     = mat_alloc(1, desc[i]);
  }

  return m;
}

void model_zero(Model m)
{
  for (size_t i = 0; i < m.len; ++i) {
    mat_fill(m.ws[i], 0);
    mat_fill(m.bs[i], 0);
    mat_fill(m.as[i], 0);
  }
  mat_fill(m.as[m.len], 0);
}

void model_rand(Model m, float low, float high)
{
  for (size_t i = 0; i < m.len; ++i) {
    mat_rand(m.ws[i], low, high);
    mat_rand(m.bs[i], low, high);
  }
}

void model_print(Model m, const char *name)
{
  char buf[256];
  printf("%s = [\n", name);
  for (size_t i = 0; i < m.len; ++i) {
    snprintf(buf, sizeof(buf), "ws%zu", i);
    mat_print(m.ws[i], buf, 2);
    snprintf(buf, sizeof(buf), "bs%zu", i);
    mat_print(m.bs[i], buf, 2);
  }
  printf("]\n");
}

#define MODEL_PRINT(m) model_print(m, #m)

void model_forward(Model m)
{
  for (size_t i = 0; i < m.len; ++i) {
    mat_dot(m.as[i + 1], m.as[i], m.ws[i]);
    mat_sum(m.as[i + 1], m.bs[i]);
    mat_sig(m.as[i + 1]);
  }
}

float model_cost(Model m, Mat ti, Mat to)
{
  ML_ASSERT(ti.rows == to.rows);
  ML_ASSERT(to.cols == MODEL_OUT(m).cols);
  size_t n = ti.rows;

  float c = 0;
  for (size_t i = 0; i < n; ++i) {
    Mat x = mat_row(ti, i);
    Mat y = mat_row(to, i);

    mat_copy(MODEL_IN(m), x);
    model_forward(m);
    size_t q = to.cols;
    for (size_t j = 0; j < q; ++j) {
      float d = MAT_AT(MODEL_OUT(m), 0, j) - MAT_AT(y, 0, j);
      c += d*d;
    }
  }

  return c/n;
}

// https://en.wikipedia.org/wiki/Finite_difference
void model_fdiff(Model m, Model g, float eps, Mat ti, Mat to)
{
  float saved;
  float c = model_cost(m, ti, to);

  for (size_t i = 0; i < m.len; ++i) {
    for (size_t j = 0; j < m.ws[i].rows; ++j) {
      for (size_t k = 0; k < m.ws[i].cols; ++k) {
        saved = MAT_AT(m.ws[i], j, k);
        MAT_AT(m.ws[i], j, k) += eps;
        MAT_AT(g.ws[i], j, k) = (model_cost(m, ti, to) - c)/eps;
        MAT_AT(m.ws[i], j, k) = saved;
      }
    }
    for (size_t j = 0; j < m.bs[i].rows; ++j) {
      for (size_t k = 0; k < m.bs[i].cols; ++k) {
        saved = MAT_AT(m.bs[i], j, k);
        MAT_AT(m.bs[i], j, k) += eps;
        MAT_AT(g.bs[i], j, k) = (model_cost(m, ti, to) - c)/eps;
        MAT_AT(m.bs[i], j, k) = saved;
      }
    }
  }
}

// https://en.wikipedia.org/wiki/Backpropagation
void model_backprop(Model m, Model g, Mat ti, Mat to)
{
  ML_ASSERT(ti.rows == to.rows);
  size_t n = ti.rows;
  ML_ASSERT(MODEL_OUT(m).cols == to.cols);

  model_zero(g);

  for (size_t i = 0; i < n; ++i) {
    mat_copy(MODEL_IN(m), mat_row(ti, i));
    model_forward(m);

		for (size_t j = 0; j <= m.len; ++j) {
  		mat_fill(g.as[j], 0);
		}

    for (size_t j = 0; j < to.cols; ++j) {
      MAT_AT(MODEL_OUT(g), 0, j) = MAT_AT(MODEL_OUT(m), 0, j) - MAT_AT(to, i, j);
    }

    for (size_t l = m.len; l > 0; --l) {
      for (size_t j = 0; j < m.as[l].cols; ++j) {
        float a = MAT_AT(m.as[l], 0, j);
        float da = MAT_AT(g.as[l], 0, j);
        MAT_AT(g.bs[l - 1], 0, j) += 2*da*a*(1 - a);
        for (size_t k = 0; k < m.as[l - 1].cols; ++k) {
          float pa = MAT_AT(m.as[l - 1], 0, k);
          float w = MAT_AT(m.ws[l - 1], k, j);
          MAT_AT(g.ws[l - 1], k, j) += 2*da*a*(1 - a)*pa;
          MAT_AT(g.as[l - 1], 0, k) += 2*da*a*(1 - a)*w;
        }
      }
    }
  }

  for (size_t i = 0; i < g.len; ++i) {
    for (size_t j = 0; j < g.ws[i].rows; ++j) {
      for (size_t k = 0; k < g.ws[i].cols; ++k) {
        MAT_AT(g.ws[i], j, k) /= n;
      }
    }
    for (size_t j = 0; j < g.bs[i].rows; ++j) {
      for (size_t k = 0; k < g.bs[i].cols; ++k) {
        MAT_AT(g.bs[i], j, k) /= n;
      }
    }
  }
}

void model_learn(Model m, Model g, float rate)
{
  for (size_t i = 0; i < m.len; ++i) {
    for (size_t j = 0; j < m.ws[i].rows; ++j) {
      for (size_t k = 0; k < m.ws[i].cols; ++k) {
        MAT_AT(m.ws[i], j, k) -= rate*MAT_AT(g.ws[i], j, k);
      }
    }
    for (size_t j = 0; j < m.bs[i].rows; ++j) {
      for (size_t k = 0; k < m.bs[i].cols; ++k) {
        MAT_AT(m.bs[i], j, k) -= rate*MAT_AT(g.bs[i], j, k);
      }
    }
  }
}

#endif // ML_IMPLEMENTATION
