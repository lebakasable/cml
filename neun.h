#ifndef NEUN_H_
#define NEUN_H_

#include <stdio.h>
#include <stddef.h>
#include <math.h>

#ifndef NEUN_MALLOC
#include <stdlib.h>
#define NEUN_MALLOC malloc
#endif // NEUN_MALLOC

#ifndef NEUN_ASSERT
#include <assert.h>
#define NEUN_ASSERT assert
#endif // NEUN_ASSERT

typedef struct {
  size_t rows;
  size_t cols;
  size_t stride;
  float *es;
} Mat;

float neun_randf(void);
float neun_sigmoidf(float x);

Mat mat_alloc(size_t rows, size_t cols);
void mat_fill(Mat m, float x);
void mat_rand(Mat m, float low, float high);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
void mat_sum(Mat dst, Mat a);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sig(Mat m);
void mat_print(Mat m, const char *name);

#endif // NEUN_H_

#ifdef NEUN_IMPLEMENTATION

float neun_randf(void)
{
  return (float)rand()/(float)RAND_MAX;
}

// https://en.wikipedia.org/wiki/Sigmoid_function
float neun_sigmoidf(float x)
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
  m.es = NEUN_MALLOC(sizeof(*m.es)*rows*cols);
  NEUN_ASSERT(m.es != NULL);
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
      MAT_AT(m, i, j) = neun_randf()*(high - low) + low;
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
  NEUN_ASSERT(dst.rows == src.rows);
  NEUN_ASSERT(dst.cols == src.cols);

  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MAT_AT(dst, i, j) = MAT_AT(src, i, j);
    }
  }
}

void mat_sum(Mat dst, Mat a)
{
  NEUN_ASSERT(dst.rows == a.rows);
  NEUN_ASSERT(dst.cols == a.cols);

  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MAT_AT(dst, i, j) += MAT_AT(a, i, j);
    }
  }
}

void mat_dot(Mat dst, Mat a, Mat b)
{
  NEUN_ASSERT(a.cols == b.rows);
  size_t n = a.cols;
  NEUN_ASSERT(dst.rows == a.rows);
  NEUN_ASSERT(dst.cols == b.cols);

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
      MAT_AT(m, i, j) = neun_sigmoidf(MAT_AT(m, i, j));
    }
  }
}

void mat_print(Mat m, const char *name)
{
  printf("%s = [\n", name);
  for (size_t i = 0; i < m.rows; ++i) {
    for (size_t j = 0; j < m.cols; ++j) {
      printf("    %f ", MAT_AT(m, i, j));
    }
    printf("\n");
  }
  printf("]\n");
}

#define MAT_PRINT(m) mat_print(m, #m)

#endif // NEUN_IMPLEMENTATION
