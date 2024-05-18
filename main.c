#include <stdint.h>
#include <time.h>

#define Model Please_Add_Prefixes_Raylib
#include <raylib.h>
#undef Model

#define ML_IMPLEMENTATION
#include "ml.h"

#define BITS 2

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600

void model_render(Model m)
{
  Color background_color = { 0x1e, 0x1e, 0x2e, 0xff };
  Color low_color = { 0xff, 0x00, 0xff, 0xff };
  Color high_color = { 0x00, 0xff, 0x00, 0xff };

  ClearBackground(background_color);

  int neuron_radius = 25;
  int layer_border_vpad = 50;
  int layer_border_hpad = 50;
  int model_width = WINDOW_WIDTH - 2*layer_border_hpad;
  int model_height = WINDOW_HEIGHT - 2*layer_border_vpad;
  int model_x = WINDOW_WIDTH/2 - model_width/2;
  int model_y = WINDOW_HEIGHT/2 - model_height/2;
  size_t arch_len = m.len + 1;
  int layer_hpad = model_width/arch_len;
  for (size_t l = 0; l < arch_len; ++l) {
    int layer_vpad1 = model_height/m.as[l].cols;
    for (size_t i = 0; i < m.as[l].cols; ++i) {
      int cx1 = model_x + l*layer_hpad + layer_hpad/2;
      int cy1 = model_y + i*layer_vpad1 + layer_vpad1/2;
      if (l + 1 < arch_len) {
        int layer_vpad2 = model_height/m.as[l + 1].cols;
        for (size_t j = 0; j < m.as[l + 1].cols; ++j) {
          int cx2 = model_x + (l + 1)*layer_hpad + layer_hpad/2;
          int cy2 = model_y + j*layer_vpad2 + layer_vpad2/2;
          high_color.a = floorf(255.0f*ml_sigmoidf(MAT_AT(m.ws[l], j, i)));
          DrawLine(cx1, cy1, cx2, cy2, ColorAlphaBlend(low_color, high_color, WHITE));
        }
      }
      if (l > 0) {
        high_color.a = floorf(255.0f*ml_sigmoidf(MAT_AT(m.bs[l - 1], 0, i)));
        DrawCircle(cx1, cy1, neuron_radius, ColorAlphaBlend(low_color, high_color, WHITE));
      } else {
        DrawCircle(cx1, cy1, neuron_radius, GRAY);
      }
    }
  }
}

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

  InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Adder");
  SetTargetFPS(60);

  size_t i = 0;
  while (!WindowShouldClose()) {
    if (i < 5000) {
      model_backprop(m, g, ti, to);
      model_learn(m, g, rate);
      i += 1;
      printf("c = %f\n", model_cost(m, ti, to));
    }

    BeginDrawing();
    model_render(m);
    EndDrawing();
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
