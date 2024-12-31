#include <stdio.h>
#include <assert.h>
#include "stb_image.h"
#define NN_IMPLEMENTATION
#include "nn.h"
#include <raylib.h>
#include <float.h>

#define IMG_FACTOR 80
#define IMG_WIDTH (16*IMG_FACTOR)
#define IMG_HEIGHT (9*IMG_FACTOR)

typedef struct {
  size_t *items;
  size_t count;
  size_t capacity;
} Arch;

typedef struct {
  float *items;
  size_t count;
  size_t capacity;
} Cost_Plot;

#define DA_INIT_CAP 256

//macro for dynamic arrays
#define da_append(da, item)                                                          \
    do {                                                                             \
        if ((da)->count >= (da)->capacity) {                                         \
            (da)->capacity = (da)->capacity == 0 ? DA_INIT_CAP : (da)->capacity*2;   \
            (da)->items = realloc((da)->items, (da)->capacity*sizeof(*(da)->items)); \
            assert((da)->items != NULL && "Buy more RAM lol");                       \
        }                                                                            \
                                                                                     \
        (da)->items[(da)->count++] = (item);                                         \
    } while (0)

char *args_shift(int *argc, char ***argv)
{
  assert(*argc > 0);
  char *result = **argv;
  (*argc) -= 1;
  (*argv) += 1;
  return result;
}

void nn_render_raylib(NN nn, int rx, int ry, int rw, int rh)
{
  Color low_color = {0xFF, 0x00, 0xFF, 0xFF};
  Color high_color = {0x00, 0xFF, 0x00, 0xFF};

  float neuron_radius = rh*0.04f;
  int layer_border_vpad = 50;
  int layer_border_hpad = 50;
  int nn_width = rw - 2*layer_border_hpad;
  int nn_height = rh - 2*layer_border_vpad;
  int nn_x = rx + rw/2 - nn_width/2;
  int nn_y = ry + rh/2 - nn_height/2;
  size_t arch_count = nn.count + 1;
  int layer_hpad = nn_width / arch_count;
  for (size_t l = 0; l < arch_count; ++l) {
    int layer_vpad1 = nn_height / nn.as[l].cols;
    for (size_t i = 0; i < nn.as[l].cols; ++i) {
      int cx1 = nn_x + l*layer_hpad + layer_hpad/2;
      int cy1 = nn_y + i*layer_vpad1 + layer_vpad1/2;
      if (l+1 < arch_count) {
        int layer_vpad2 = nn_height / nn.as[l+1].cols;
        for (size_t j = 0; j < nn.as[l+1].cols; ++j) {
          int cx2 = nn_x + (l+1)*layer_hpad + layer_hpad/2;
          int cy2 = nn_y + j*layer_vpad2 + layer_vpad2/2;
          float value = sigmoidf(MAT_AT(nn.ws[l], j, i));
          high_color.a = floorf(255.f*value);
          float thick = rh*0.004f;
          Vector2 start = {cx1, cy1};
          Vector2 end = {cx2, cy2};
          DrawLineEx(start, end, thick, ColorAlphaBlend(low_color, high_color, WHITE));
        }
      }
      if (l > 0) {
        high_color.a = floorf(255.f*sigmoidf(MAT_AT(nn.bs[l-1], 0, i)));
        DrawCircle(cx1, cy1, neuron_radius, ColorAlphaBlend(low_color, high_color, WHITE));
      } else {
        DrawCircle(cx1, cy1, neuron_radius, GRAY);
      }
    }
  }
}

void cost_plot_minmax(Cost_Plot plot, float *min, float *max)
{
  *min = FLT_MAX;
  *max = FLT_MIN;
  for (size_t i = 0; i < plot.count; ++i) {
    if(*max < plot.items[i]) {
      *max = plot.items[i];
    }
    if(*min > plot.items[i]) {
      *min = plot.items[i];
    }
  }
}

void plot_cost(Cost_Plot plot, int rx, int ry, int rw, int rh)
{
  float min, max;
  cost_plot_minmax(plot, &min, &max);
  if (min > 0) min = 0;
  size_t n = plot.count;
  if (n < 1000) n = 1000;
  for (size_t i = 0; i+1 < plot.count; ++i) {
    float x1 = rx + (float)rw/n*i;
    float y1 = ry + (1- (plot.items[i] - min) / (max - min))*rh;
    float x2 = rx + (float)rw/n*(i+1);
    float y2 = ry + (1- (plot.items[i+1] - min) / (max - min))*rh;

    DrawLineEx((Vector2){x1, y1}, (Vector2){x2, y2}, rh*0.005f, RED);
  }
}

int main (int argc, char **argv)
{
  const char *program = args_shift(&argc, &argv);

  if(argc <= 0) {
    fprintf(stderr, "Usage: %s <input>\n", program);
    fprintf(stderr, "ERROR: no input file is provided\n");
    return 1;
  }

  const char *img_file_path = args_shift(&argc, &argv);

  int img_width, img_height, img_comp; //img_comp = amounts of bytes per pixel in the image, greyscale = 1
  uint8_t *img_pixels = (uint8_t *)stbi_load(img_file_path, &img_width, &img_height, &img_comp, 0);

  if(img_pixels == NULL) {
    fprintf(stderr, "ERROR: could not read image %s\n", img_file_path);
  }
  if (img_comp != 1) {
    fprintf(stderr, "ERROR: %s is %d bits image. Only 8 bit grayscale images are supported\n", img_file_path);
    return 1;
  }

  printf("%s size %dx%d %d bits\n", img_file_path, img_width, img_height, img_comp*8);

  //always: rows = amount of data, columns = input + output
  //in this case input: 3 since values for x, y, brightness
  Mat t = mat_alloc(img_width*img_height, 3);

  //iterate through each pixel from img
  for (size_t y = 0; y < img_height; ++y) {
    for (size_t x = 0; x < img_width; ++x) {
      size_t i = y*img_width + x;

      //normalize x-value by dividing by img_width but x can never reach img_width because
      //of loop-condition x < img_width, thats why x/(img_width -1)
      MAT_AT(t, i, 0) = (float)x/(img_width - 1);
      //same for y
      MAT_AT(t, i, 1) = (float)y/(img_height -1);
      //normalize brightness by dividing each pixel value by 255
      MAT_AT(t, i, 2) = img_pixels[i]/255.f;
    }
  }

  Mat ti = {
    .rows = t.rows,
    //we defined two inputs
    .cols = 2,
    .stride = t.stride,
    .es = &MAT_AT(t, 0, 0),
  };

  Mat to = {
    .rows = t.rows,
    .cols = 1,
    .stride = t.stride,
    .es = &MAT_AT(t, 0, ti.cols),
  };

  //MAT_PRINT(ti);
  //MAT_PRINT(to);

  size_t arch[] ={2, 28/4, 1};
  NN nn = nn_alloc(arch, ARRAY_LEN(arch));
  NN g = nn_alloc(arch, ARRAY_LEN(arch));
  nn_rand(nn, -1, 1);

  SetConfigFlags(FLAG_WINDOW_RESIZABLE);
  InitWindow(IMG_WIDTH, IMG_HEIGHT, "gym");
  SetTargetFPS(60);

  Cost_Plot plot = {0};

  size_t epoch = 0;
  size_t max_epoch = 100*1000;
  size_t epochs_per_frame = 103;
  float rate = 1.0f;
  bool paused = true;

  while(!WindowShouldClose()) {
    if (IsKeyPressed(KEY_SPACE)) {
      paused = !paused;
    }
    if (IsKeyPressed(KEY_R)) {
      epoch = 0;
      nn_rand(nn, -1, 1);
      plot.count = 0;
    }
    for (size_t i = 0; i < 10 && epochs_per_frame && !paused && epoch < max_epoch; ++i) {
      if (epoch < max_epoch) {
        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, rate);
        epoch += 1;
        da_append(&plot, nn_cost(nn, ti, to));
      }
    }
    BeginDrawing();
    Color background_color = {0x18, 0x18, 0x18, 0xFF};
    ClearBackground(background_color);
    {
      int rw, rh, rx, ry;
      int w = GetRenderWidth();
      int h = GetRenderHeight();

      rw = w/2;
      rh = h*2/3;
      rx = 0;
      ry = h/2 - rh/2;
      plot_cost(plot, rx, ry, rw, rh);

      rw = w/2;
      rh = h*2/3;
      rx = w - rw;
      ry = h/2 - rh/2;
      nn_render_raylib(nn, rx, ry, rw, rh);

      char buffer[256];
      snprintf(buffer, sizeof(buffer),"Epoch: %zu/%zu, Rate: %f", epoch, max_epoch, rate);
      DrawText(buffer, 0, 0, h*0.04, WHITE);
    }
    EndDrawing();
  }

  //render original image
  for (size_t y = 0; y < (size_t)img_height; ++y) {
    for (size_t x = 0; x < (size_t)img_width; ++x) {
      uint8_t pixel = img_pixels[y*img_width + x];
      if (pixel) printf("%3u ", pixel); else printf("    ");
    }
    printf("\n");
  }

  //render neural network output
  for (size_t y = 0; y < (size_t)img_height; ++y) {
    for (size_t x = 0; x < (size_t)img_width; ++x) {
      size_t i = y*img_width + x;
      MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(img_width -1);
      MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(img_height -1);
      nn_forward(nn);
      uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.0f;
      if (pixel) printf("%3u ", pixel);
    }
    printf("\n");
    }

  return 0;

  const char *out_file_path = "../output_files/img.mat";
  FILE *out = fopen(out_file_path, "wb");
  if (out == NULL) {
    fprintf(stderr, "ERROR: could not open file %s\n", out_file_path);
    return 1;
  }
  mat_save(out, t);

  printf("Generated &s from %s\n", out_file_path, img_file_path);
  return 0;
}
