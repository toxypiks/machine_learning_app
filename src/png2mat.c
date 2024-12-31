#include <stdio.h>
#include <assert.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define NN_IMPLEMENTATION
#include "nn.h"

char *args_shift(int *argc, char ***argv)
{
  assert(*argc > 0);
  char *result = **argv;
  (*argc) -= 1;
  (*argv) += 1;
  return result;
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

  size_t arch[] ={2, 28, 1};
  NN nn = nn_alloc(arch, ARRAY_LEN(arch));
  NN g = nn_alloc(arch, ARRAY_LEN(arch));
  nn_rand(nn, -1, 1);

  float rate = 1.0f;
  size_t max_epoch= 20000;

  for (size_t i = 0; i < max_epoch; ++i) {
    nn_backprop(nn, g, ti, to);
    nn_learn(nn, g, rate);
    if (i % 100 == 0) {
      printf("%zu: cost: %f\n", i, nn_cost(nn, ti, to));
    }
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
