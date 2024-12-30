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

  return 0;
}
