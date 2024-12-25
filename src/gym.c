// Gym is a GUI app that trains your NN on the data you give it
// it returns a binary file that can be loaded up with nn.h and used in your application

#include <stdio.h>
#include "raylib.h"

#define IMG_FACTOR 80
#define IMG_WIDTH (16*IMG_FACTOR)
#define IMG_HEIGHT (9*IMG_FACTOR)

int main (int argc, char **argv)
{
  unsigned int buffer_len = 0;
  unsigned char *buffer = LoadFileData("../data/adder.arch", &buffer_len);
  fwrite(buffer, buffer_len, 1, stdout);

  //InitWindow(IMG_WIDTH, IMG_HEIGHT, "gym");
  //SetTargetFPS(60);
  return 0;
}
