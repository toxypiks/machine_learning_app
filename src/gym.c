// Gym is a GUI app that trains your NN on the data you give it
// it returns a binary file that can be loaded up with nn.h and used in your application

#include <stdio.h>
#include "raylib.h"
#define SV_IMPLEMENTATION
#include "sv.h"

#define IMG_FACTOR 80
#define IMG_WIDTH (16*IMG_FACTOR)
#define IMG_HEIGHT (9*IMG_FACTOR)

int main (int argc, char **argv)
{
  unsigned int buffer_len = 0;
  unsigned char *buffer = LoadFileData("../data/adder.arch", &buffer_len);

  String_View content = sv_from_parts((const char*)buffer, buffer_len);

  content = sv_trim_left(content);
  while(content.count > 0 && isdigit(content.data[0])) {
    int x = sv_chop_u64(&content);
    printf("%d\n", x);
    content = sv_trim_left(content);
  }
  //InitWindow(IMG_WIDTH, IMG_HEIGHT, "gym");
  //SetTargetFPS(60);
  return 0;
}
