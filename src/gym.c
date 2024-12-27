// Gym is a GUI app that trains your NN on the data you give it
// it returns a binary file that can be loaded up with nn.h and used in your application

#include <stdio.h>
#include "raylib.h"
#define SV_IMPLEMENTATION
#include "sv.h"
#include <assert.h>
#define NN_IMPLEMENTATION
#include "nn.h"

#define IMG_FACTOR 80
#define IMG_WIDTH (16*IMG_FACTOR)
#define IMG_HEIGHT (9*IMG_FACTOR)

//struct for macro for dynamic arrays

typedef struct {
  size_t *items;
  size_t count;
  size_t capacity;
} Arch;

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

int main (int argc, char **argv)
{
  const char *program = args_shift(&argc, &argv);

  if (argc <= 0) {
    fprintf(stderr, "Usage: %s <model.arch> <model.mat>\n", program);
    fprintf(stderr, "ERROR: no architecture file was provided\n");
    return 1;
  }

  const char *arch_file_path = args_shift(&argc, &argv);

  if(argc <= 0) {
    fprintf(stderr, "Usage: %s <model.arch> <model.mat>\n", program);
    fprintf(stderr, "ERROR: no data file was provided\n");
    return 1;
  }

  const char *data_file_path = args_shift(&argc, &argv);

  unsigned int buffer_len = 0;
  unsigned char *buffer = LoadFileData("../data/adder.arch", &buffer_len);
  if (buffer == NULL) {
    return 1;
  }

  String_View content = sv_from_parts((const char*)buffer, buffer_len);

  Arch arch = {0};

  content = sv_trim_left(content);
  while(content.count > 0 && isdigit(content.data[0])) {
  size_t x = sv_chop_u64(&content);
    printf("%d\n", x);
    da_append(&arch, x);
    content = sv_trim_left(content);
  }

  FILE *in = fopen(data_file_path, "rb");
  if (in == NULL) {
    fprintf(stderr, "ERROR: could not read file %s\n", data_file_path);
    return 1;
  }
  Mat t = mat_load(in);
  fclose(in);

  MAT_PRINT(t);

  /*InitWindow(IMG_WIDTH, IMG_HEIGHT, "gym");
  SetTargetFPS(60);

  size_t i = 0;
  while(!WindowShouldClose()) {
    if (i < 5000) {
      nn_backprop(nn, g, ti, to);
      nn_apply_finite_diff(nn, g, rate);
      i += 1;
      printf("c = %f\n", nn_cost(nn, ti, to));
    }
    BeginDrawing();
    ClearBackground(RAYWHITE);
    nn_render_raylib(nn);
    EndDrawing();
	}*/

  return 0;
}
