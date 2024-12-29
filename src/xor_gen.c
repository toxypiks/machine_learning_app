#define NN_IMPLEMENTATION
#include "nn.h"

int main (void)
{
  Mat t = mat_alloc(4, 3);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      size_t row = i*2 + j;
      MAT_AT(t, row, 0) = i;
      MAT_AT(t, row, 1) = j;
      MAT_AT(t, row, 2) = i^j;
    }
  }
  MAT_PRINT(t);
  return 0;
}
