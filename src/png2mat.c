#include <assert.h>
#include <stdio.h>
#include <float.h>

#include <raylib.h>

#include "stb_image.h"
#include "stb_image_write.h"

#define NN_IMPLEMENTATION
#define NN_ENABLE_GYM
#include "nn.h"

char *args_shift(int *argc, char ***argv)
{
    assert(*argc > 0);
    char *result = **argv;
    (*argc) -= 1;
    (*argv) += 1;
    return result;
}

int main(int argc, char **argv)
{
    const char *program = args_shift(&argc, &argv);

    if (argc <= 0) {
        fprintf(stderr, "Usage: %s <image1> <image2>\n", program);
        fprintf(stderr, "ERROR: no image1 file is provided\n");
        return 1;
    }

    const char *img1_file_path = args_shift(&argc, &argv);

    if (argc <= 0) {
        fprintf(stderr, "Usage: %s <image1> <image2>\n", program);
        fprintf(stderr, "ERROR: no image2 file is provided\n");
        return 1;
    }

    const char *img2_file_path = args_shift(&argc, &argv);

    int img1_width, img1_height, img1_comp;
    uint8_t *img1_pixels = (uint8_t *)stbi_load(img1_file_path, &img1_width, &img1_height, &img1_comp, 0);
    if (img1_pixels == NULL) {
        fprintf(stderr, "ERROR: could not read image1 %s\n", img1_file_path);
        return 1;
    }
    if (img1_comp != 1) {
        fprintf(stderr, "ERROR: %s is %d bits image. Only 8 bit grayscale images are supported\n", img1_file_path, img1_comp*8);
        return 1;
    }

    int img2_width, img2_height, img2_comp;
    uint8_t *img2_pixels = (uint8_t *)stbi_load(img2_file_path, &img2_width, &img2_height, &img2_comp, 0);
    if (img2_pixels == NULL) {
        fprintf(stderr, "ERROR: could not read image2 %s\n", img2_file_path);
        return 1;
    }
    if (img2_comp != 1) {
        fprintf(stderr, "ERROR: %s is %d bits image. Only 8 bit grayscale images are supported\n", img2_file_path, img2_comp*8);
        return 1;
    }

    printf("%s size %dx%d %d bits\n", img1_file_path, img1_width, img1_height, img1_comp*8);
    printf("%s size %dx%d %d bits\n", img2_file_path, img2_width, img2_height, img2_comp*8);

    //always: rows = amount of data, columns = input + output
    //in this case input: 3 since values for x, y, brightness
    Mat t = mat_alloc(img1_width*img1_height + img2_width*img2_height, 4);

    //iterate through each pixel from img1
    for (int y = 0; y < img1_height; ++y) {
        for (int x = 0; x < img1_width; ++x) {
            size_t i = y*img1_width + x;
            //normalize x-value by dividing by img_width but x can never reach img_width because
            //of loop-condition x < img_width, thats why x/(img_width -1)
            MAT_AT(t, i, 0) = (float)x/(img1_width - 1);
            //same for y
            MAT_AT(t, i, 1) = (float)y/(img1_height - 1);
            //index for current image
            MAT_AT(t, i, 2) = 0.0f;
            //normalize brightness by dividing each pixel value by 255
            MAT_AT(t, i, 3) = img1_pixels[i]/255.f;
        }
    }

    //image 2
    for (int y = 0; y < img2_height; ++y) {
      for (int x = 0; x < img2_width; ++x) {
        //off-set index by img1
        size_t i = img1_width*img1_height + y*img2_width + x;
        MAT_AT(t, i, 0) = (float)x/(img2_width - 1);
        MAT_AT(t, i, 1) = (float)y/(img2_height - 1);
        MAT_AT(t, i, 2) = 1.0f;
        MAT_AT(t, i, 3) = img1_pixels[i]/255.f;
      }
    }

    size_t arch[] = {3, 7, 7, 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, -1, 1);

    size_t WINDOW_FACTOR = 80;
    size_t WINDOW_WIDTH = (16*WINDOW_FACTOR);
    size_t WINDOW_HEIGHT = (9*WINDOW_FACTOR);

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "gym");
    SetTargetFPS(60);

    Plot plot = {0};

    Image preview_image = GenImageColor(img1_width, img1_height, BLACK);
    Texture2D preview_texture = LoadTextureFromImage(preview_image);

    Image original_image = GenImageColor(img1_width, img1_height, BLACK);
    for (size_t y = 0; y < (size_t) img1_height; ++y) {
        for (size_t x = 0; x < (size_t) img1_width; ++x) {
            uint8_t pixel = img1_pixels[y*img1_width + x];
            ImageDrawPixel(&original_image, x, y, CLITERAL(Color) { pixel, pixel, pixel, 255 });
        }
    }
    Texture2D original_texture = LoadTextureFromImage(original_image);

    size_t epoch = 0;
    size_t max_epoch = 100*1000;
    size_t batches_per_frame = 200;
    size_t batch_size = 28;
    size_t batch_count = (t.rows + batch_size - 1)/batch_size;
    size_t batch_begin = 0;
    float average_cost = 0.0f;
    float rate = 1.0f;
    bool paused = true;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
        }
        if (IsKeyPressed(KEY_R)) {
            epoch = 0;
            nn_rand(nn, -1, 1);
            plot.count = 0;
        }

        for (size_t i = 0; i < batches_per_frame && !paused && epoch < max_epoch; ++i) {
            size_t size = batch_size;
            if (batch_begin + batch_size >= t.rows)  {
                size = t.rows - batch_begin;
            }

            Mat batch_ti = {
                .rows = size,
                .cols = 3,
                .stride = t.stride,
                .es = &MAT_AT(t, batch_begin, 0),
            };

            Mat batch_to = {
                .rows = size,
                .cols = 1,
                .stride = t.stride,
                .es = &MAT_AT(t, batch_begin, batch_ti.cols),
            };

            nn_backprop(nn, g, batch_ti, batch_to);
            nn_learn(nn, g, rate);
            average_cost += nn_cost(nn, batch_ti, batch_to);
            batch_begin += batch_size;

            if (batch_begin >= t.rows) {
                epoch += 1;
                da_append(&plot, average_cost/batch_count);
                average_cost = 0.0f;
                batch_begin = 0;
                mat_shuffle_rows(t);
            }
        }

        BeginDrawing();
        Color background_color = {0x18, 0x18, 0x18, 0xFF};
        ClearBackground(background_color);
        {
            int w = GetRenderWidth();
            int h = GetRenderHeight();

            int rw = w/3;
            int rh = h*2/3;
            int rx = 0;
            int ry = h/2 - rh/2;

            gym_plot(plot, rx, ry, rw, rh);
            rx += rw;
            gym_render_nn(nn, rx, ry, rw, rh);
            rx += rw;

            float scale = 10;

            for (size_t y = 0; y < (size_t) img1_height; ++y) {
                for (size_t x = 0; x < (size_t) img1_width; ++x) {
                    MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(img1_width - 1);
                    MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(img1_height - 1);
                    MAT_AT(NN_INPUT(nn), 0, 2) = 0.0f;
                    nn_forward(nn);
                    uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
                    ImageDrawPixel(&preview_image, x, y, CLITERAL(Color) { pixel, pixel, pixel, 255 });
                }
            }

            UpdateTexture(preview_texture, preview_image.data);
            DrawTextureEx(preview_texture, CLITERAL(Vector2) { rx, ry }, 0, scale, WHITE);
            DrawTextureEx(original_texture, CLITERAL(Vector2) { rx, ry + img1_height*scale }, 0, scale, WHITE);

            char buffer[256];
            snprintf(buffer, sizeof(buffer), "Epoch: %zu/%zu, Rate: %f, Cost: %f", epoch, max_epoch, rate, plot.count > 0 ? plot.items[plot.count - 1] : 0);
            DrawText(buffer, 0, 0, h*0.04, WHITE);
        }
        EndDrawing();
    }

    //render original image
    for (size_t y = 0; y < (size_t) img1_height; ++y) {
        for (size_t x = 0; x < (size_t) img1_width; ++x) {
            uint8_t pixel = img1_pixels[y*img1_width + x];
            if (pixel) printf("%3u ", pixel); else printf("    ");
        }
        printf("\n");
    }

    //render neural network output
    for (size_t y = 0; y < (size_t) img1_height; ++y) {
        for (size_t x = 0; x < (size_t) img1_width; ++x) {
            MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(img1_width - 1);
            MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(img1_height - 1);
            nn_forward(nn);
            uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
            if (pixel) printf("%3u ", pixel); else printf("    ");
        }
        printf("\n");
    }

    //save neural network output as img
    size_t out_width = 512;
    size_t out_height = 512;
    uint8_t *out_pixels = malloc(sizeof(*out_pixels)*out_width*out_height);
    assert(out_pixels != NULL);

    for (size_t y = 0; y < out_height; ++y) {
        for (size_t x = 0; x < out_width; ++x) {
            MAT_AT(NN_INPUT(nn), 0, 0) = (float)x/(out_width - 1);
            MAT_AT(NN_INPUT(nn), 0, 1) = (float)y/(out_height - 1);
            nn_forward(nn);
            uint8_t pixel = MAT_AT(NN_OUTPUT(nn), 0, 0)*255.f;
            out_pixels[y*out_width + x] = pixel;
        }
    }

    const char *out_file_path = "upscaled.png";
    if (!stbi_write_png(out_file_path, out_width, out_height, 1, out_pixels, out_width*sizeof(*out_pixels))) {
        fprintf(stderr, "ERROR: could not save image %s\n", out_file_path);
        return 1;
    }

    printf("Generated %s from %s\n", out_file_path, img1_file_path);

    return 0;
}
