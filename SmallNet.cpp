#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 2
#define HIDDEN_SIZE 4
#define OUTPUT_SIZE 1

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

int main() {
    double inputs[4][2] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    double outputs[4][1] = { {0}, {1}, {1}, {0} };

    double hidden_weights[INPUT_SIZE][HIDDEN_SIZE] = { {-0.16595599, 0.44064899, -0.99977125, -0.39533485},
                                                      {-0.70648822, -0.81532281, -0.62747958, -0.30887855} };
    double output_weights[HIDDEN_SIZE][OUTPUT_SIZE] = { {-0.20646505}, {0.07763347}, {-0.16161097}, {0.370439} };
    double hidden_biases[HIDDEN_SIZE] = { 0.5, 0.5, 0.5, 0.5 };
    double output_bias[OUTPUT_SIZE] = { 0.5 };

    // 前向传播
    double hidden[HIDDEN_SIZE]{};
    double output[OUTPUT_SIZE]{};
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        hidden[j] = 0.0;
        for (int k = 0; k < INPUT_SIZE; k++) {
            hidden[j] += inputs[0][k] * hidden_weights[k][j];
        }
        hidden[j] += hidden_biases[j];
        hidden[j] = sigmoid(hidden[j]);
    }
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        output[j] = 0.0;
        for (int k = 0; k < HIDDEN_SIZE; k++) {
            output[j] += hidden[k] * output_weights[k][j];
        }
        output[j] += output_bias[j];
        output[j] = sigmoid(output[j]);
    }

    // 输出结果
    printf("Output: %f\n", output[0]);

    return 0;
}
