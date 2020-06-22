#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mol.h"
#include "datapt.h"
#include "batch.h"
#include "mpn.h"
#include "cuda.h"
#include <cuda_runtime.h>

void shuffle(int n, int *l) {
    for (int i = 0; i < n - 1; i++) {
        int j = (rand() % (n - i)) + i;
        int tmp = l[j];
        l[j] = l[i];
        l[i] = tmp;
    }
}

void test(struct mpn *, struct datapt *, int, int);

static int BATCHSIZE = 128;
static int EPOCHS = 10;

struct mpn *train(struct datapt *data, int N) {
    // prepare batches
    int l[N];
    for (int i = 0; i < N; i++) {
        l[i] = i;
    }

    // create network
    struct mpn *mpn = mpn_create();

    mpn_init(mpn);

    // training loop
    int iters = 0;
    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        shuffle(N, l);
        struct batch *batches = batch_create(N, l, BATCHSIZE, data);

        float tot_loss = 0;

        for (int i = 0; i < N / BATCHSIZE; i++) {
            mpn_forward(mpn, &batches[i]);

            float loss = mpn_loss(mpn, &batches[i]);
            tot_loss += loss;

            mpn_backward(mpn, &batches[i]);
            mpn_adam(mpn, ++iters, 0.001, 0.9, 0.999);
        }
        printf("epoch %d tot %f\n", epoch, tot_loss);

        for (int i = 0; i < N / BATCHSIZE; i++) {
            free(batches[i].m_atoms);
            free_mol(batches[i].mol);
            free(batches[i].labels);
        }
        free(batches);
    }

    return mpn;
}

void test(struct mpn *mpn, struct datapt *data, int start, int end) {
    int N = end - start;
    int l[N];
    for (int i = 0; i < N; i++) {
        l[i] = i;
    }

    struct batch *batch = batch_create(N, l, N, data + start);
    float *outputs = mpn_test(mpn, batch);

    int correct0 = 0;
    int correct1 = 0;
    int incorrect0 = 0;
    int incorrect1 = 0;
    int num0 = 0;
    int num1 = 0;
    for (int i = 0; i < N; i++) {
        if (batch->labels[i] == 1) {
            correct1 += (outputs[i] > 0.5);
            incorrect1 += (outputs[i] < 0.5);
            num1++;
        } else {
            correct0 += (outputs[i] < 0.5);
            incorrect0 += (outputs[i] > 0.5);
            num0++;
        }
    }
    free(outputs);

    free(batch->m_atoms);
    free_mol(batch->mol);
    free(batch->labels);
    free(batch);

    printf("%d correct of %d\n", correct0 + correct1, N);
    printf("%d of %d ones (%d)\n", correct1, num1, incorrect1);
    printf("%d of %d zeros (%d)\n", correct0, num0, incorrect0);
}

int main(int argc, char **argv) {
    // parse arguments
    int opt;
    int LABEL = 0;
    while ((opt = getopt(argc, argv, "b:e:l:sh")) != -1) {
        switch (opt) {
            case 'b':
                BATCHSIZE = atoi(optarg);
                break;

            case 'e':
                EPOCHS = atoi(optarg);
                break;

            case 'l':
                LABEL = atoi(optarg);
                break;

            case 's':
                cuda_set_device_flags(cudaDeviceScheduleBlockingSync);
                break;

            case 'h':
                printf("Options:\n");
                printf("-b [BATCHSIZE]\n");
                printf("-e [EPOCHS]\n");
                printf("-s - use blocking sync\n");
                return 0;

            case '?':
                if (optopt == 'b' || optopt == 'e') {
                    printf("Option -%c requires an argument!\n", optopt);
                } else if (isprint(optopt)) {
                    printf("Unknown option `-%c'.\n", optopt);
                } else {
                    printf("Unknown option character `\\x%x'.\n", optopt);
                }
                return 1;
        }
    }

    // parse datapoints
    int N;
    struct datapt *data = parse_datapts(stdin, LABEL, &N);
    printf("parsed %d data points\n", N);

    // run training
    struct mpn *mpn = train(data, N);

    cuda_device_synchronize();

    for (int i = 0; i < N; i++) {
        free_mol(data[i].mol);
    }
    free(data);

    return 0;
}
