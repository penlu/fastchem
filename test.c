#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mol.h"
#include "datapt.h"

int main(int argc, char **argv) {
    int N;
    struct datapt *data = parse_datapts(stdin, 0, &N);

    printf("parsed %d data points\n", N);

    for (int i = 0; i < N; i++) {
        free_mol(data[i].mol);
    }
    free(data);

    return 0;
}
