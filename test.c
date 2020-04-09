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
        //printf("%d\n", data[i].mol->n_bonds);
        struct mol *mol = data[i].mol;
        for (int a = 0; a < mol->n_atoms; a++) {
            for (int b = mol->a_bonds[a]; b < mol->a_bonds[a + 1]; b++) {
                if (a != mol->b2a[mol->b2revb[mol->a2b[b]]]) {
                    printf("%d %d %d\b", a, b, mol->b2a[mol->b2revb[mol->a2b[b]]]);
                }
            }
        }
        free_mol(data[i].mol);
    }
    free(data);

    return 0;
}
