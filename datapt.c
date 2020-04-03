#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mol.h"
#include "datapt.h"

static int BUFSZ = 0;
static int BUFLEN = 0;
static char *BUF = NULL;

// return: 0 if file continues else -1
static int read_line(FILE *f) {
    int r = 0;
    int c;
    while ((c = fgetc(f)) != EOF && c != '\n') {
        if (r >= BUFSZ) {
            BUFSZ += 1024;
            BUF = realloc(BUF, BUFSZ + 1);
        }
        BUF[r++] = c;
        BUF[r] = '\0';
    }
    BUFLEN = r;
    if (c == EOF) {
        return -1;
    } else {
        return 0;
    }
}

// target: get nth label
struct datapt *parse_datapts(FILE *f, int target, int *N) {
    int num = 0;
    struct datapt *data = NULL;
    struct mol *mol;
    while (mol = parse_mol(f)) {
        data = realloc(data, (num + 1) * sizeof(struct datapt));

        // parse the molecule
        data[num].mol = mol;

        // find nth label
        if (read_line(f)) {
            free(data);
            return NULL;
        }
        char *p = BUF;
        for (int i = 0; i < target; i++) {
            p = strchr(p, ',');
            p++;
        }
        char *end;
        data[num].label = strtof(p, &end);

        if (end != p) {
            num++;
        } else {
            free_mol(data[num].mol);
        }
    }
    *N = num;
    return data;
}
