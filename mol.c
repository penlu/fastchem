#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mol.h"

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

void free_mol(struct mol *mol) {
    free(mol->f_atoms);
    free(mol->f_bonds);
    free(mol->a_bonds);
    free(mol->a2b);
    free(mol->b2a);
    free(mol->b2revb);
    free(mol);
}

// parse molecule from given file handle
// if read is short, returns NULL
// not gonna handle other possible errors
struct mol *parse_mol(FILE *f) {
    struct mol *mol = malloc(sizeof(struct mol));

    // read counts
    if (read_line(f)) {
        goto cleanup;
    }
    mol->n_atoms = strtol(BUF, NULL, 10);

    if (read_line(f)) {
        goto cleanup;
    }
    mol->n_bonds = strtol(BUF, NULL, 10);

    // allocate feature storage
    mol->f_atoms = malloc(sizeof(float) * mol->n_atoms * ATOM_FDIM);
    mol->f_bonds = malloc(sizeof(float) * mol->n_bonds * BOND_FDIM);

    // allocate graph storage
    mol->a_bonds = malloc(sizeof(int) * (mol->n_atoms + 1));
    mol->a2b = malloc(sizeof(int) * mol->n_bonds);
    mol->b2a = malloc(sizeof(int) * mol->n_bonds);
    mol->b2revb = malloc(sizeof(int) * mol->n_bonds);

    // read atom features
    for (int i = 0; i < mol->n_atoms; i++) {
        if (read_line(f)) {
            goto cleanup2;
        }

        char *p = BUF;
        char *end;
        for (int f = 0; f < ATOM_FDIM; f++) {
            mol->f_atoms[i * ATOM_FDIM + f] = strtof(p, &end);
            p = end + 1;
        }
    }

    // read bond features
    for (int i = 0; i < mol->n_bonds; i++) {
        if (read_line(f)) {
            goto cleanup2;
        }

        char *p = BUF;
        char *end;
        for (int f = 0; f < BOND_FDIM; f++) {
            mol->f_bonds[i * BOND_FDIM + f] = strtof(p, &end);
            p = end + 1;
        }
    }

    // read a2b
    int bonds = 0;
    for (int i = 0; i < mol->n_atoms; i++) {
        if (read_line(f)) {
            goto cleanup2;
        }

        mol->a_bonds[i] = bonds;
        char *end;
        for (char *p = BUF; p < BUF + BUFLEN; p = end + 1) {
            mol->a2b[bonds++] = strtol(p, &end, 10);
        }
    }
    mol->a_bonds[mol->n_atoms] = bonds;
    if (bonds != mol->n_bonds) {
        goto cleanup2;
    }

    // read b2a
    char *p;
    char *end;
    if (read_line(f)) {
        goto cleanup2;
    }
    p = BUF;
    for (int i = 0; i < mol->n_bonds; i++) {
        mol->b2a[i] = strtol(p, &end, 10);
        p = end + 1;
    }

    // read b2revb
    if (read_line(f)) {
        goto cleanup2;
    }
    p = BUF;
    for (int i = 0; i < mol->n_bonds; i++) {
        mol->b2revb[i] = strtol(p, &end, 10);
        p = end + 1;
    }

    return mol;

cleanup2:
    free(mol->f_atoms);
    free(mol->f_bonds);
    free(mol->a_bonds);
    free(mol->a2b);
    free(mol->b2a);
    free(mol->b2revb);
    printf("PARTIAL READ!\n");

cleanup:
    free(mol);

    return NULL;
}
