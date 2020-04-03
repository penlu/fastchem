#define ATOM_FDIM 133
#define BOND_FDIM 14

// graph representation of one molecule
struct mol {
    int n_atoms; // number of atoms
    int n_bonds; // number of bonds

    float *f_atoms; // (n_atoms * ATOM_FDIM), atom features
    float *f_bonds; // (n_bonds * BOND_FDIM), bond features

    // flattened list representation of bonds arriving at each atom
    int *a_bonds; // (n_atoms + 1), starting index in a2b for each atom
    int *a2b; // (n_bonds), flattened list of bonds arriving at each atom

    int *b2a; // (n_bonds), source atom for each bond
    int *b2revb; // (n_bonds), reverse-direction bond
};

void free_mol(struct mol *);
struct mol *parse_mol(FILE *);
