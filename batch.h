#define BATCHSIZE 50

struct batch {
    int n_mols; // number of molecules
    int *m_atoms; // (n_mols + 1) start indices for atoms in molecules
    struct mol *mol; // disconnected molecule graph
    float *labels; // set of labels
};

struct batch *batch_create(int, int *, int, struct datapt *);
