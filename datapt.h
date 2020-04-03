// representation of a datapoint
struct datapt {
    struct mol *mol; // associated molecule
    float label;
};

struct datapt *parse_datapts(FILE *f, int target, int *N);
