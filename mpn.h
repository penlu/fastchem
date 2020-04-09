struct mpn;
struct mpn *mpn_create(void);
void mpn_init(struct mpn *);
float mpn_forward(struct mpn *, struct mol *);
float mpn_backward(struct mpn *, struct mol *);
void mpn_adam(struct mpn *, int, float, float, float);
