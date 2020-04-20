void cuda_malloc(void **, size_t);
void cuda_free(void *);
void cuda_memcpy_htod(void *, void *, size_t);
void cuda_memcpy_dtoh(void *, void *, size_t);
void cuda_memcpy_2d(void *, size_t, void *, size_t, size_t, size_t);
void cuda_device_synchronize(void);
void cuda_set_device_flags(unsigned int);

// cublas
void cublas_saxpy(int, float, float *, int, float *, int);
void cublas_sscal(int, float, float *, int);
void cublas_sgemm(int, int, int, int, int,
    float, float *, int, float *, int, float, float *, int);
void cublas_sgemv(int, int, int, float, float *, int, float *, float, float *);
void cublas_sger(int, int, float, float *, int, float *, int, float *, int);

// cusparse
void cusparse_scsr2csc(int, int, int,
    float *, int *, int *, float *, int *, int *);
void cusparse_sgemmi(int, int, int, int,
    float, float *, int, float *, int *, int *, float, float *, int);
void cusparse_scsrmm2(int, int, int, int, int,
    int, float, float *, int *, int *, float *, int, float, float *, int);

// curand
void curand_seed_generator(unsigned long long);
void curand_generate_uniform(float *, size_t);
void curand_generate_normal(float *, size_t, float, float);
