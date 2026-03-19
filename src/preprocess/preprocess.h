#ifndef __PREPROCESS__
#define __PREPROCESS__

#include "nics_config.h"
#include "nicslu.h"
#include "type.h"

#define IN__
#define OUT__

#ifdef __cplusplus
extern "C" {
#endif

int my_DumpA(
    IN__ SNicsLU *nicslu,
    OUT__ double **ax,
    OUT__ unsigned int **ai,
    OUT__ unsigned int **ap);

int preprocess(
    IN__ const char *matrixName,
    IN__ SNicsLU *nicslu,
    OUT__ double **ax,
    OUT__ unsigned int **ai,
    OUT__ unsigned int **ap);

int preprocess_from_arrays(
    IN__ unsigned int n,
    IN__ unsigned int nnz,
    IN__ const double *ax_in,
    IN__ const unsigned int *ai_in,
    IN__ const unsigned int *ap_in,
    IN__ SNicsLU *nicslu,
    OUT__ double **ax_out,
    OUT__ unsigned int **ai_out,
    OUT__ unsigned int **ap_out);

#ifdef __cplusplus
}
#endif

#endif
