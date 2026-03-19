#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nicslu.h"
#include "preprocess.h"

/*
 * preprocess_from_arrays — like preprocess() but accepts an in-memory CSC
 * matrix instead of a file path.  Intended for use by the Python bindings.
 *
 * Ownership rules:
 *   - Caller owns nicslu (allocated externally, e.g. malloc(sizeof(SNicsLU))).
 *   - NicsLU_CreateMatrix takes ownership of the internal copies of ax/ai/ap;
 *     do NOT free them after this call.
 *   - *ax_out, *ai_out, *ap_out are malloc'd by my_DumpA; caller must free.
 *   - On error, NicsLU_Destroy is called and all output pointers are NULL.
 */
int preprocess_from_arrays(
    unsigned int n,
    unsigned int nnz,
    const double *ax_in,
    const unsigned int *ai_in,
    const unsigned int *ap_in,
    SNicsLU *nicslu,
    double **ax_out,
    unsigned int **ai_out,
    unsigned int **ap_out)
{
    int ret;
    double    *ax_copy = NULL;
    uint__t   *ai_copy = NULL;
    uint__t   *ap_copy = NULL;

    if (!ax_in || !ai_in || !ap_in || !nicslu || !ax_out || !ai_out || !ap_out)
        return -1;
    if (n == 0 || nnz == 0)
        return -1;

    *ax_out = NULL;
    *ai_out = NULL;
    *ap_out = NULL;

    NicsLU_Initialize(nicslu);

    /* Heap-copy the caller's arrays.  NicsLU_CreateMatrix takes ownership of
     * these buffers, so we must not alias the caller's memory. */
    ax_copy = (double *)malloc(sizeof(double) * nnz);
    ai_copy = (uint__t *)malloc(sizeof(uint__t) * nnz);
    ap_copy = (uint__t *)malloc(sizeof(uint__t) * (n + 1));
    if (!ax_copy || !ai_copy || !ap_copy)
    {
        free(ax_copy);
        free(ai_copy);
        free(ap_copy);
        NicsLU_Destroy(nicslu);
        return -1;
    }

    memcpy(ax_copy, ax_in, sizeof(double) * nnz);
    /* unsigned int and uint__t are both 32-bit on the default NICSLU build */
    memcpy(ai_copy, ai_in, sizeof(uint__t) * nnz);
    memcpy(ap_copy, ap_in, sizeof(uint__t) * (n + 1));

    ret = NicsLU_CreateMatrix(nicslu, (uint__t)n, (uint__t)nnz,
                              ax_copy, ai_copy, ap_copy, NULL);
    /* NICSLU owns ax_copy/ai_copy/ap_copy from here on */
    ax_copy = NULL;
    ai_copy = NULL;
    ap_copy = NULL;

    if (ret != NICS_OK)
    {
        fprintf(stderr, "preprocess_from_arrays: NicsLU_CreateMatrix error %d\n", ret);
        NicsLU_Destroy(nicslu);
        return -1;
    }

    nicslu->cfgi[0] = 1;
    nicslu->cfgf[1] = 0;

    ret = NicsLU_Analyze(nicslu);
    if (ret != NICS_OK)
    {
        fprintf(stderr, "preprocess_from_arrays: NicsLU_Analyze error %d\n", ret);
        NicsLU_Destroy(nicslu);
        return -1;
    }

    ret = my_DumpA(nicslu, ax_out, ai_out, ap_out);
    if (ret != 0)
    {
        fprintf(stderr, "preprocess_from_arrays: my_DumpA error %d\n", ret);
        NicsLU_Destroy(nicslu);
        return -1;
    }

    return 0;
}
