#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"
#include "nicslu.h"
#include "nicslu_util.h"

int my_DumpA(SNicsLU *nicslu, double **ax, unsigned int **ai, unsigned int **ap)
{
    uint__t n, nnz;
    double *ax0;
    unsigned int *ai0, *ap0;
    uint__t *rowperm, *pinv, *piv, oldrow, start, end;
    uint__t i, j, p;

    if (NULL == nicslu || NULL == ax || NULL == ai || NULL == ap)
    {
        return -1;
    }

    if (*ax != NULL)
    {
        free(*ax);
        *ax = NULL;
    }
    if (*ai != NULL)
    {
        free(*ai);
        *ai = NULL;
    }
    if (*ap != NULL)
    {
        free(*ap);
        *ap = NULL;
    }

    n = nicslu->n;
    nnz = nicslu->nnz;
    ax0 = nicslu->ax;
    ai0 = nicslu->ai;
    ap0 = nicslu->ap;
    rowperm = nicslu->row_perm;/*row_perm[i]=j-->row i in the permuted matrix is row j in the original matrix*/
    pinv = (uint__t *)nicslu->pivot_inv;/*pivot_inv[i]=j-->column i is the jth pivot column*/
    piv = (uint__t *)nicslu->pivot;

    // Generate identity pivots so we can dump the reordered/scaled matrix as-is.
    for (i = 0; i < n; ++i)
    {
        pinv[i] = i;
        piv[i] = i;
    }

    *ax = (double *)malloc(sizeof(double)*nnz);
    *ai = (unsigned int *)malloc(sizeof(unsigned int)*nnz);
    *ap = (unsigned int *)malloc(sizeof(unsigned int)*(n+1));
    // *ax = (real__t *)malloc(sizeof(real__t)*nnz);
    // *ai = (uint__t *)malloc(sizeof(uint__t)*nnz);
    // *ap = (uint__t *)malloc(sizeof(uint__t)*(n+1));

    if (NULL == *ax || NULL == *ai || NULL == *ap)
    {
        goto FAIL;
    }
    (*ap)[0] = 0;

    p = 0;
    for (i=0; i<n; ++i)
    {
        oldrow = rowperm[i];
        start = ap0[oldrow];
        end = ap0[oldrow+1];
        (*ap)[i+1] = (*ap)[i] + end - start;

        for (j=start; j<end; ++j)
        {
            (*ax)[p] = ax0[j];
            (*ai)[p++] = pinv[ai0[j]];
        }
    }

    return 0;

FAIL:
    if (*ax != NULL)
    {
        free(*ax);
        *ax = NULL;
    }
    if (*ai != NULL)
    {
        free(*ai);
        *ai = NULL;
    }
    if (*ap != NULL)
    {
        free(*ap);
        *ap = NULL;
    }
    return -2;
}

int preprocess(const char *matrixName, SNicsLU *nicslu, double **ax, unsigned int **ai, unsigned int **ap)
{
    int ret = NICS_OK;
    uint__t n = 0;
    uint__t nnz = 0;

    if (matrixName == NULL || nicslu == NULL || ax == NULL || ai == NULL || ap == NULL)
        return -1;

    *ax = NULL;
    *ai = NULL;
    *ap = NULL;

    // Caller owns nicslu; preprocess only initializes and analyzes it.
    NicsLU_Initialize(nicslu);

    printf("Reading matrix...\n");

    // NICSLU API takes non-const file path but does not modify it.
    ret = NicsLU_ReadTripletColumnToSparse((char *)matrixName, &n, &nnz, ax, ai, ap);
    if (ret == NICSLU_MATRIX_INVALID)
    {    
        printf("Read invalid matrix\n");
        goto EXIT;
    }
    else if (ret == NICSLU_FILE_CANNOT_OPEN) 
    {    
        printf("File cannot open\n");
        goto EXIT;
    }
    else if (ret != NICS_OK) 
    {    
        printf("Open file error\n");
        goto EXIT;
    }

    ret = NicsLU_CreateMatrix(nicslu, n, nnz, *ax, *ai, *ap);
    if (ret != NICS_OK)
    {
        printf("Create matrix error\n");
        goto EXIT;
    }

    nicslu->cfgi[0] = 1;
    nicslu->cfgf[1] = 0;

    printf("Preprocessing matrix...\n");

    ret = NicsLU_Analyze(nicslu);
    if (ret != NICS_OK)
    {
        printf("Analyze matrix error\n");
        goto EXIT;
    }
    printf("Preprocessing time: %f ms\n", nicslu->stat[0] * 1000);

    ret = my_DumpA(nicslu, ax, ai, ap);
    if (ret != 0)
    {
        printf("DumpA failed\n");
        goto EXIT;
    }
    //rp = nicslu->col_perm;
    //cp = nicslu->row_perm_inv;
    //piv = nicslu->pivot;
    //rows = nicslu->col_scale_perm;
    //cols = nicslu->row_scale;
    //cscale = nicslu->cscale;

    return 0;
EXIT:
    // Keep ownership simple: destroy NICSLU internals, but do not free caller memory.
    NicsLU_Destroy(nicslu);
    if (*ax != NULL)
    {
        free(*ax);
        *ax = NULL;
    }
    if (*ai != NULL)
    {
        free(*ai);
        *ai = NULL;
    }
    if (*ap != NULL)
    {
        free(*ap);
        *ap = NULL;
    }
    return -1;

}
