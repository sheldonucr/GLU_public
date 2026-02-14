#include <iostream>
#include <vector>
#include <set>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <fstream>
#include "symbolic.h"
#include "numeric.h"
#include "Timer.h"
#include "preprocess.h"
#include "nicslu.h"

using namespace std;

void help_message()
{
    cout << endl;
    cout << "GLU program V3.0" << endl;
    cout << "Usage: ./lu_cmd -i inputfile" << endl;
    cout << "Additional usage: ./lu_cmd -i inputfile -p" << endl;
    cout << "-p to enable perturbation" << endl;
}

int main(int argc, char** argv)
{
    Timer t;
    double utime;
    SNicsLU *nicslu = nullptr;

    const char *matrixName = nullptr;
    bool PERTURB = false;

    double *ax = NULL;
    unsigned int *ai = NULL, *ap = NULL;
    unsigned int n;

    if (argc < 3) {
        help_message();
        return -1;
    }

    for (int i = 1; i < argc;) {
        if (strcmp(argv[i], "-i") == 0) {
            if(i + 1 >= argc) {
                help_message();
                return -1;
            }
            matrixName = argv[i+1];
            i += 2;
        }
        else if (strcmp(argv[i], "-p") == 0) {
            PERTURB = true;
            i += 1;
        }        
        else {
            help_message();
            return -1;
        }
    }

    if (matrixName == nullptr) {
        help_message();
        return -1;
    }

    nicslu = static_cast<SNicsLU *>(malloc(sizeof(SNicsLU)));
    if (nicslu == nullptr) {
        cerr << "Failed to allocate NicsLU context." << endl;
        return -1;
    }

    // Build a permutation/scaling-adjusted matrix and keep the numeric arrays in CSC.
    int err = preprocess(matrixName, nicslu, &ax, &ai, &ap);
    if (err)
    {
        cerr << "Matrix preprocessing failed." << endl;
        free(nicslu);
        return -1;
    }

    n = nicslu->n;

    cout << "Matrix Row: " << n << endl;
    cout << "Original nonzero: " << nicslu->nnz << endl;

    t.start();

    // Symbolic analysis: fill-in prediction, CSR transpose, and level scheduling.
    Symbolic_Matrix A_sym(n, cout, cerr);
    A_sym.fill_in(ai, ap);
    t.elapsedUserTime(utime);
    cout << "Symbolic time: " << utime << " ms" << endl;

    t.start();
    A_sym.csr();
    t.elapsedUserTime(utime);
    cout << "CSR time: " << utime << " ms" << endl;

    t.start();
    A_sym.predictLU(ai, ap, ax);
    t.elapsedUserTime(utime);
    cout << "PredictLU time: " << utime << " ms" << endl;

    t.start();
    A_sym.leveling();
    t.elapsedUserTime(utime);
    cout << "Leveling time: " << utime << " ms" << endl;

#if GLU_DEBUG
    A_sym.ABFTCalculateCCA();
//    A_sym.PrintLevel();
#endif

    // Numeric factorization on GPU updates A_sym.val in place to LU factors.
    LUonDevice(A_sym, cout, cerr, PERTURB);

#if GLU_DEBUG
    A_sym.ABFTCheckResult();
#endif

    // Solve Ax=b with the computed factors and NICSLU permutations/scales.
    vector<REAL> b(n, 1.);
    vector<REAL> x = A_sym.solve(nicslu, b);
    {
        ofstream x_f("x.dat");
        for (double xx: x)
            x_f << xx << '\n';
    }

    NicsLU_Destroy(nicslu);
    free(nicslu);
    free(ax);
    free(ai);
    free(ap);

    return 0;
}
