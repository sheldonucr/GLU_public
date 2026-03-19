#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nicslu.h"
#include "preprocess.h"
#include "symbolic.h"
#include "numeric.h"
#include "type.h"

namespace py = pybind11;

class GLUFactorization {
public:
    GLUFactorization(
        py::array_t<double,  py::array::c_contiguous> data,
        py::array_t<int32_t, py::array::c_contiguous> indices,
        py::array_t<int32_t, py::array::c_contiguous> indptr,
        int n_dim,
        bool perturb);

    py::array_t<double> solve(
        py::array_t<double, py::array::c_contiguous> b);

    unsigned int n() const { return m_n; }

    ~GLUFactorization();

private:
    SNicsLU          *m_nicslu = nullptr;
    Symbolic_Matrix  *m_sym    = nullptr;
    unsigned          m_n      = 0;
};


GLUFactorization::GLUFactorization(
    py::array_t<double,  py::array::c_contiguous> data,
    py::array_t<int32_t, py::array::c_contiguous> indices,
    py::array_t<int32_t, py::array::c_contiguous> indptr,
    int n_dim,
    bool perturb)
{
    if (n_dim <= 0)
        throw std::invalid_argument("n must be positive");

    unsigned n   = static_cast<unsigned>(n_dim);
    unsigned nnz = static_cast<unsigned>(data.shape(0));

    if (indices.shape(0) != static_cast<py::ssize_t>(nnz))
        throw std::invalid_argument("data and indices length mismatch");
    if (indptr.shape(0) != static_cast<py::ssize_t>(n + 1))
        throw std::invalid_argument("indptr length must be n+1");

    auto d   = data.unchecked<1>();
    auto idx = indices.unchecked<1>();
    auto ptr = indptr.unchecked<1>();

    /* Convert int32 indices to unsigned int (validate non-negative) */
    std::vector<unsigned> ai_u(nnz), ap_u(n + 1);
    for (unsigned i = 0; i < nnz; ++i) {
        if (idx(i) < 0)
            throw std::invalid_argument("Negative row index in CSC matrix");
        ai_u[i] = static_cast<unsigned>(idx(i));
    }
    for (unsigned i = 0; i <= n; ++i) {
        if (ptr(i) < 0)
            throw std::invalid_argument("Negative indptr value");
        ap_u[i] = static_cast<unsigned>(ptr(i));
    }

    m_nicslu = static_cast<SNicsLU *>(std::malloc(sizeof(SNicsLU)));
    if (!m_nicslu)
        throw std::bad_alloc();

    double       *ax_out = nullptr;
    unsigned int *ai_out = nullptr;
    unsigned int *ap_out = nullptr;

    int ret = preprocess_from_arrays(
        n, nnz,
        data.data(),
        ai_u.data(),
        ap_u.data(),
        m_nicslu,
        &ax_out, &ai_out, &ap_out);

    if (ret != 0) {
        std::free(m_nicslu);
        m_nicslu = nullptr;
        throw std::runtime_error(
            "GLU preprocessing failed (error " + std::to_string(ret) + ")");
    }

    m_n = m_nicslu->n;

    std::ostringstream out_ss, err_ss;
    m_sym = new Symbolic_Matrix(m_n, out_ss, err_ss);
    m_sym->fill_in(ai_out, ap_out);
    m_sym->csr();
    m_sym->predictLU(ai_out, ap_out, ax_out);
    m_sym->leveling();

    /* CSC arrays no longer needed after symbolic phase */
    std::free(ax_out);
    std::free(ai_out);
    std::free(ap_out);

    /* GPU factorization — overwrites m_sym->val with LU factors */
    LUonDevice(*m_sym, out_ss, err_ss, perturb);

    /* Surface CUDA errors: scan for NaN/Inf in the LU factors */
    std::string err_msg = err_ss.str();
    if (!err_msg.empty()) {
        bool bad = false;
        for (float v : m_sym->val)
            if (std::isnan(v) || std::isinf(v)) { bad = true; break; }
        if (bad)
            throw std::runtime_error("GPU factorization failed: " + err_msg);
    }
}


py::array_t<double> GLUFactorization::solve(
    py::array_t<double, py::array::c_contiguous> b_arr)
{
    if (static_cast<unsigned>(b_arr.shape(0)) != m_n)
        throw std::invalid_argument(
            "RHS length " + std::to_string(b_arr.shape(0)) +
            " != matrix dimension " + std::to_string(m_n));

    auto b_buf = b_arr.unchecked<1>();

    /* double → float (REAL) */
    std::vector<REAL> rhs(m_n);
    for (unsigned i = 0; i < m_n; ++i)
        rhs[i] = static_cast<REAL>(b_buf(i));

    std::vector<REAL> x_f = m_sym->solve(m_nicslu, rhs);

    /* float (REAL) → double */
    py::array_t<double> x_out(static_cast<py::ssize_t>(m_n));
    auto x_buf = x_out.mutable_unchecked<1>();
    for (unsigned i = 0; i < m_n; ++i)
        x_buf(i) = static_cast<double>(x_f[i]);

    return x_out;
}


GLUFactorization::~GLUFactorization()
{
    delete m_sym;
    m_sym = nullptr;
    if (m_nicslu) {
        NicsLU_Destroy(m_nicslu);
        std::free(m_nicslu);
        m_nicslu = nullptr;
    }
}


PYBIND11_MODULE(_pyglu, m) {
    m.doc() = "GLU GPU-accelerated sparse LU factorization — Python bindings";

    py::class_<GLUFactorization>(m, "GLUFactorization",
        R"(GPU LU factorization of a sparse matrix.

        Constructed by pyglu.splu(). Call .solve(b) to solve Ax=b.
        )")
        .def(py::init<
                py::array_t<double,  py::array::c_contiguous>,
                py::array_t<int32_t, py::array::c_contiguous>,
                py::array_t<int32_t, py::array::c_contiguous>,
                int,
                bool>(),
            py::arg("data"),
            py::arg("indices"),
            py::arg("indptr"),
            py::arg("n"),
            py::arg("perturb") = false,
            R"(Factorize a sparse matrix given in CSC format.

            Parameters
            ----------
            data : ndarray, float64, shape (nnz,)
            indices : ndarray, int32, shape (nnz,)   — row indices
            indptr  : ndarray, int32, shape (n+1,)   — column pointers
            n       : int                            — matrix dimension
            perturb : bool                           — enable GESP perturbation
            )")
        .def("solve", &GLUFactorization::solve, py::arg("b"),
            R"(Solve Ax = b.

            Parameters
            ----------
            b : ndarray, float64, shape (n,)

            Returns
            -------
            x : ndarray, float64, shape (n,)
            )")
        .def_property_readonly("n", &GLUFactorization::n,
            "Matrix dimension.");
}
