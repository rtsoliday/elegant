// sddsupsample.cc
//
// Upsample an SDDS particle distribution with required columns:
//   x, xp, y, yp, t, p
//
// Goals:
//   - Increase particles from N to M*N (M>1) while preserving non-Gaussian,
//     filamentary / manifold-like structure in 6D phase space.
//   - Read/write SDDS binary using the SDDS C API (soliday / elegant SDDS library).
//
// Methods:
//   --mode tangent (default): filament-preserving local tangent sampling in globally-whitened space.
//       * Tangent step: isotropic in local tangent subspace, magnitude ~ alpha * r_k, capped by tan-cap-frac * r_k.
//       * Optional orthogonal step: isotropic in orthogonal subspace, magnitude ~ beta * sigma_o (local orth thickness).
//       * Rejection: total step <= reject-factor*r_k AND orth step <= ortho-factor*sigma_o (if enabled).
//       * Non-6D numeric columns: copied from base particle.
//   --mode interp: SMOTE-like chord interpolation in globally-whitened space.
//       * Choose random neighbor among kNN, sample lambda in [0,1], point = base + lambda*(nbr-base) + noise*N(0,1)
//       * Non-6D numeric columns: interpolated using same (base,nbr,lambda). Strings copied from base.
//
// Notes:
//   - All geometry (PCA, steps, kNN) is done in globally-whitened coordinates.
//   - Local PCA is computed from kNN set. If your filaments curve/branch tightly, lower --k (e.g. 8–16).
//
// Build (example):
//   g++ -O3 -fopenmp -I/usr/include/eigen3 -o sddsupsample sddsupsample.cc \
//       -lSDDS1 -lm -lz
//
// Usage:
//   ./sddsupsample in.sdds out.sdds --multiplier 100 --mode tangent --k 16 --alpha 0.02 --beta 0 \
//       --tan-cap-frac 0.05 --ortho-factor 0.5 --reject-factor 1.01
//
//   ./sddsupsample in.sdds out.sdds --multiplier 100 --mode interp --k 32 --noise 0.02
//
// SPDX-License-Identifier: MIT

#include <SDDS.h>

#include <Eigen/Dense>
#include <nanoflann.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// -------------------------- small utilities --------------------------

static inline bool iequals(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i])) return false;
    }
    return true;
}

[[noreturn]] static void die(const std::string& msg) {
    std::cerr << "Error: " << msg << "\n";
    std::exit(1);
}

static void sdds_check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "SDDS error while: " << what << "\n";
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        std::exit(1);
    }
}

// -------------------------- nanoflann adaptor --------------------------

struct PointCloud6D {
    std::vector<double> data; // row-major [N][6]
    size_t N = 0;

    inline size_t kdtree_get_point_count() const { return N; }
    inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
        return data[idx * 6 + dim];
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTree6 = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloud6D>,
    PointCloud6D,
    6,
    size_t>;

struct LocalModel {
    Eigen::Matrix<double, 6, 6> vecs; // eigenvectors columns, descending eigenvalues
    Eigen::Matrix<double, 6, 1> vals; // eigenvalues descending
    int d_tangent = 1;
    double r_k = 0.0; // distance to k-th neighbor (whitened)
};

enum class Mode { Tangent, Interp };

// -------------------------- SDDS column storage --------------------------

enum class ColKind { NumericDouble, String, Other };

struct ColumnStore {
    std::string name;
    int32_t sddsType = 0;
    ColKind kind = ColKind::Other;

    std::vector<double> in_num;      // numeric as double
    std::vector<std::string> in_str; // string

    std::vector<double> out_num;
    std::vector<std::string> out_str;
};

static ColKind classify_sdds_type(int32_t t) {
    switch (t) {
        case SDDS_DOUBLE:
        case SDDS_FLOAT:
        case SDDS_LONG:
        case SDDS_ULONG:
        case SDDS_SHORT:
        case SDDS_USHORT:
        case SDDS_LONG64:
        case SDDS_ULONG64:
            return ColKind::NumericDouble;
        case SDDS_STRING:
        case SDDS_CHARACTER:
            return ColKind::String;
        default:
            return ColKind::Other;
    }
}

// -------------------------- whitening + models --------------------------

static void compute_whitening(
    const Eigen::MatrixXd& X, // Nx6
    Eigen::Matrix<double, 6, 1>& mu,
    Eigen::Matrix<double, 6, 6>& W,
    Eigen::Matrix<double, 6, 6>& Winv
) {
    const int64_t N = X.rows();
    mu = X.colwise().mean().transpose();
    Eigen::MatrixXd X0 = X.rowwise() - mu.transpose();

    Eigen::Matrix<double, 6, 6> C =
        (X0.transpose() * X0) / double(std::max<int64_t>(1, N - 1));

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(C);
    if (es.info() != Eigen::Success) die("Eigen decomposition failed for global covariance.");

    Eigen::Matrix<double, 6, 1> vals = es.eigenvalues();   // ascending
    Eigen::Matrix<double, 6, 6> vecs = es.eigenvectors();  // columns correspond

    const double eps = 1e-14;
    for (int i = 0; i < 6; ++i) vals(i) = std::max(vals(i), eps);

    Eigen::Matrix<double, 6, 6> DinvSqrt = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 6> DSqrt    = Eigen::Matrix<double, 6, 6>::Zero();
    for (int i = 0; i < 6; ++i) {
        DinvSqrt(i, i) = 1.0 / std::sqrt(vals(i));
        DSqrt(i, i) = std::sqrt(vals(i));
    }

    // Whitening: xw = (x - mu)^T * W  (we use row vectors; see usage)
    W    = vecs * DinvSqrt * vecs.transpose();
    Winv = vecs * DSqrt    * vecs.transpose();
}

static int choose_tangent_dim(const Eigen::Matrix<double, 6, 1>& vals_desc, double var_keep) {
    double total = vals_desc.sum();
    if (!(total > 0.0)) return 1;
    double cum = 0.0;
    for (int d = 1; d <= 6; ++d) {
        cum += vals_desc(d - 1);
        if (cum / total >= var_keep) return d;
    }
    return 6;
}

static std::vector<LocalModel> precompute_local_models(
    const Eigen::MatrixXd& Xw,
    KDTree6& kdtree,
    int k,
    double var_keep,
    std::vector<int32_t>& neighbors_out,
    int& k_eff_out
) {
    const size_t N = (size_t)Xw.rows();
    const int k_eff = std::max(2, std::min(k, int(N) - 1));
    k_eff_out = k_eff;

    neighbors_out.assign(N * (size_t)k_eff, 0);

    std::vector<LocalModel> models(N);

    std::vector<size_t> idx(k_eff + 1);
    std::vector<double> dist2(k_eff + 1);
    nanoflann::KNNResultSet<double, size_t> rs(k_eff + 1);

    for (size_t i = 0; i < N; ++i) {
        rs.init(idx.data(), dist2.data());
        kdtree.findNeighbors(rs, Xw.row((int)i).data(), nanoflann::SearchParameters(32));

        models[i].r_k = std::sqrt(std::max(dist2.back(), 0.0));

        // neighbors excluding self
        std::vector<size_t> nbrs;
        nbrs.reserve((size_t)k_eff);
        for (size_t j = 0; j < idx.size(); ++j) {
            if (idx[j] == i) continue;
            nbrs.push_back(idx[j]);
            if ((int)nbrs.size() == k_eff) break;
        }
        while ((int)nbrs.size() < k_eff) nbrs.push_back(i);

        for (int j = 0; j < k_eff; ++j)
            neighbors_out[i * (size_t)k_eff + (size_t)j] = (int32_t)nbrs[(size_t)j];

        // local PCA from neighbor cloud (in whitened space)
        Eigen::MatrixXd Xi(k_eff, 6);
        for (int j = 0; j < k_eff; ++j) Xi.row(j) = Xw.row((int)nbrs[(size_t)j]);

        Eigen::Matrix<double, 6, 1> mu_loc = Xi.colwise().mean().transpose();
        Eigen::MatrixXd Xc = Xi.rowwise() - mu_loc.transpose();
        Eigen::Matrix<double, 6, 6> C =
            (Xc.transpose() * Xc) / double(std::max(1, k_eff - 1));

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(C);
        if (es.info() != Eigen::Success) {
            models[i].vals.setOnes();
            models[i].vecs.setIdentity();
            models[i].d_tangent = 1;
            continue;
        }

        Eigen::Matrix<double, 6, 1> vals = es.eigenvalues();   // ascending
        Eigen::Matrix<double, 6, 6> vecs = es.eigenvectors();

        // ascending -> descending
        for (int j = 0; j < 3; ++j) {
            std::swap(vals(j), vals(5 - j));
            vecs.col(j).swap(vecs.col(5 - j));
        }

        for (int j = 0; j < 6; ++j) vals(j) = std::max(vals(j), 1e-14);

        models[i].vals = vals;
        models[i].vecs = vecs;
        models[i].d_tangent = choose_tangent_dim(vals, var_keep);
    }

    return models;
}

// Filament-preserving local tangent step in whitened space.
// - Tangent step magnitude is set by alpha * r_k (NOT by sqrt(eigenvalue)).
// - Tangent step is capped at tan_cap_frac * r_k (0 disables).
// - Optional orthogonal step magnitude is set by beta * sigma_o (sigma_o = orth thickness).
static Eigen::RowVector<double, 6> sample_tangent_step(
    const LocalModel& m,
    std::mt19937_64& gen,
    std::normal_distribution<double>& ndist,
    double alpha,
    double beta,
    double tan_cap_frac
) {
    Eigen::Matrix<double, 6, 1> z;
    for (int i = 0; i < 6; ++i) z(i) = ndist(gen);

    const int d = std::max(1, std::min(m.d_tangent, 6));
    const double rk = std::max(m.r_k, 1e-14);

    // Tangent direction (unit vector in tangent subspace)
    Eigen::Matrix<double, 6, 1> step_t = Eigen::Matrix<double, 6, 1>::Zero();
    for (int j = 0; j < d; ++j) step_t += m.vecs.col(j) * z(j);

    double nt = step_t.norm();
    if (nt > 0.0) step_t *= (alpha * rk / nt);

    if (tan_cap_frac > 0.0) {
        const double cap = tan_cap_frac * rk;
        const double nrm = step_t.norm();
        if (nrm > cap) step_t *= (cap / nrm);
    }

    // Orthogonal step (scaled to local orth thickness)
    Eigen::Matrix<double, 6, 1> step_o = Eigen::Matrix<double, 6, 1>::Zero();
    if (beta > 0.0 && d < 6) {
        double s = 0.0;
        for (int j = d; j < 6; ++j) s += std::max(m.vals(j), 1e-14);
        const double sigma_o = std::sqrt(s / double(6 - d));

        for (int j = d; j < 6; ++j) step_o += m.vecs.col(j) * z(j);

        double no = step_o.norm();
        if (no > 0.0) step_o *= (beta * sigma_o / no);
    }

    return (step_t + step_o).transpose();
}

// -------------------------- CLI --------------------------

struct Args {
    std::string input;
    std::string output;
    int64_t multiplier = 0;

    Mode mode = Mode::Tangent;

    // shared
    int k = 32;
    uint64_t seed = 12345;

    // interp
    double noise = 0.02; // whitened units; 0 disables

    // tangent (whitened geometry controls)
    double alpha = 0.02;        // tangent magnitude = alpha * r_k
    double beta  = 0.0;         // orth magnitude = beta * sigma_o
    double tan_cap_frac = 0.05; // cap ||dx_tan|| <= tan_cap_frac * r_k (0 disables)
    double ortho_factor = 0.5;  // reject if ||dx_orth|| > ortho_factor*sigma_o (0 disables)

    double var_keep = 0.99;     // choose tangent dim to keep this variance fraction
    double reject_factor = 1.01; // reject if ||dx|| > reject_factor * r_k
    int max_tries = 50;
};

static void usage() {
    std::cerr <<
        "Usage:\n"
        "  sddsupsample in.sdds out.sdds --multiplier M [options]\n\n"
        "Required:\n"
        "  --multiplier M        upsample factor (>1)\n\n"
        "Options:\n"
        "  --mode MODE           tangent|interp (default tangent)\n"
        "  --k K                 kNN neighborhood (default 32)\n"
        "  --seed S              RNG seed (default 12345)\n\n"
        "Tangent mode options:\n"
        "  --alpha A             ||dx_tan|| ~ A*r_k (default 0.02)\n"
        "  --beta B              ||dx_orth|| ~ B*sigma_o (default 0.0)\n"
        "  --tan-cap-frac F      cap ||dx_tan|| <= F*r_k (default 0.05; 0 disables)\n"
        "  --ortho-factor F      require ||dx_orth|| <= F*sigma_o (default 0.5; 0 disables)\n"
        "  --var-keep V          tangent variance fraction (default 0.99)\n"
        "  --reject-factor R     require ||dx|| <= R*r_k (default 1.01)\n"
        "  --max-tries T         attempts per new particle (default 50)\n\n"
        "Interp mode options:\n"
        "  --noise S             Gaussian noise (whitened), default 0.02; 0 disables\n";
}

static Args parse_args(int argc, char** argv) {
    if (argc < 4) { usage(); die("Not enough arguments."); }

    Args a;
    a.input = argv[1];
    a.output = argv[2];

    for (int i = 3; i < argc; ++i) {
        std::string opt = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) die(std::string("Missing value for ") + name);
            return argv[++i];
        };

        if (opt == "--multiplier" || opt == "-M") {
            a.multiplier = std::stoll(need("--multiplier"));
        } else if (opt == "--mode") {
            std::string m = need("--mode");
            if (iequals(m, "tangent")) a.mode = Mode::Tangent;
            else if (iequals(m, "interp")) a.mode = Mode::Interp;
            else die("--mode must be 'tangent' or 'interp'");
        } else if (opt == "--k") {
            a.k = std::stoi(need("--k"));
        } else if (opt == "--seed") {
            a.seed = (uint64_t)std::stoull(need("--seed"));
        } else if (opt == "--noise") {
            a.noise = std::stod(need("--noise"));
        } else if (opt == "--alpha") {
            a.alpha = std::stod(need("--alpha"));
        } else if (opt == "--beta") {
            a.beta = std::stod(need("--beta"));
        } else if (opt == "--tan-cap-frac") {
            a.tan_cap_frac = std::stod(need("--tan-cap-frac"));
        } else if (opt == "--ortho-factor") {
            a.ortho_factor = std::stod(need("--ortho-factor"));
        } else if (opt == "--var-keep") {
            a.var_keep = std::stod(need("--var-keep"));
        } else if (opt == "--reject-factor") {
            a.reject_factor = std::stod(need("--reject-factor"));
        } else if (opt == "--max-tries") {
            a.max_tries = std::stoi(need("--max-tries"));
        } else {
            usage();
            die("Unknown option: " + opt);
        }
    }

    if (a.multiplier <= 1) die("Multiplier M must be > 1.");
    if (a.k < 1) die("--k must be >= 1.");
    if (a.alpha < 0 || a.beta < 0) die("--alpha/--beta must be >= 0.");
    if (a.tan_cap_frac < 0) die("--tan-cap-frac must be >= 0.");
    if (a.ortho_factor < 0) die("--ortho-factor must be >= 0.");
    if (a.noise < 0) die("--noise must be >= 0.");
    if (a.var_keep < 0.5 || a.var_keep > 1.0) die("--var-keep must be in [0.5, 1.0].");
    if (a.reject_factor <= 0) die("--reject-factor must be > 0.");
    if (a.max_tries < 1) die("--max-tries must be >= 1.");
    return a;
}

// -------------------------- SDDS helpers --------------------------

static int32_t get_column_type(SDDS_TABLE* in, const std::string& name) {
    int32_t type = 0;
    if (!SDDS_GetColumnInformation(in,
                                   const_cast<char*>("type"),
                                   &type,
                                   SDDS_GET_BY_NAME,
                                   const_cast<char*>(name.c_str()))) {
        return 0;
    }
    return type;
}

static std::vector<double> get_column_as_doubles(SDDS_TABLE* in, const std::string& name, int64_t nrows) {
    double* ptr = (double*)SDDS_GetColumnInDoubles(in, const_cast<char*>(name.c_str()));
    if (!ptr) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        die("Failed SDDS_GetColumnInDoubles for column: " + name);
    }
    std::vector<double> v(ptr, ptr + nrows);
    free(ptr);
    return v;
}

static std::vector<std::string> get_column_as_strings(SDDS_TABLE* in, const std::string& name, int64_t nrows) {
    char** ptr = (char**)SDDS_GetColumn(in, const_cast<char*>(name.c_str()));
    if (!ptr) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        die("Failed SDDS_GetColumn for string column: " + name);
    }
    std::vector<std::string> v;
    v.reserve((size_t)nrows);
    for (int64_t i = 0; i < nrows; ++i) v.emplace_back(ptr[i] ? ptr[i] : "");
    SDDS_FreeStringArray(ptr, nrows);
    return v;
}

static void set_column_from_doubles(SDDS_TABLE* out, const std::string& name, const std::vector<double>& v) {
    if (!SDDS_SetColumnFromDoubles(out,
                                  SDDS_SET_BY_NAME,
                                  const_cast<double*>(v.data()),
                                  (int64_t)v.size(),
                                  const_cast<char*>(name.c_str()))) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        die("Failed SDDS_SetColumnFromDoubles for column: " + name);
    }
}

static void set_column_from_strings(SDDS_TABLE* out, const std::string& name, const std::vector<std::string>& v) {
    std::vector<char*> tmp;
    tmp.reserve(v.size());
    for (const auto& s : v) tmp.push_back(const_cast<char*>(s.c_str()));
    if (!SDDS_SetColumn(out,
                        SDDS_SET_BY_NAME,
                        (void*)tmp.data(),
                        (int64_t)tmp.size(),
                        const_cast<char*>(name.c_str()))) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        die("Failed SDDS_SetColumn for string column: " + name);
    }
}

// -------------------------- page processing --------------------------

static void process_page(SDDS_TABLE* in, SDDS_TABLE* out, const Args& args) {
    const int64_t N = SDDS_RowCount(in);
    if (N <= 0) return;
    if (N < 2) die("Need at least 2 particles per page to upsample.");

    const int64_t Nout = args.multiplier * N;

    int32_t nCols = 0;
    char** colNames = SDDS_GetColumnNames(in, &nCols);
    if (!colNames || nCols <= 0) die("No columns found.");

    std::vector<ColumnStore> cols;
    cols.reserve((size_t)nCols);

    for (int32_t i = 0; i < nCols; ++i) {
        std::string name = colNames[i];
        int32_t t = get_column_type(in, name);

        ColumnStore cs;
        cs.name = name;
        cs.sddsType = t;
        cs.kind = classify_sdds_type(t);

        if (cs.kind == ColKind::NumericDouble) cs.in_num = get_column_as_doubles(in, name, N);
        else if (cs.kind == ColKind::String)  cs.in_str = get_column_as_strings(in, name, N);

        cols.emplace_back(std::move(cs));
    }
    SDDS_FreeStringArray(colNames, nCols);

    auto find_col = [&](const std::string& n) -> ColumnStore* {
        for (auto& c : cols) if (iequals(c.name, n)) return &c;
        return nullptr;
    };

    ColumnStore* cx  = find_col("x");
    ColumnStore* cxp = find_col("xp");
    ColumnStore* cy  = find_col("y");
    ColumnStore* cyp = find_col("yp");
    ColumnStore* ct  = find_col("t");
    ColumnStore* cp  = find_col("p");
    if (!cx || !cxp || !cy || !cyp || !ct || !cp)
        die("Input must contain numeric columns: x, xp, y, yp, t, p");

    Eigen::MatrixXd X(N, 6);
    for (int64_t i = 0; i < N; ++i) {
        X(i,0) = cx->in_num[(size_t)i];
        X(i,1) = cxp->in_num[(size_t)i];
        X(i,2) = cy->in_num[(size_t)i];
        X(i,3) = cyp->in_num[(size_t)i];
        X(i,4) = ct->in_num[(size_t)i];
        X(i,5) = cp->in_num[(size_t)i];
    }

    Eigen::Matrix<double, 6, 1> mu;
    Eigen::Matrix<double, 6, 6> W, Winv;
    compute_whitening(X, mu, W, Winv);

    // Xw: whitened coordinates
    Eigen::MatrixXd Xw = (X.rowwise() - mu.transpose()) * W;

    // KD-tree backing store
    PointCloud6D pc;
    pc.N = (size_t)N;
    pc.data.resize((size_t)N * 6);
    for (int64_t i = 0; i < N; ++i)
        for (int d = 0; d < 6; ++d)
            pc.data[(size_t)i * 6 + (size_t)d] = Xw(i, d);

    KDTree6 kdtree(6, pc, nanoflann::KDTreeSingleIndexAdaptorParams(16));
    kdtree.buildIndex();

    // Precompute local PCA models + neighbor list
    std::vector<int32_t> neighbors;
    int k_eff = 0;
    std::vector<LocalModel> models = precompute_local_models(Xw, kdtree, args.k, args.var_keep, neighbors, k_eff);

    // RNG
    std::mt19937_64 gen(args.seed);
    std::normal_distribution<double> ndist(0.0, 1.0);
    std::uniform_int_distribution<int64_t> base_dist(0, N - 1);
    std::uniform_int_distribution<int> nbr_rank_dist(0, std::max(0, k_eff - 1));
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    // Bookkeeping for other columns (interp mode interpolation)
    std::vector<int64_t> base_idx((size_t)Nout);
    std::vector<int64_t> nbr_idx((size_t)Nout);
    std::vector<double>  lam((size_t)Nout, 0.0);

    for (int64_t i = 0; i < N; ++i) {
        base_idx[(size_t)i] = i;
        nbr_idx[(size_t)i]  = i;
        lam[(size_t)i]      = 0.0;
    }

    Eigen::MatrixXd Xw_out(Nout, 6);
    Xw_out.topRows(N) = Xw;

    for (int64_t j = N; j < Nout; ++j) {
        int64_t b = base_dist(gen);
        base_idx[(size_t)j] = b;

        if (args.mode == Mode::Tangent) {
            nbr_idx[(size_t)j] = b;
            lam[(size_t)j] = 0.0;

            const LocalModel& m = models[(size_t)b];
            const Eigen::RowVector<double, 6> x0 = Xw.row(b);

            Eigen::RowVector<double, 6> cand = x0;
            bool ok = false;

            const double rk = std::max(m.r_k, 1e-14);
            const double total_lim = args.reject_factor * rk;
            const int d = std::max(1, std::min(m.d_tangent, 6));

            // orthogonal thickness sigma_o for rejection
            double sigma_o = 0.0;
            if (d < 6) {
                double s = 0.0;
                for (int jj = d; jj < 6; ++jj)
                    s += std::max(m.vals(jj), 1e-14);
                sigma_o = std::sqrt(s / double(6 - d));
            }
            const bool ortho_enabled = (args.ortho_factor > 0.0 && sigma_o > 0.0);
            const double ortho_lim = ortho_enabled ? (args.ortho_factor * sigma_o) : 0.0;

            for (int ttry = 0; ttry < args.max_tries; ++ttry) {
                Eigen::RowVector<double, 6> step =
                    sample_tangent_step(m, gen, ndist, args.alpha, args.beta, args.tan_cap_frac);

                Eigen::Matrix<double,6,1> s = step.transpose();

                // project onto tangent basis to find orth component
                Eigen::Matrix<double,6,1> s_t = Eigen::Matrix<double,6,1>::Zero();
                for (int jj = 0; jj < d; ++jj) {
                    double c = m.vecs.col(jj).dot(s);
                    s_t += m.vecs.col(jj) * c;
                }
                Eigen::Matrix<double,6,1> s_o = s - s_t;

                if (s.norm() <= total_lim && (!ortho_enabled || s_o.norm() <= ortho_lim)) {
                    cand = x0 + step;
                    ok = true;
                    break;
                }
            }
            if (!ok) {
                Eigen::RowVector<double, 6> step =
                    sample_tangent_step(m, gen, ndist, 0.25 * args.alpha, 0.0, args.tan_cap_frac);
                cand = x0 + step;
            }
            Xw_out.row(j) = cand;

        } else { // Mode::Interp
            int r = nbr_rank_dist(gen);
            int64_t nidx = (int64_t)neighbors[(size_t)b * (size_t)k_eff + (size_t)r];
            if (nidx < 0 || nidx >= N) nidx = b;

            nbr_idx[(size_t)j] = nidx;
            double l = uni01(gen);
            lam[(size_t)j] = l;

            Eigen::RowVector<double, 6> x0 = Xw.row(b);
            Eigen::RowVector<double, 6> x1 = Xw.row(nidx);
            Eigen::RowVector<double, 6> cand = x0 + l * (x1 - x0);

            if (args.noise > 0.0)
                for (int d = 0; d < 6; ++d) cand(d) += args.noise * ndist(gen);

            Xw_out.row(j) = cand;
        }
    }

    // Unwhiten back to physical coordinates
    Eigen::MatrixXd X_out = Xw_out * Winv;
    X_out.rowwise() += mu.transpose();

    sdds_check(SDDS_StartPage(out, (int32_t)Nout), "SDDS_StartPage");

    for (auto& c : cols) {
        if (c.kind == ColKind::NumericDouble) {
            c.out_num.assign((size_t)Nout, 0.0);

            // 6D columns: always from generated X_out
            if (iequals(c.name, "x") || iequals(c.name, "xp") || iequals(c.name, "y") ||
                iequals(c.name, "yp") || iequals(c.name, "t") || iequals(c.name, "p")) {

                int col = 0;
                if (iequals(c.name, "x")) col = 0;
                else if (iequals(c.name, "xp")) col = 1;
                else if (iequals(c.name, "y")) col = 2;
                else if (iequals(c.name, "yp")) col = 3;
                else if (iequals(c.name, "t")) col = 4;
                else col = 5;

                for (int64_t i = 0; i < Nout; ++i) c.out_num[(size_t)i] = X_out(i, col);

            } else {
                if (args.mode == Mode::Interp) {
                    for (int64_t i = 0; i < Nout; ++i) {
                        int64_t b = base_idx[(size_t)i];
                        int64_t n = nbr_idx[(size_t)i];
                        double l  = lam[(size_t)i];
                        double vb = c.in_num[(size_t)b];
                        double vn = c.in_num[(size_t)n];
                        c.out_num[(size_t)i] = vb + l * (vn - vb);
                    }
                } else {
                    for (int64_t i = 0; i < Nout; ++i) {
                        int64_t b = base_idx[(size_t)i];
                        c.out_num[(size_t)i] = c.in_num[(size_t)b];
                    }
                }
            }

            set_column_from_doubles(out, c.name, c.out_num);

        } else if (c.kind == ColKind::String) {
            c.out_str.clear();
            c.out_str.reserve((size_t)Nout);
            for (int64_t i = 0; i < Nout; ++i) {
                int64_t b = base_idx[(size_t)i];
                c.out_str.emplace_back(c.in_str[(size_t)b]);
            }
            set_column_from_strings(out, c.name, c.out_str);

        } else {
            // Best-effort: numeric via doubles. Copy/interp like numeric.
            if (c.sddsType == 0)
                die("Unknown column type for '" + c.name + "'. Extend handling if needed.");

            std::vector<double> tmp = get_column_as_doubles(in, c.name, N);
            std::vector<double> outv((size_t)Nout);

            if (args.mode == Mode::Interp) {
                for (int64_t i = 0; i < Nout; ++i) {
                    int64_t b = base_idx[(size_t)i];
                    int64_t n = nbr_idx[(size_t)i];
                    double l  = lam[(size_t)i];
                    double vb = tmp[(size_t)b];
                    double vn = tmp[(size_t)n];
                    outv[(size_t)i] = vb + l * (vn - vb);
                }
            } else {
                for (int64_t i = 0; i < Nout; ++i)
                    outv[(size_t)i] = tmp[(size_t)base_idx[(size_t)i]];
            }

            set_column_from_doubles(out, c.name, outv);
        }
    }

    // Preserve parameters/arrays from input page
    sdds_check(SDDS_CopyParameters(out, in), "SDDS_CopyParameters");
    sdds_check(SDDS_CopyArrays(out, in), "SDDS_CopyArrays");
    sdds_check(SDDS_WritePage(out), "SDDS_WritePage");
}

// -------------------------- main --------------------------

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);

        SDDS_TABLE in_tbl;
        SDDS_TABLE out_tbl;
        std::memset(&in_tbl, 0, sizeof(in_tbl));
        std::memset(&out_tbl, 0, sizeof(out_tbl));

        sdds_check(SDDS_InitializeInput(&in_tbl, const_cast<char*>(args.input.c_str())),
                   "SDDS_InitializeInput");

        // Always write binary output
        sdds_check(SDDS_InitializeOutput(&out_tbl, SDDS_BINARY, 1,
                                         nullptr, nullptr,
                                         const_cast<char*>(args.output.c_str())),
                   "SDDS_InitializeOutput");

        sdds_check(SDDS_CopyLayout(&out_tbl, &in_tbl), "SDDS_CopyLayout");
        sdds_check(SDDS_WriteLayout(&out_tbl), "SDDS_WriteLayout");

        while (true) {
            int32_t code = SDDS_ReadPage(&in_tbl);
            if (code <= 0) break;
            process_page(&in_tbl, &out_tbl, args);
        }

        SDDS_Terminate(&in_tbl);
        SDDS_Terminate(&out_tbl);
        return 0;

    } catch (const std::exception& e) {
        die(std::string("Exception: ") + e.what());
    }
}
