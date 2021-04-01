#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <locale>
#include <unordered_map>
#include <boost/math/distributions/normal.hpp>
#include <limits>
#include <chrono>
#include <random>
#include <Windows.h>

typedef std::chrono::steady_clock _clock_t;
typedef std::chrono::time_point<_clock_t> _time_t;

struct csv_file_t : std::ifstream {
    static std::ctype_base::mask const *get_table() {
        typedef std::ctype<char> ctype_t;
        static std::vector<ctype_t::mask> rc(ctype_t::table_size, ctype_t::mask());
        rc[','] = ctype_t::space;
        rc['\n'] = ctype_t::space;
        return rc.data();
    }

    csv_file_t(const std::string &file, bool skip_header = true) : std::ifstream(file) {
        imbue(std::locale(std::locale(), new std::ctype<char>(get_table())));
        if (skip_header) {
            std::string tmp;
            std::getline(*this, tmp);
        }
    }
};

void report_error(const std::string &msg) {
    std::cerr << msg << std::endl;
    throw std::runtime_error(msg);
}

void report_error_code(const std::string &msg, int code) {
    std::ostringstream ss;
    ss << msg << " [code: " << code << "]";
    std::cerr << ss.str() << std::endl;
    throw std::runtime_error(ss.str());
}

typedef cl_float3 mat3x3_t[3];

cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;

struct {
    cl_mem loans, losses, lower;
} buffers;

cl_kernel sim_ker;

#define _if_err_ret(msg) if (CL_SUCCESS != err) { report_error_code(msg, err); }

typedef struct {
    int region;
    float ead;
    float lgd;
    float pd;
    float alpha;
    float treshold;
    float gamma;
    int pad;
} loan_t;

std::vector<loan_t> loans;
float ch_lower[3 * 3];

void init_cl() {
    cl_int err;

    cl_uint num_plat;
    err = clGetPlatformIDs(0, NULL, &num_plat);
    _if_err_ret("clGetPlatformIDs() failed");
    if (0 == num_plat) {
        report_error("No platforms available");
    }

    cl_platform_id plat;
    err = clGetPlatformIDs(1, &plat, NULL);
    _if_err_ret("clGetPlatformIDs() failed");

    cl_uint num_dev;
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &device, &num_dev);
    _if_err_ret("clGetDeviceIDs() failed");
    if (0 == num_dev) {
        report_error("No devices available");
    }

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    _if_err_ret("clCreateContext() failed");
    queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
    _if_err_ret("clCreateCommandQueueWithProperties() failed");
}

void destroy_cl() {
    cl_int err;

    if (NULL != buffers.loans) {
        err = clReleaseMemObject(buffers.loans);
        _if_err_ret("clReleaseMemObject() failed");
    }
    if (NULL != buffers.losses) {
        err = clReleaseMemObject(buffers.losses);
        _if_err_ret("clReleaseMemObject() failed");
    }
    if (NULL != buffers.lower) {
        err = clReleaseMemObject(buffers.lower);
        _if_err_ret("clReleaseMemObject() failed");
    }

    if (NULL != sim_ker) {
        err = clReleaseKernel(sim_ker);
        _if_err_ret("clReleaseKernel() failed");
    }

    if (NULL != program) {
        err = clReleaseProgram(program);
        _if_err_ret("clReleaseProgram() failed");
    }
    if (NULL != queue) {
        err = clReleaseCommandQueue(queue);
        _if_err_ret("clReleaseCommandQueue() failed");
    }
    if (NULL != device) {
        err = clReleaseDevice(device);
        _if_err_ret("clReleaseDevice() failed");
    }
    if (NULL != context) {
        err = clReleaseContext(context);
        _if_err_ret("clReleaseContext() failed");
    }
}

void create_program() {
    cl_int err;

    std::ifstream in("simulation.cl");
    std::string str{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};

    const char *algo_src = str.c_str();
    size_t len = str.length();
    program = clCreateProgramWithSource(context, 1, &algo_src, &len, &err);
    _if_err_ret("clCreateProgramWithSource() failed");

    err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
    if (CL_SUCCESS != err) {
        if (CL_BUILD_PROGRAM_FAILURE != err) {
            report_error("clBuildProgram() failed without a build failure");
        }

        size_t log_len;
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_len);
        _if_err_ret("clGetProgramBuildInfo() failed");

        std::vector<char> log(log_len);
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_len, log.data(), NULL);
        _if_err_ret("clGetProgramBuildInfo() failed");

        report_error(std::string(log.begin(), log.end()));
    }
}

// Cholesky-Banachiewicz algorithm to compute the lower triangular matrix
// of the decomposition of a n*n matrix.
template<class T>
void cholesky_lower(const T *matrix, T *result, size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            T s = matrix[i * n + j];
            for (size_t k = 0; k < j; k++) {
                s -= result[i * n + k] * result[j * n + k];
            }
            if (i == j) {
                T q = std::sqrt(s);
                if (std::abs(q) < std::numeric_limits<T>::epsilon()) {
                    throw std::runtime_error("Matrix is too singular");
                }
                result[j * n + j] = q;
            } else if (i > j) {
                result[i * n + j] = s / result[j * n + j];
            } else {
                result[i * n + j] = T(0);
            }
        }
    }
}

void load_data() {
    std::string tmp;

    csv_file_t corr_file("in/Correlation.csv");

    float corr[3 * 3];
    corr_file >> tmp >> corr[0] >> corr[1] >> corr[2];
    corr_file >> tmp >> corr[3] >> corr[4] >> corr[5];
    corr_file >> tmp >> corr[6] >> corr[7] >> corr[8];
    cholesky_lower(corr, ch_lower, 3);

    std::unordered_map<std::string, float> rating_to_pd;

    csv_file_t pd_file("in/PD_Table.csv");

    std::string rating;
    float pd;
    while (pd_file >> rating >> pd) {
        rating_to_pd[rating] = pd;
    }

    csv_file_t pf_file("in/Portfolio.csv"), fl_file("in/Factor_Loadings.csv");

    loans.reserve(40'000);

    boost::math::normal_distribution<float> dist;

    loan_t loan;
    std::string region;
    float alpha[3];

    while ((pf_file >> tmp >> region >> rating >> loan.ead >> loan.lgd) &&
        (fl_file >> tmp >> alpha[0] >> alpha[1] >> alpha[2])) {
        if ("CH" == region) {
            loan.region = 0;
        } else if ("EU" == region) {
            loan.region = 1;
        } else {
            loan.region = 2;
        }

        loan.alpha = alpha[loan.region];
        loan.pd = rating_to_pd[rating];
        loan.gamma = std::sqrt(1 - loan.alpha * loan.alpha);
        loan.treshold = boost::math::quantile(dist, loan.pd);
        loans.push_back(loan);
    }
}

extern void create_histogram(std::vector<double> &losses, const std::wstring &sub);

int main() {
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    const size_t num_iter = 100'000;

    cl_int err;

    init_cl();
    create_program();
    load_data();

    buffers.loans = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(loan_t) * loans.size(), loans.data(), &err);
    _if_err_ret("clCreateBuffer() failed");

    buffers.losses = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(double) * num_iter, NULL, &err);
    _if_err_ret("clCreateBuffer() failed");

    mat3x3_t lower_mat;

    for (unsigned i = 0; i < 3; i++) {
        for (unsigned j = 0; j < 3; j++) {
            lower_mat[i].s[j] = ch_lower[i * 3 + j];
        }
    }

    buffers.lower = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(lower_mat), lower_mat, &err);
    _if_err_ret("clCreateBuffer() failed");

    sim_ker = clCreateKernel(program, "simulation", &err);
    _if_err_ret("Failed to create simulation() kernel");

    err = clSetKernelArg(sim_ker, 0, sizeof(cl_mem), &buffers.loans);
    _if_err_ret("simulation!clSetKernelArg(0) failed");

    err = clSetKernelArg(sim_ker, 1, sizeof(cl_mem), &buffers.losses);
    _if_err_ret("simulation!clSetKernelArg(1) failed");

    err = clSetKernelArg(sim_ker, 2, sizeof(cl_mem), &buffers.lower);
    _if_err_ret("simulation!clSetKernelArg(2) failed");

    cl_uint num_loans = loans.size();
    err = clSetKernelArg(sim_ker, 3, sizeof(cl_uint), &num_loans);
    _if_err_ret("simulation!clSetKernelArg(3) failed");

    std::random_device rd;
    std::mt19937 mt(rd());

    cl_uint seed = mt();
    err = clSetKernelArg(sim_ker, 4, sizeof(cl_uint), &seed);
    _if_err_ret("simulation!clSetKernelArg(4) failed");

    // ==== Timer start ====

    _time_t start_time = _clock_t::now();

    err = clEnqueueNDRangeKernel(queue, sim_ker, 1, NULL, &num_iter, NULL, 0, NULL, NULL);
    _if_err_ret("simulation!clEnqueueNDRangeKernel() failed");

    clFlush(queue);
    clFinish(queue);

    // ==== Timer end ====

    std::wostringstream ss;
    ss << "Simulation time: " << std::chrono::duration_cast<std::chrono::milliseconds>(
        _clock_t::now() - start_time).count() << " ms" << std::endl;

    char name[256];
    size_t name_len;

    clGetDeviceInfo(device, CL_DEVICE_NAME, 255, name, &name_len);
    ss << "Device identifier: " << std::wstring(&name[0], &name[name_len]) << std::endl;

    std::vector<double> losses(num_iter);
    clEnqueueReadBuffer(queue, buffers.losses, CL_TRUE, 0, num_iter * sizeof(double),
        losses.data(), 0, NULL, NULL);

    /*std::ofstream out("results_raw.csv");
    out << "Iteration,Loss" << std::endl;
    out << std::fixed << std::setprecision(0);

    for (unsigned i = 0; i < num_iter; i++) {
        out << i << "," << losses[i] << std::endl;
    }*/

    destroy_cl();

    create_histogram(losses, ss.str());

    return 0;
}
