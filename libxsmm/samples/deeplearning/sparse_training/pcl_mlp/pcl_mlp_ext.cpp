
#include <torch/extension.h>
//#include <torch/csrc/autograd/record_function.h>
#include <ATen/record_function.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>

//#include "time.h"
//#include <stdio.h>
#include <vector>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#pragma message "Using OpenMP"
#else
#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#endif
#include <libxsmm.h>

#define CHKERR_LIBXSMM_DNN(A) { const int chkerr_libxsmm_dnn_ = A; if (LIBXSMM_DNN_SUCCESS != chkerr_libxsmm_dnn_) { \
  fprintf(stderr, "%s\n", libxsmm_dnn_get_error(chkerr_libxsmm_dnn_)); global_status = chkerr_libxsmm_dnn_; } \
}

// To measure time 
#include <chrono>

#if 1
# define PRINT_LAYOUT(DESC, LAYOUT, PT_TENSOR) print_layout(DESC, LAYOUT, PT_TENSOR)
#else
# define PRINT_LAYOUT(DESC, LAYOUT, PT_TENSOR)
#endif

/* Helper functions for sparse matrix */
void BlockSpMatStep1(int K, int C, int KB, int CB, unsigned int *colptr,
        unsigned int *rowidx, unsigned int *b_colptr[],
        int *nnzb) {
    int num_blocks = K / KB * C / CB;
    int blk_idx, i, k;

    int n_em = 0;

    for (blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
        nnzb[blk_idx] = 0;
        for (i = 0; i <= KB; ++i) {
            b_colptr[blk_idx][i] = 0;
        }
    }
    for (k = 0; k < K; ++k) {
        int k_blk_idx = k / KB;
        int k_blk_offset = k % KB;
        unsigned colstart = colptr[k];
        unsigned colend = colptr[k + 1];
        for (i = colstart; i < (int)colend; ++i) {
            int c = rowidx[i];
            int c_blk_idx = c / CB;
            blk_idx = k_blk_idx * C / CB + c_blk_idx;
            nnzb[blk_idx]++;
            b_colptr[blk_idx][k_blk_offset + 1]++;
            n_em++;
        }
    }
    for (blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
        for (i = 0; i < KB; ++i) {
            b_colptr[blk_idx][i + 1] += b_colptr[blk_idx][i];
        }
    }
}

void BlockSpMatStep2(int K, int C, int KB, int CB, unsigned int *colptr,
        unsigned int *rowidx, float *values,
        unsigned int *b_colptr[], unsigned int *b_rowidx[],
        float *b_values[]) {
    int num_blocks = K / KB * C / CB;
    int blk_idx, k, i;

    int n_em = 0;
    for (k = 0; k < K; ++k) {
        int k_blk_idx = k / KB;
        int k_blk_offset = k % KB;
        unsigned colstart = colptr[k];
        unsigned colend = colptr[k + 1];
        for (i = colstart; i < (int)colend; ++i) {
            int c = rowidx[i];
            int c_blk_idx = c / CB;
            int c_blk_offset = c % CB;
            n_em++;
            blk_idx = k_blk_idx * C / CB + c_blk_idx;
            b_rowidx[blk_idx][b_colptr[blk_idx][k_blk_offset]] = c_blk_offset;
            b_values[blk_idx][b_colptr[blk_idx][k_blk_offset]] = values[i];
            b_colptr[blk_idx][k_blk_offset]++;
        }
    }

    for (blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
        for (i = KB; i > 0; --i) {
            b_colptr[blk_idx][i] = b_colptr[blk_idx][i - 1];
        }
        b_colptr[blk_idx][0] = 0;
    }
}


const at::ScalarType dt_map[] = {at::kDouble, at::kFloat, at::kBFloat16, at::kInt, at::kShort, at::kChar, at::kByte/*"UNK"*/};
void print_layout(char *desc, libxsmm_dnn_tensor_datalayout *layout, at::Tensor pt_tensor) {
  char *dim_name[] = {"N", "H", "W", "C", "K", "R", "S", "X", "RLM", "RLK", "RLN"};
  char *xsmm_dtypes[] = {"F64", "F32", "BF16", "I32", "I16", "I8", "UNK"};
  int i;
  //return;
  auto ndims = layout->num_dims;
  bool check = true;
  check = check && pt_tensor.dim() == layout->num_dims;
  for(i = 0; i < ndims; i++) {
    check = check && pt_tensor.size(i) == layout->dim_size[ndims - i - 1];
  }
  check = check && pt_tensor.scalar_type() == dt_map[layout->datatype];
  check = check && pt_tensor.is_contiguous();

  if(!check) {
    std::stringstream ss;
    ss << desc << ": F:" << layout->format << " TT: " << layout->tensor_type << " DT: " << xsmm_dtypes[layout->datatype] << " [";
    //printf("%s: F:%d TT: %d DT: %d [", desc, layout->format, layout->tensor_type, layout->datatype);
    for(i = layout->num_dims - 1; i >= 0; i--) {
      //printf("%s:%d%s", dim_name[layout->dim_type[i]], layout->dim_size[i], i == 0 ? "" : ", ");
      ss << dim_name[layout->dim_type[i]] << ":" << layout->dim_size[i] << (i == 0 ? "" : ", ");
    }
    //printf("]\n");
    ss << "]\n";
    ss << "  PyTorch Tensor type: " << pt_tensor.scalar_type() << " size: " << pt_tensor.sizes() << " cont: " << pt_tensor.is_contiguous() << std::endl;
    std::cout << ss.str();
  }
  if(!check) {
    //exit(1);
  }

}

at::Tensor mlp_set_relu_mask(void *libxsmm_handle_)
{
  libxsmm_dnn_fullyconnected *handle = (libxsmm_dnn_fullyconnected*)libxsmm_handle_;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_tensor_datalayout* layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout(handle, LIBXSMM_DNN_RELU_MASK, &status); CHKERR_LIBXSMM_DNN( status );
  std::vector<long> dim_size;
  for(int i = layout->num_dims - 1; i >= 0; i--) {
    dim_size.push_back(layout->dim_size[i]);
  }
  at::Tensor pt_tensor = at::empty(dim_size, torch::TensorOptions().dtype(dt_map[layout->datatype]));
  void *ptr = pt_tensor.data_ptr();
  PRINT_LAYOUT("ReLU_Mask", layout, pt_tensor);
  libxsmm_dnn_tensor* tensor = libxsmm_dnn_link_tensor( layout, ptr, &status ); CHKERR_LIBXSMM_DNN( status );
  libxsmm_dnn_destroy_tensor_datalayout( layout );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( handle, tensor, LIBXSMM_DNN_RELU_MASK ) );
  return pt_tensor;
}

void libxsmm_dnn_fullyconnected_set_ptr_helper(libxsmm_dnn_fullyconnected *handle, const libxsmm_dnn_tensor_type type, at::Tensor pt_tensor, char *desc)
{
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status;
  void *ptr;
  if(pt_tensor.scalar_type() == at::kFloat) ptr = (void*)pt_tensor.data_ptr<float>();
  else if(pt_tensor.scalar_type() == at::kBFloat16) ptr = (void*)pt_tensor.data_ptr<at::BFloat16>();
  else ptr = NULL;
  libxsmm_dnn_tensor* tensor = libxsmm_dnn_fullyconnected_get_tensor(handle, type, &status); CHKERR_LIBXSMM_DNN( status );
  //libxsmm_dnn_tensor_datalayout* layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout(handle, type, &status); CHKERR_LIBXSMM_DNN( status );
  //PRINT_LAYOUT(desc, layout, pt_tensor);
  if(!tensor) {
    libxsmm_dnn_tensor_datalayout* layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout(handle, type, &status); CHKERR_LIBXSMM_DNN( status );
    PRINT_LAYOUT(desc, layout, pt_tensor);
    tensor = libxsmm_dnn_link_tensor( layout, ptr, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( handle, tensor, type ) );
  } else {
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_set_tensor_data_ptr(tensor, ptr) );
  }
  //libxsmm_dnn_destroy_tensor_datalayout( layout );
}

void libxsmm_dnn_fullyconnected_release_tensor_helper(libxsmm_dnn_fullyconnected *handle, const libxsmm_dnn_tensor_type type)
{
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_tensor* tensor = libxsmm_dnn_fullyconnected_get_tensor(handle, type, &status); CHKERR_LIBXSMM_DNN( status );
  if(tensor) {
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor(tensor) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_tensor( handle, type ) );
  }
}

void *create_handle(int N, int C, int K, int bn, int bc, int bk, int dtype, int fuse_bias, int act_type)
{
  libxsmm_dnn_fullyconnected_desc fullyconnected_desc;
  libxsmm_dnn_fullyconnected* libxsmm_handle;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status;
  fullyconnected_desc.N = N;
  fullyconnected_desc.C = C;
  fullyconnected_desc.K = K;
  fullyconnected_desc.bn = bn;
  fullyconnected_desc.bk = bk;
  fullyconnected_desc.bc = bc;
  fullyconnected_desc.threads = omp_get_max_threads();
  fullyconnected_desc.datatype_in = (dtype == 1 ? LIBXSMM_DNN_DATATYPE_F32 : LIBXSMM_DNN_DATATYPE_BF16);
  fullyconnected_desc.datatype_out = (dtype == 1 ? LIBXSMM_DNN_DATATYPE_F32 : LIBXSMM_DNN_DATATYPE_BF16);
  fullyconnected_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_NCPACKED;
  fullyconnected_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_CKPACKED;
  fullyconnected_desc.fuse_ops = LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE;
  if(fuse_bias == 1) {
    if(act_type == 0)
      fullyconnected_desc.fuse_ops = LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS;
    else if(act_type == 1)
      fullyconnected_desc.fuse_ops = LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_RELU;
    else if(act_type == 2)
      fullyconnected_desc.fuse_ops = LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_SIGMOID;
    else
      { printf("Unknown activation type (%d)\n", act_type); exit(1); }
  } else {
     if(act_type == 0)
      fullyconnected_desc.fuse_ops = LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE;
    else if(act_type == 1)
      fullyconnected_desc.fuse_ops = LIBXSMM_DNN_FULLYCONNECTED_FUSE_RELU;
    else if(act_type == 2)
      fullyconnected_desc.fuse_ops = LIBXSMM_DNN_FULLYCONNECTED_FUSE_SIGMOID;
    else
      { printf("Unknown activation type (%d)\n", act_type); exit(1); }
 }

  libxsmm_handle = libxsmm_dnn_create_fullyconnected( fullyconnected_desc, &status );
  CHKERR_LIBXSMM_DNN( status );
  auto scratch_size = libxsmm_dnn_fullyconnected_get_scratch_size( libxsmm_handle, &status );
  CHKERR_LIBXSMM_DNN( status );
  auto scratch = libxsmm_aligned_malloc( scratch_size, 2097152 );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_scratch( libxsmm_handle, scratch ) );
  //std::cout << "Create Handle = " << libxsmm_handle << std::endl;
  return (void *)libxsmm_handle;
}

template<typename scalar_t>
at:: Tensor bias_add_grad(at::Tensor &grad_out)
{
  auto nbn = grad_out.size(0);
  auto nbk = grad_out.size(1);
  auto bn = grad_out.size(2);
  auto bk = grad_out.size(3);
  auto grad_bias = at::empty({nbk * bk}, grad_out.options());
  auto gbptr = grad_bias.data_ptr<scalar_t>();
  auto goptr = grad_out.data_ptr<scalar_t>();
#ifdef _OPENMP
  auto nthr = omp_get_max_threads();
  if(nthr > nbk) nthr = nbk;
#pragma omp parallel for num_threads(nthr)
#endif
  for(auto ik = 0; ik < nbk; ik++) {
    auto gb = &gbptr[ik * bk];
    for(auto jk = 0; jk < bk; jk++) {
      gb[jk] = 0;
      for(auto in = 0; in < nbn; in++) {
        auto go = &goptr[(((in * nbk) + ik) * bn + jk) * bk];
        for(auto jn = 0; jn < bn; jn++) {
          gb[jk] += go[jn];
        }
      }
    }
  }
  return grad_bias;
}

/*
 * input 2D
 **/
at::Tensor mlp_sparse_forward(
    void *libxsmm_handle_,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias)
{
  using namespace std::chrono;
  FILE *fp;
  char fname[] = "timed_run_frw_improved_[layer_count].csv";
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  //typedef std::chrono::high_resolution_clock Clock;
  auto nbn = input.size(0);
  auto nbc = input.size(1);
  auto bn = input.size(2);
  auto bc = input.size(3);

  auto nbk = weight.size(0);
  auto bk = weight.size(3);

  // A: NxC, B: CxK, O: NxK
  auto N = nbn * bn;
  auto C = nbc * bc;
  auto K = nbk * bk;

  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> total_time = duration_cast<duration<double>>(t2 - t1);
  printf("\n\nmlp_sparse_forward\n\n");
  printf("Section 1 time: %lf\n", total_time);
  fp = fopen(fname, "a");
  fprintf(fp, "1,%lf\n", total_time);
  fclose(fp);
  t1 = high_resolution_clock::now();
  //printf("input shape: (%d, %d)\n", N, C);
  //printf("weight shape: (%d, %d)\n", C, K);
  //printf("output shape: (%d, %d)\n", N, K);
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_fullyconnected* libxsmm_handle = (libxsmm_dnn_fullyconnected*)libxsmm_handle_;

  int NB = bn;
  int CB = bc;
  int KB = bk; // set K block size the same as CB

  // int nb = N/NB;
  int nb = 16;

  libxsmm_gemm_prefetch_type prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  int flags = LIBXSMM_GEMM_FLAGS('N', 'N');

  float *l_A = (float *)libxsmm_aligned_malloc(sizeof(float) * N * C, 64);
  float *l_B = (float *)libxsmm_aligned_malloc(sizeof(float) * C * K, 64);
  float *l_C = (float *)libxsmm_aligned_malloc(sizeof(float) * N * K, 64);


  int l_n, l_c, l_nn, l_cc, l_nnn;
  LIBXSMM_VLA_DECL(5, float, l_p_A, l_A, C / CB, NB / nb, CB, nb);
  LIBXSMM_VLA_DECL(5, float, l_p_C, l_C, K / KB, NB / nb, KB, nb);
  // LIBXSMM_VLA_DECL(5, float, l_p_C_gold, l_C_gold, K / KB, NB / nb, KB, 16);

  t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 2 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "2,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);

  // auto input_ = input.permute({0, 2, 1, 3}).reshape({N, C});

  t1 = high_resolution_clock::now();
  // Converting A to 5 dim

  //██████████████████████
  //###FLAT POINTER###
  //██████████████████████
  int aa = 0;
  for (l_n = 0; l_n < N / NB; ++l_n) {
      for (l_c = 0; l_c < C / CB; ++l_c) {
          for (l_nn = 0; l_nn < NB / nb; ++l_nn) {
              for (l_cc = 0; l_cc < CB; ++l_cc) {
                  // Flat Pointer
                  float * flat_ptr = (float*)input[l_n][l_c].data_ptr();
                  for (l_nnn = 0; l_nnn < nb; ++l_nnn) {
                      int i = l_nn * nb + l_nnn;
                      int j = l_cc;

                      // printf("accessing index i, j: (%d, %d, %d, %d)\n", l_n, l_c, i, j);
                      LIBXSMM_VLA_ACCESS(5, l_p_A, l_n, l_c, l_nn, l_cc,
                              l_nnn, C / CB, NB / nb, CB, nb) =

                         //(float)input[l_n][l_c][i][j].item().to<float>();
                         flat_ptr[i*NB+j];
                         
                      // Test Passed
                      // if (flat_ptr[i*NB+j] != (float)input[l_n][l_c][i][j].item().to<float>())
                      // {
                      //   printf("ERROR! STOP! %lf - %lf\n", flat_ptr[i*NB+j], (float)input[l_n][l_c][i][j].item().to<float>());
                      // }
                         // (float)input_[l_n * NB + i][l_c * CB + j].item().to<float>();
                    aa++;
                  }
              }
          }
      }
  }
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 3 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "3,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);

  // printf("\n\n\nA created, number of elements: %d \n\n\n", aa);

  t1 = high_resolution_clock::now();
  int cc = 0;
  int l_k, l_kk;
  /* touch C */
  for (l_n = 0; l_n < N / NB; ++l_n) {
      for (l_k = 0; l_k < K / KB; ++l_k) {
          for (l_nn = 0; l_nn < NB / nb; ++l_nn) {
              for (l_kk = 0; l_kk < KB; ++l_kk) {
                  for (l_nnn = 0; l_nnn < nb; ++l_nnn) {
                      cc++;
                      LIBXSMM_VLA_ACCESS(5, l_p_C, l_n, l_k, l_nn, l_kk,
                              l_nnn, K / KB, NB / nb, KB, nb) =
                          0.0f;
                  }
              }
          }
      }
  }
  // printf("\n\n\nC created, number of elements: %d \n\n\n", cc);
  // Create sparse B

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 4 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "4,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

 /* touch dense B and init sparse B*/
  int nnz = 0;
  float tmp = 0.0;
  unsigned int *colptr = (unsigned int *)libxsmm_aligned_malloc((K + 1) * sizeof(unsigned int), 64);
  colptr[0] = 0;

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 5 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "5,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  // printf("weight shape: (%d, %d)\n", C, K);
  // printf("C: %d K: %d bk: %d bc: %d\n", C, K, bk, bc);
  // int nbk_max = 0;
  // int nbc_max = 0;
  // int bk_max = 0;
  // int bc_max = 0;

  //██████████████████████
  //###FLAT POINTER###
  //██████████████████████
  for (l_k = 0; l_k < K; l_k++) {
      colptr[l_k + 1] = 0;
      float * flat_ptr = (float*)weight.data_ptr();
      for (l_c = 0; l_c < C; l_c++) {
          //██████████████████████████████████████████████████████████████████
          // Init once not every round?
          //██████████████████████████████████████████████████████████████████
          int depth = C/bc;
          int nbk_idx = l_k / bk;
          int nbc_idx = l_c / bc;
          int bk_idx = l_k % bk;
          int bc_idx = l_c % bc;
          // printf("nbk_idx: %d nbc_idx: %d bk_idx: %d bc_idx: %d\n", nbk_idx, nbc_idx, bk_idx, bc_idx);
          // if (nbk_idx > nbk_max)
          // { nbk_max = nbk_idx;}
          // if (nbc_idx > nbc_max)
          // { nbc_max = nbc_idx;}
          // if (bk_idx > bk_max)
          // { bk_max = bk_idx;}
          // if (bc_idx > bc_max)
          // { bc_max = bc_idx;}

          // tmp = (float)weight[l_k][l_c].item().to<float>();
          //tmp = (float)weight[nbk_idx][nbc_idx][bc_idx][bk_idx].item().to<float>();
          tmp = flat_ptr[nbk_idx * (depth*bc*bk) + nbc_idx * (bc*bk) + bc_idx * bk + bk_idx];
          
          // Test Passed
          // if (flat_ptr[nbk_idx * (depth*bc*bk) + nbc_idx * (bc*bk) + bc_idx * bk + bk_idx] !=
          //     (float)weight[nbk_idx][nbc_idx][bc_idx][bk_idx].item().to<float>())
          //     {printf("DONT MATCH!\n");}
          // printf("%lf   %lf\n", flat_ptr[nbk_idx * (depth*bc*bk) + nbc_idx * (bc*bk) + bc_idx * bk + bk_idx],
          // (float)weight[nbk_idx][nbc_idx][bc_idx][bk_idx].item().to<float>());
          if (tmp == 0.0) {
            // pass
          }
          else {
              nnz++;
              colptr[l_k + 1]++;
          }
          l_B[l_k * C + l_c] = (float)tmp;
      }
  }
  //printf("nbk_idx: %d nbc_idx: %d bk_idx: %d bc_idx: %d\n", nbk_max, nbc_max, bk_max, bc_max);

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 6 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "6,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  // Seems to work correctly
  // printf("sparsity of B: %f\n", (float)(nnz)/(float)(K*C));

  for (l_k = 0; l_k < K; l_k++) {
      colptr[l_k + 1] += colptr[l_k];
  }
  unsigned int *rowidx =
      (unsigned int *)libxsmm_aligned_malloc(nnz * sizeof(unsigned int), 64);
  float *values = (float *)libxsmm_aligned_malloc(nnz * sizeof(float), 64);
  for (l_k = 0; l_k < K; l_k++) {
      int offset = colptr[l_k];
      for (l_c = 0; l_c < C; l_c++) {
          if (l_B[l_k * C + l_c] != 0) {
              rowidx[offset] = l_c;
              values[offset] = l_B[l_k * C + l_c];
              offset++;
          }
      }
  }

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 7 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "7,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  unsigned num_k_blocks = K / KB;
  unsigned num_c_blocks = C / CB;
  int num_blocks = num_k_blocks * num_c_blocks;
  unsigned int **b_colptr = (unsigned int **)libxsmm_aligned_malloc(
          num_blocks * sizeof(unsigned int *), 64);
  unsigned int **b_rowidx = (unsigned int **)libxsmm_aligned_malloc(
          num_blocks * sizeof(unsigned int *), 64);
  float **b_values =
      (float **)libxsmm_aligned_malloc(num_blocks * sizeof(float *), 64);
  int *nnzb = (int *)libxsmm_aligned_malloc(num_blocks * sizeof(int), 64);
  int blk_idx;
  for (blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
      b_colptr[blk_idx] = (unsigned int *)libxsmm_aligned_malloc(
              (KB + 1) * sizeof(unsigned int), 64);
  }

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 8 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "8,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();
  
  BlockSpMatStep1(K, C, KB, CB, colptr, rowidx, b_colptr, nnzb);
  
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 9 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "9,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();
  
  for (blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
      b_rowidx[blk_idx] = (unsigned int *)libxsmm_aligned_malloc(
              nnzb[blk_idx] * sizeof(unsigned int), 64);
      b_values[blk_idx] =
          (float *)libxsmm_aligned_malloc(nnzb[blk_idx] * sizeof(float), 64);
  }
  
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 10 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "10,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();
  
  BlockSpMatStep2(K, C, KB, CB, colptr, rowidx, values, b_colptr, b_rowidx,
          b_values);

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 11 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "11,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  /* Create FWD kernels */
  float alpha = 1.0;
  float beta = 1.0;
  libxsmm_descriptor_blob l_xgemm_blob;
  libxsmm_gemm_descriptor **l_xgemm_desc =
      (libxsmm_gemm_descriptor **)libxsmm_aligned_malloc(
              num_blocks * sizeof(libxsmm_gemm_descriptor *), 64);
  libxsmm_smmfunction *mykernel =
      (libxsmm_smmfunction *)libxsmm_aligned_malloc(
              num_blocks * sizeof(libxsmm_smmfunction), 64);
  for (blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
      l_xgemm_desc[blk_idx] = libxsmm_gemm_descriptor_dinit(
              &l_xgemm_blob, LIBXSMM_DATATYPE(float), NB / nb, KB, CB, CB,
              0, KB, alpha, beta, flags, prefetch);
      mykernel[blk_idx] =
          libxsmm_create_packed_spxgemm_csc(l_xgemm_desc[blk_idx], nb, b_colptr[blk_idx],
                  b_rowidx[blk_idx],
                  (const void *)b_values[blk_idx]).smm;
  }
  
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  //total_time += time_span;
  printf("Section 12 time: %lf (kernel creation)\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "12,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();
  // Execute kernels amoung threads
  int k, n, c;
#ifdef _OPENMP
#   pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(k,n,c)
#endif
  for (k = 0; k < K / KB; ++k) {
      for (n = 0; n < N / NB; ++n) {
          for (c = 0; c < C / CB; ++c) {
              mykernel[k * C / CB + c](
                &(l_A[(n * C / CB + c) * CB * NB]),
                b_values[k * C / CB + c],
                &(l_C[(n * K / KB + k) * NB * KB]));
          }
      }
  }

  auto _output = at::empty({N, K}, input.options());

t2 = high_resolution_clock::now();
time_span = duration_cast<duration<double>>(t2 - t1);
//total_time += time_span;
printf("Section 13 time: %lf (kernel execution)\n", time_span);
fp = fopen(fname, "a");
  fprintf(fp, "13,%lf\n", time_span);
  fclose(fp);
total_time = total_time + duration_cast<duration<double>>(t2 - t1);

t1 = high_resolution_clock::now();

int max_j = 0;
int max_i = 0;
//* Why doesn't this work? Doesnt work because no privates!!! XD --> works with private(l_n, l_k, l_nn)
#ifdef _OPENMP
#   pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(l_n, l_k, l_nn)
#endif
  for (l_n = 0; l_n < N / NB; ++l_n) {
      for (l_k = 0; l_k < K / KB; ++l_k) {
          int offset = (l_n * K / KB + l_k) * NB * KB;

          // Block Grid
          int i_blk_offset = l_k * KB;
          int j_blk_offset = l_n * NB;

          float * flat_pointer = (float*)_output.data_ptr();
          for (l_nn = 0; l_nn < NB * KB; ++l_nn) {
              // Subblock grid
              // blocks are sized 4x64x16
              // First we trying 16x64 and 4 of those

              int i = (l_nn % (NB * nb)) / nb;
              int j_small_offset = l_nn % nb;
              int j_big_offset = l_nn / (nb * NB);;
              int j = j_small_offset + (j_big_offset * nb);

              //if (i_blk_offset + i > max_i) { max_i = i_blk_offset + i;}
		          //if (j_blk_offset + j > max_j) { max_j = j_blk_offset + j;}

              //_output.index_put_({j_blk_offset + j, i_blk_offset + i}, l_C[offset + l_nn]);
              flat_pointer[(j_blk_offset+j)*K + i_blk_offset+i] = l_C[offset + l_nn];
              //Passed Test
              //if (flat_pointer[(j_blk_offset+j)*K + i_blk_offset+i] != l_C[offset + l_nn])
              //{printf("ERROR: %lf - %lf\n", flat_pointer[(j_blk_offset+j)*K + i_blk_offset+i], l_C[offset + l_nn]);}
              //getchar();
          }
      }
  }

  // printf("nb: %d\n", nb);
  // printf("N: %d NB: %d N/NB: %d\n", N, NB, N/NB);
  // printf("K: %d KB: %d K/KB: %d\n", K, KB, K/KB);
  // printf("i: %d j: %d\n", max_i, max_j);
  // getchar();

  auto output = _output.reshape({N, K});

t2 = high_resolution_clock::now();
time_span = duration_cast<duration<double>>(t2 - t1);
printf("Section 14 time: %lf\n", time_span);
fp = fopen(fname, "a");
fprintf(fp, "14,%lf\n", time_span);
fclose(fp);
total_time = total_time + duration_cast<duration<double>>(t2 - t1);
t1 = high_resolution_clock::now();

{
RECORD_FUNCTION("xsmm_mm_fwd", std::vector<c10::IValue>({input, weight}), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
{
  int tid = omp_get_thread_num();
  // CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid) );
}
}
//  output += bias.reshape({nbk, 1, bk});
    /* clean up */
    libxsmm_free(l_A);
    libxsmm_free(l_B);
    libxsmm_free(l_C);
    for (blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
        libxsmm_free(b_values[blk_idx]);
        libxsmm_free(b_colptr[blk_idx]);
        libxsmm_free(b_rowidx[blk_idx]);
    }
    libxsmm_free(b_values);
    libxsmm_free(b_colptr);
    libxsmm_free(b_rowidx);

  output += bias;

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  //total_time += time_span;
  printf("Section 15 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "15,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  printf("Total time: %lf\n", total_time);
  return output;

}

at::Tensor mlp_forward(void *libxsmm_handle_, torch::Tensor input, torch::Tensor weight, torch::Tensor bias)
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_fullyconnected* libxsmm_handle = (libxsmm_dnn_fullyconnected*)libxsmm_handle_;
  auto nbn = input.size(0);
  //auto nbc = input.size(1);
  auto bn = input.size(2);
  //auto bc = input.size(3);
  auto nbk = weight.size(0);
  auto bk = weight.size(3);

  printf("input shape: (%d, %d)\n", input.size(0), input.size(1));
  printf("weight shape: (%d, %d)\n", weight.size(0), weight.size(1));
  auto output = at::empty({nbn, nbk, bn, bk}, input.options());
  //std::cout << "FWD Handle = " << libxsmm_handle_ << std::endl;
  libxsmm_dnn_fullyconnected_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, input, "Input");
  libxsmm_dnn_fullyconnected_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER, weight, "Weight");
  libxsmm_dnn_fullyconnected_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_BIAS, bias.view({nbk, bk}), "Bias");
  libxsmm_dnn_fullyconnected_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, output, "Output");
{
RECORD_FUNCTION("xsmm_mm_fwd", std::vector<c10::IValue>({input, weight}), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
{
  int tid = omp_get_thread_num();
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid) );
}
}
//  output += bias.reshape({nbk, 1, bk});
  auto output_ = output.permute({0, 2, 1, 3}).reshape({nbn*bn, nbk*bk});
  return output_;

}

/**
 * Sparse backward function consists of two major parts:
 *   - The backward pass, where we compute grad_input
 *     - dO x W.T = dI (NxK) x (KxC) = (NxC)
 *   - The update pass, where we compute grad_weight
 *     - I.T x dO = dW (CxN) x (NxK) = (CxK)
 */
at::Tensor mlp_sparse_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight)
{
  using namespace std::chrono;
  FILE *fp;
  char fname[] = "timed_run_bkw_improved_[layer_count].csv";
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  libxsmm_dnn_err_t global_status;
  auto nbn = input.size(0);
  auto nbc = input.size(1);
  auto bn = input.size(2);
  auto bc = input.size(3);

  /* weight is packages in (nbk, nbc, bc, bk) */
  auto nbk = weight.size(0);
  auto bk = weight.size(3);

  auto N = nbn * bn;
  auto C = nbc * bc;
  auto K = nbk * bk;;

  int NB = bn;
  int CB = bc;
  int KB = bk;

  int nb = 16; // or 32

  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> total_time = duration_cast<duration<double>>(t2 - t1);
  printf("\n\nmlp_sparse_backward\n\n");
  printf("Section 1 time: %lf\n", total_time);
  fp = fopen(fname, "a");
  fprintf(fp, "1,%lf\n", total_time);
  fclose(fp);
  t1 = high_resolution_clock::now();
  //printf("input shape: (%d, %d)\n", N, C);
  //printf("weight shape: (%d, %d)\n", C, K);
  //printf("grad_output shape: (%d, %d)\n", N, K);

  /* Declare return variables */
  auto grad_input = at::empty(input.sizes(), input.options());

  /* Used throughout this function */
  libxsmm_gemm_prefetch_type prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  int flags = LIBXSMM_GEMM_FLAGS('N', 'N');

  // In terms of A * B(sparse) = C
  // l_A: grad_output, l_B: weight.T, l_C: grad_input
  float *l_A = (float *)libxsmm_aligned_malloc(sizeof(float) * N * K, 64);
  float *l_B = (float *)libxsmm_aligned_malloc(sizeof(float) * K * C, 64);
  float *l_C = (float *)libxsmm_aligned_malloc(sizeof(float) * N * C, 64);

  int l_n, l_k, l_nn, l_kk, l_nnn;
  LIBXSMM_VLA_DECL(5, float, l_p_A, l_A, K / KB, NB / nb, KB, nb);
  LIBXSMM_VLA_DECL(5, float, l_p_C, l_C, C / CB, NB / nb, CB, nb);

{
RECORD_FUNCTION("xsmm_mm_bwdupd", std::vector<c10::IValue>({grad_output, weight}), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
}

  t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 2 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "2,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();
  // Converting A to 5 dim
  int i_max = 0;
  int j_max = 0;
  int aa = 0;
  //██████████████████████
  //###FLAT POINTER###
  //██████████████████████
  for (l_n = 0; l_n < N / NB; ++l_n) {
      for (l_k = 0; l_k < K / KB; ++l_k) {
          for (l_nn = 0; l_nn < NB / nb; ++l_nn) {
              for (l_kk = 0; l_kk < KB; ++l_kk) {
                  float * flat_ptr = (float*)grad_output.data_ptr();
                  for (l_nnn = 0; l_nnn < nb; ++l_nnn) {
                      int depth = ((K/KB)-1) * KB + KB;
                      int i = l_n * NB + l_nn * nb + l_nnn;
                      //printf("l_n: %d NB: %d l_nn: %d nb: %d l_nnn: %d\n", l_n, NB, l_nn, nb, l_nnn);
                      int j = l_k * KB + l_kk;
                      // printf("l_k: %d KB: %d l_kk: %d\n", l_k, KB, l_kk);

                      // if (i > i_max) {i_max = i;}
                      // if (j > j_max) {j_max = j;}
                      // printf("accessing index i, j: (%d, %d, %d, %d)\n", l_n, l_c, i, j);
                      //printf("accessing through - i: %d j: %d\n", i, j);
                      
                      // Test Passed
                      // if ((float)grad_output[i][j].item().to<float>() != flat_ptr[i*depth + j])
                      // {
                      //   printf("TERMINAL ERROR!\n");
                      //   printf("%d\n", depth);
                      //   printf("%lf vs %lf\n", (float)grad_output[i][j].item().to<float>(), flat_ptr[i*depth + j]);
                      // }
                      LIBXSMM_VLA_ACCESS(5, l_p_A, l_n, l_k, l_nn, l_kk,
                              l_nnn, K / KB, NB / nb, KB, nb) =
                              //(float)grad_output[i][j].item().to<float>();
                              flat_ptr[i*depth + j];
                  }
              }
          }
      }
  }
  // getchar();
  // printf("MAX - i: %d j: %d\n", i_max, j_max);
  // printf("N: %d NB: %d N/NB: %d\n", N, NB, N/NB);
  // printf("K: %d KB: %d K/KB: %d\n", K, KB, K/KB);
  // printf("NB: %d nb: %d NB/nb: %d\n", NB, nb, NB/nb);
  // printf("KB: %d nb: %d\n", KB, nb);

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 3 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "3,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  int cc = 0;
  int l_c, l_cc;
  /* touch C */
  for (l_n = 0; l_n < N / NB; ++l_n) {
      for (l_c = 0; l_c < C / CB; ++l_c) {
          for (l_nn = 0; l_nn < NB / nb; ++l_nn) {
              for (l_cc = 0; l_cc < CB; ++l_cc) {
                  for (l_nnn = 0; l_nnn < nb; ++l_nnn) {
                      cc++;
                      LIBXSMM_VLA_ACCESS(5, l_p_C, l_n, l_c, l_nn, l_cc,
                              l_nnn, C / CB, NB / nb, CB, nb) =
                          0.0f;
                  }
              }
          }
      }
  }

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 4 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "4,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  // Create sparse B
 /* touch dense B and init sparse B*/
  int nnz = 0;
  float tmp = 0.0;
  // K rows x C columns
  auto weight_transpose = weight.permute({1, 2, 0, 3}).reshape({C, K}).transpose(0, 1);
  unsigned int *colptr = (unsigned int *)libxsmm_aligned_malloc((C + 1) * sizeof(unsigned int), 64);
  
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 5 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "5,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  colptr[0] = 0;
  float * flat_ptr_s4 = (float*)weight_transpose.data_ptr();
  //██████████████████████
  //###FLAT POINTER###
  //██████████████████████
  //printf("C: %d K: %d\n", C, K);
  for (l_c = 0; l_c < C; l_c++) {
      colptr[l_c + 1] = 0;
      for (l_k = 0; l_k < K; l_k++) {
          //tmp = (float)weight_transpose[l_k][l_c].item().to<float>();
          tmp = flat_ptr_s4[l_c*K + l_k];

          //Test Passed
          // if ((float)weight_transpose[l_k][l_c].item().to<float>() != flat_ptr_s4[l_c*K + l_k])
          // {
          //   printf("Critical ERROR!\n");
          // }

          // Did not pass, Confused. As though not transposed, just tensor approached differently
          // if ((float)weight_transpose[l_k][l_c].item().to<float>() != flat_ptr_s4[l_k*C + l_c])
          // {
          //   printf("Critical ERROR! SHOULD BE THE CORRECT ONE\n");
          // }
          if (tmp == 0.0) {}
          else {
              nnz++;
              colptr[l_c + 1]++;
          }
          l_B[l_c * K + l_k] = (float)tmp;
      }
  }
  //getchar();

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 6 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "6,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  for (l_c = 0; l_c < C; l_c++) {
      colptr[l_c + 1] += colptr[l_c];
  }

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 7 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "7,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  unsigned int *rowidx =
      (unsigned int *)libxsmm_aligned_malloc(nnz * sizeof(unsigned int), 64);

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 8 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "8,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  float *values = (float *)libxsmm_aligned_malloc(nnz * sizeof(float), 64);
  for (l_c = 0; l_c < C; l_c++) {
      int offset = colptr[l_c];
      for (l_k = 0; l_k < K; l_k++) {
          if (l_B[l_c * K + l_k] != 0) {
              rowidx[offset] = l_k;
              values[offset] = l_B[l_c * K + l_k];
              offset++;
          }
      }
  }
  unsigned num_c_blocks = C / CB;
  unsigned num_k_blocks = K / KB;

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 9 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "9,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  int num_blocks = num_k_blocks * num_c_blocks;
  unsigned int **b_colptr = (unsigned int **)libxsmm_aligned_malloc(
          num_blocks * sizeof(unsigned int *), 64);
  unsigned int **b_rowidx = (unsigned int **)libxsmm_aligned_malloc(
          num_blocks * sizeof(unsigned int *), 64);
  float **b_values =
      (float **)libxsmm_aligned_malloc(num_blocks * sizeof(float *), 64);
  int *nnzb = (int *)libxsmm_aligned_malloc(num_blocks * sizeof(int), 64);
  int blk_idx;
  for (blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
      b_colptr[blk_idx] = (unsigned int *)libxsmm_aligned_malloc(
              (CB + 1) * sizeof(unsigned int), 64);
  }

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 10 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "10,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  BlockSpMatStep1(C, K, CB, KB, colptr, rowidx, b_colptr, nnzb);
  
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 11 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "11,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  for (blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
      b_rowidx[blk_idx] = (unsigned int *)libxsmm_aligned_malloc(
              nnzb[blk_idx] * sizeof(unsigned int), 64);
      b_values[blk_idx] =
          (float *)libxsmm_aligned_malloc(nnzb[blk_idx] * sizeof(float), 64);
  }
  
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 12 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "12,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  BlockSpMatStep2(C, K, CB, KB, colptr, rowidx, values, b_colptr, b_rowidx,
          b_values);

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 13 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "13,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  /* Create FWD kernels */
  float alpha = 1.0;
  float beta = 1.0;
  libxsmm_descriptor_blob l_xgemm_blob;
  libxsmm_gemm_descriptor **l_xgemm_desc =
      (libxsmm_gemm_descriptor **)libxsmm_aligned_malloc(
              num_blocks * sizeof(libxsmm_gemm_descriptor *), 64);
  libxsmm_smmfunction *mykernel =
      (libxsmm_smmfunction *)libxsmm_aligned_malloc(
              num_blocks * sizeof(libxsmm_smmfunction), 64);
  for (blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
      l_xgemm_desc[blk_idx] = libxsmm_gemm_descriptor_dinit(
              &l_xgemm_blob, LIBXSMM_DATATYPE(float), NB / nb, CB, KB, KB,
              0, CB, alpha, beta, flags, prefetch);
      mykernel[blk_idx] =
          libxsmm_create_packed_spxgemm_csc(l_xgemm_desc[blk_idx], nb, b_colptr[blk_idx],
                  b_rowidx[blk_idx],
                  (const void *)b_values[blk_idx]).smm;
  }
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 14 time: %lf (create kernels)\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "14,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  //printf("Executing kernels\n");
  // Execute kernels amoung threads
  int k, n, c;
#ifdef _OPENMP
#   pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(k,n,c)
#endif
  for (c = 0; c < C / CB; ++c) {
      for (n = 0; n < N / NB; ++n) {
          for (k = 0; k < K / KB; ++k) {
              mykernel[c * K / KB + k](
                      &(l_A[(n * K / KB + k) * KB * NB]),
                      b_values[c * K / KB + k],
                      &(l_C[(n * C / CB + c) * NB * CB])
                      );
          }
      }
  }

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 15 time: %lf (execute kernels)\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "15,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  auto grad_input_temp = grad_input.permute({0,2,1,3}).reshape({N,C});

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 16 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "16,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  // Write results back to output
  // nbn nbc bn bc

  //int max_i = 0;
  //int max_j = 0;

  for (l_n = 0; l_n < N / NB; ++l_n) {
      for (l_c = 0; l_c < C / CB; ++l_c) {
          int offset = (l_n * C / CB + l_c) * NB * CB;

          // Block Grid
          int i_blk_offset = l_c * CB;
          int j_blk_offset = l_n * NB;

          //printf("offset: %d i_blk_offset: %d j_blk_offset: %d \n", offset, i_blk_offset, j_blk_offset);

          float * flat_pointer = (float*)grad_input_temp.data_ptr();

          for (l_nn = 0; l_nn < NB * CB; ++l_nn) {
              // Subblock grid
              // blocks are sized 4x64x16

              int i = (l_nn % (NB * nb)) / nb;
              int j_small_offset = l_nn % nb;
              int j_big_offset = l_nn / (nb * NB);;
              int j = j_small_offset + (j_big_offset * nb);

              //printf("i: %d j_small: %d j_big: %d j: %d\n", i, j_small_offset, j_blk_offset, j);

              //printf("i: %d j: %d\n", j_blk_offset + j, i_blk_offset + i);
              //if (i_blk_offset + i > max_i) { max_i = i_blk_offset + i;}
              //if (j_blk_offset + j > max_j) { max_j = j_blk_offset + j;}
              //grad_input_temp.index_put_({j_blk_offset + j, i_blk_offset + i}, l_C[offset + l_nn]);
              //grad_input_temp.index_put_({j_blk_offset + j, i_blk_offset + i}, l_C[offset + l_nn]);
              // flat_pointer[(j_blk_offset+j)*C + i_blk_offset+i] = l_C[offset + l_nn];
              // printf("%d - %d\n", (j_blk_offset+j)*C + i_blk_offset+i, offset + l_nn);
              // getchar();
              //Passsed Test
              //if (flat_pointer[(j_blk_offset+j)*C + i_blk_offset+i] != l_C[offset + l_nn])
              //{printf("ERROR: %lf - %lf\n", flat_pointer[(j_blk_offset+j)*C + i_blk_offset+i], l_C[offset + l_nn]);}
              //getchar();
          }
      }
  }
  // printf("nb: %d\n", nb);
  // printf("N: %d NB: %d N/NB: %d\n", N, NB, N/NB);
  // printf("C: %d CB: %d C/CB: %d\n", C, CB, C/CB);
  // printf("i: %d j: %d\n", max_i, max_j);
  // getchar();


  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 17 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "17,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  printf("Total time: %lf\n", total_time);

  return grad_input_temp.reshape({nbn, bn, nbc, bc}).permute({0, 2, 1, 3});

}

at::Tensor mlp_sparse_update(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight)
{
  using namespace std::chrono;
  FILE *fp;
  char fname[] = "timed_run_updt_improved_[layer_count].csv";
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  libxsmm_dnn_err_t global_status;

  auto nbn = input.size(0);
  auto nbc = input.size(1);
  auto bn = input.size(2);
  auto bc = input.size(3);

  /* weight is packages in (nbk, nbc, bc, bk) */
  auto nbk = weight.size(0);
  auto bk = weight.size(3);

  auto N = nbn * bn;
  auto C = nbc * bc;
  auto K = nbk * bk;;

  int NB = bn;
  int CB = bc;
  int KB = bk;

  int nb = 16; // or 32
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> total_time = duration_cast<duration<double>>(t2 - t1);
  printf("\n\n mlp sparse update \n\n");
  printf("Section 1 time: %lf\n", total_time);
  fp = fopen(fname, "a");
  fprintf(fp, "1,%lf\n", total_time);
  fclose(fp);
  t1 = high_resolution_clock::now();
  //printf("input shape: (%d, %d)\n", N, C);
  //printf("weight shape: (%d, %d)\n", C, K);
  //printf("grad_output shape: (%d, %d)\n", N, K);

  /* # # # # # BUG # # # # # */
  // Times test required
  /* Declare return variables */
  /*clock_t start, end;
  double cpu_time_used;
  start = clock();*/
  auto grad_weight = at::zeros(weight.sizes(), weight.options());
  //auto grad_weight = at::empty(weight.sizes(), weight.options());
  /*end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("\nInit time\n: %lf", cpu_time_used);
  FILE *fp;
  fp = fopen("bug_timing.txt", "a");
  fprintf(fp, "%lf\n", cpu_time_used);
  fclose(fp);*/
  /* # # # # # # # # # # # # */
  
  /* Used throughout this function */
  libxsmm_gemm_prefetch_type prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  int flags = LIBXSMM_GEMM_FLAGS('N', 'N');

  // update pass: input.transpose * grad_output = grad_weight
  float *l_input = (float *)libxsmm_aligned_malloc(sizeof(float) * N * C, 64);
  float *l_grad_output = (float *)libxsmm_aligned_malloc(sizeof(float) * N * K, 64);
  // float *l_B_upd = (float *)libxsmm_aligned_malloc(sizeof(float) * N * K, 64);
  // Reuse l_A from backward operation
  float *l_C = (float *)libxsmm_aligned_malloc(sizeof(float) * C * K, 64);
  float *l_C_upd = (float *)libxsmm_aligned_malloc(sizeof(float) * C * K, 64);

  LIBXSMM_VLA_DECL(5, float, l_p_input, l_input, C / CB, NB / nb, CB, nb);
  LIBXSMM_VLA_DECL(5, float, l_p_grad_output, l_grad_output, K / KB, NB / nb, KB, nb);

  t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 2 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "2,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();
  /* touch l_input - identical to forward pass - except it is transposed */
  //██████████████████████
  //###FLAT POINTER###
  //██████████████████████
  //float * fp_temp = NULL;
  //int temp_i = 0;
  //int temp_j = 0;
  for (int l_n = 0; l_n < N / NB; ++l_n) {
      for (int l_c = 0; l_c < C / CB; ++l_c) {
          for (int l_nn = 0; l_nn < NB / nb; ++l_nn) {
              for (int l_cc = 0; l_cc < CB; ++l_cc) {
                  float * flat_ptr = (float*)input[l_n][l_c].data_ptr();
                  //fp_temp = (float*)input[l_n][l_c].data_ptr();
                  for (int l_nnn = 0; l_nnn < nb; ++l_nnn) {
                      //temp_i = l_nn * nb + l_nnn;
                      int i = l_nn * nb + l_nnn;
                      //temp_j = l_cc;
                      int j = l_cc;

/*
                      LIBXSMM_VLA_ACCESS(5, l_p_input, l_n, l_c, l_nn, l_cc,
                              l_nnn, C / CB, NB / nb, CB, nb) =
                              (float)input[l_c][l_n][j][i].item().to<float>();
                              */
                      LIBXSMM_VLA_ACCESS(5, l_p_input, l_n, l_c, l_nn, l_cc,
                              l_nnn, C / CB, NB / nb, CB, nb) =
                              //(float)input[l_n][l_c][i][j].item().to<float>();
                              flat_ptr[i*CB+j];
                              //fp_temp[temp_i*CB+temp_j];
                              
                              // Test complete
                              // if ((float)input[l_n][l_c][i][j].item().to<float>() != flat_ptr[i*CB+j])
                              // {
                              //   printf("Error! %lf - %lf\n", (float)input[l_n][l_c][i][j].item().to<float>(), flat_ptr[i*CB+j]);
                              // }
                  }
              }
          }
      }
  }

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 3 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "3,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  //██████████████████████
  //###FLAT POINTER###
  //██████████████████████
  /* touch l_grad_output */
  for (int l_n = 0; l_n < N / NB; ++l_n) {
      for (int l_k = 0; l_k < K / KB; ++l_k) {
          for (int l_nn = 0; l_nn < NB / nb; ++l_nn) {
              for (int l_kk = 0; l_kk < KB; ++l_kk) {
                float * flat_ptr = (float*)grad_output.data_ptr();
                  for (int l_nnn = 0; l_nnn < nb; ++l_nnn) {
                      int depth = ((K/KB)-1) * KB + KB;
                      int i = l_n * NB + l_nn * nb + l_nnn;
                      int j = l_k * KB + l_kk;

                      LIBXSMM_VLA_ACCESS(5, l_p_grad_output, l_n, l_k, l_nn, l_kk,
                              l_nnn, K / KB, NB / nb, KB, nb) =
                              //(float)grad_output[i][j].item().to<float>();
                              flat_ptr[i*KB + j];
                              
                              // Test complete
                              // if ((float)grad_output[i][j].item().to<float>() != flat_ptr[i*depth+j])
                              // {
                              //   printf("Error! %lf - %lf\n", (float)grad_output[i][j].item().to<float>(), flat_ptr[i*depth+j]);
                              //   exit(0);
                              // }
                  }
              }
          }
      }
  }

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 4 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "4,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  /* touch C */
  for (int k = 0; k < K; ++k) {
      for (int c = 0; c < C; ++c) {
          l_C_upd[k * C + c] = 0.0f;
          l_C[k * C + c] = 0.0f;
      }
  }

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 5 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "5,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  /* init sparse C */
  int nnz = 0;

  unsigned int *colptr = (unsigned int *)libxsmm_aligned_malloc(
          (K + 1) * sizeof(unsigned int), 64);

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 6 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "6,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  colptr[0] = 0;
  //██████████████████████
  //###FLAT POINTER###
  //██████████████████████
  for (int l_k = 0; l_k < K; l_k++) {
      colptr[l_k + 1] = 0;
      float * flat_ptr = (float*)weight.data_ptr();
      for (int l_c = 0; l_c < C; l_c++) {
          int depth = C/bc;
          int nbk_idx = l_k / bk;
          int nbc_idx = l_c / bc;
          int bk_idx = l_k % bk;
          int bc_idx = l_c % bc;
          //float tmp = (float)weight[nbk_idx][nbc_idx][bc_idx][bk_idx].item().to<float>();
          float tmp = flat_ptr[nbk_idx * (depth*bc*bk) + nbc_idx * (bc*bk) + bc_idx * bk + bk_idx];
          
          // Test complete
          // if ((float)weight[nbk_idx][nbc_idx][bc_idx][bk_idx].item().to<float>() != tmp)
          // {
          //   printf("Error! %lf - %lf\n", (float)weight[nbk_idx][nbc_idx][bc_idx][bk_idx].item().to<float>(), tmp);
          // }
          if (tmp == 0.0) {
            // pass
          }
          else {
              nnz++;
              colptr[l_k + 1]++;
          }
          l_C[l_k * C + l_c] = (float)tmp;
      }
  }

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 7 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "7,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  for (int l_k = 0; l_k < K; l_k++) {
      colptr[l_k + 1] += colptr[l_k];
  }

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 8 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "8,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  unsigned int *rowidx = (unsigned int *)libxsmm_aligned_malloc(nnz * sizeof(unsigned int), 64);
  float *values = (float *)libxsmm_aligned_malloc(nnz * sizeof(float), 64);

  for (int l_k = 0; l_k < K; l_k++) {
      int offset = colptr[l_k];
      for (int l_c = 0; l_c < C; l_c++) {
          if (l_C[l_k * C + l_c] != 0) {
              rowidx[offset] = l_c;
              values[offset] = 0.0;
              offset++;
          }
      }
  }
  unsigned num_k_blocks = K / KB;
  unsigned num_c_blocks = C / CB;
  int num_blocks = num_k_blocks * num_c_blocks;

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 9 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "9,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  unsigned int **c_colptr = (unsigned int **)libxsmm_aligned_malloc(
          num_blocks * sizeof(unsigned int *), 64);
  unsigned int **c_rowidx = (unsigned int **)libxsmm_aligned_malloc(
          num_blocks * sizeof(unsigned int *), 64);
  float **c_values = (float **)libxsmm_aligned_malloc(num_blocks * sizeof(float *), 64);

  int *nnzb = (int *)libxsmm_aligned_malloc(num_blocks * sizeof(int), 64);

  // Init c_colptr
  for (int blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
      c_colptr[blk_idx] = (unsigned int *)libxsmm_aligned_malloc(
              (KB + 1) * sizeof(unsigned int), 64);
  }

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 10 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "10,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  BlockSpMatStep1(K, C, KB, CB, colptr, rowidx, c_colptr, nnzb);

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 11 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "11,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  // Init c_rowidx, c_values
  for (int blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
      c_rowidx[blk_idx] = (unsigned int *)libxsmm_aligned_malloc(
              nnzb[blk_idx] * sizeof(unsigned int), 64);
      c_values[blk_idx] =
          (float *)libxsmm_aligned_malloc(nnzb[blk_idx] * sizeof(float), 64);
  }

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 12 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "12,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();

  BlockSpMatStep2(K, C, KB, CB, colptr, rowidx, values, c_colptr, c_rowidx, c_values);

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  //duration<double> total_time = duration_cast<duration<double>>(0);
  //total_time += time_span;
  printf("Section 13 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "13,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  t1 = high_resolution_clock::now();
  
  // Update kernels
  float alpha = 1.0;
  float beta = 1.0;
  libxsmm_descriptor_blob l_xgemm_blob;

  libxsmm_gemm_descriptor **l_xgemm_desc = (libxsmm_gemm_descriptor **)libxsmm_aligned_malloc(
          num_blocks * sizeof(libxsmm_gemm_descriptor *), 64);

  libxsmm_smmfunction *upd_kernel = (libxsmm_smmfunction *)libxsmm_aligned_malloc(
          num_blocks * sizeof(libxsmm_smmfunction), 64);

  for (int blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
      l_xgemm_desc[blk_idx] = libxsmm_gemm_descriptor_dinit(
              &l_xgemm_blob, LIBXSMM_DATATYPE(float), CB, KB, NB / nb, CB,
              KB, 0, alpha, beta, flags, prefetch);
      // Creating update micro kernels
      upd_kernel[blk_idx] =
          libxsmm_create_packed_spxgemm_csc(
                  l_xgemm_desc[blk_idx], nb,
                  c_colptr[blk_idx],
                  c_rowidx[blk_idx],
                  (const void *)c_values[blk_idx]).smm;
  }

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  //total_time += time_span;
  printf("Section 14 time: %lf(kernel creation/update)\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "14,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);


t1 = high_resolution_clock::now();
int k, n, c;
#ifdef _OPENMP
#   pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(k,n,c)
#endif
  for (k = 0; k < K / KB; ++k) {
      for (c = 0; c < C / CB; ++c) {
          for (n = 0; n < N / NB; ++n) {
              if (c_values[k * C/CB + c] != NULL) {
                  upd_kernel[k * C / CB + c](
                          &(l_input[(n * C / CB + c) * CB * NB]),
                          &(l_grad_output[(n * K / KB + k) * KB * NB]),
                          c_values[k * C / CB + c]);
              }
          }
      }
  }

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  //total_time += time_span;
  printf("Section 15 time: %lf(kernel execution)\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "15,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);

  t1 = high_resolution_clock::now();
  
  // printf("dims 0: %d 1: %d 2: %d 3: %d\n", grad_weight.sizes()[0], grad_weight.sizes()[1],
  //                                         grad_weight.sizes()[2], grad_weight.sizes()[3]);
  // auto grad_weight_temp = grad_weight.permute({1, 2, 0, 3});
  // printf("dims 0: %d 1: %d 2: %d 3: %d\n", grad_weight_temp.sizes()[0], grad_weight_temp.sizes()[1],
  //                                         grad_weight_temp.sizes()[2], grad_weight_temp.sizes()[3]);
  // grad_weight_temp = grad_weight_temp.reshape({C,K});
  // printf("dims 0: %d 1: %d 2: %d 3: %d\n", grad_weight_temp.sizes()[0], grad_weight_temp.sizes()[1],
  //                                         grad_weight_temp.sizes()[2], grad_weight_temp.sizes()[3]);
  auto grad_weight_temp = grad_weight.permute({1, 2, 0, 3}).reshape({C,K});

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  printf("Section 16 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "16,%lf\n", time_span);
  fclose(fp);
  t1 = high_resolution_clock::now();

  /* Convert back to grad_weight */
  //printf("K: %d KB: %d K/KB: %d\n", K, KB, K/KB);
  int l_cc;
  for (int l_k = 0; l_k < K/KB; ++l_k) {
      for (int l_c = 0; l_c < C/CB; ++l_c) {
          int blk_idx = l_k * C/CB + l_c;
          for (int l_kk = 0; l_kk < KB; ++l_kk) {
              int colstart = c_colptr[blk_idx][l_kk];
              int colend = c_colptr[blk_idx][l_kk + 1];
              k = l_k * KB + l_kk;

              float * flat_pointer = (float*)grad_weight_temp.data_ptr();

              for (int i = colstart; i < colend; ++i) {
                  l_cc = c_rowidx[blk_idx][i];
                  c = l_c * CB + l_cc;

                  //printf("flat: %lf\n", flat_pointer[c*K + k]);

                  //grad_weight_temp.index_put_({c, k}, c_values[blk_idx][i]);
                  
                  // Test complete
                  // if (flat_pointer[c*K + k] != c_values[blk_idx][i])
                  // {
                  //   printf("Error! %lf - %lf\n", flat_pointer[c*K + k], c_values[blk_idx][i]);
                  // }
                  flat_pointer[c*K + k] = c_values[blk_idx][i];
                  //printf("C: %d K: %d c_value: %lf\n", c, k, c_values[blk_idx][i]);
                  //printf("flat: %lf c_value: %lf\n", flat_pointer[c*K + k], c_values[blk_idx][i]);
              }
          }
      }
  }

  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  //total_time += time_span;
  printf("Section 17 time: %lf\n", time_span);
  fp = fopen(fname, "a");
  fprintf(fp, "17,%lf\n", time_span);
  fclose(fp);
  total_time = total_time + duration_cast<duration<double>>(t2 - t1);
  printf("Total time: %lf\n", total_time);

  return grad_weight_temp.reshape({nbc, bc, nbk, bk}).permute({2, 0, 1, 3});
}


std::vector<at::Tensor> mlp_backward(void *libxsmm_handle_, torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight)
{
  libxsmm_dnn_err_t global_status;
  auto nbn = input.size(0);
  auto nbc = input.size(1);
  auto bn = input.size(2);
  auto bc = input.size(3);
  auto nbk = weight.size(0);
  auto bk = weight.size(3);
  //auto grad_input = at::empty({nbn, nbc, bn, bc}, input.options());
  //auto grad_weight = at::empty({nbk, nbc, bc, bk}, weight.options());
  auto grad_input = at::empty(input.sizes(), input.options());
  auto grad_weight = at::empty(weight.sizes(), weight.options());
  auto grad_bias = at::empty({nbk * bk}, weight.options());
#if 1
  //auto grad_bias = grad_output.sum_to_size({1, nbk, 1, bk}).view({-1}).contiguous();
#else
  at::Tensor grad_bias;
  if(grad_output.scalar_type() == at::kBFloat16)
    grad_bias = bias_add_grad<at::BFloat16>(grad_output);
  else
    grad_bias = bias_add_grad<float>(grad_output);
#endif
  libxsmm_dnn_fullyconnected* libxsmm_handle = (libxsmm_dnn_fullyconnected*)libxsmm_handle_;
  //std::cout << "BWD Handle = " << libxsmm_handle_ << std::endl;
  libxsmm_dnn_fullyconnected_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, input, "Input");
  libxsmm_dnn_fullyconnected_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER, weight, "Weight");
  libxsmm_dnn_fullyconnected_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_BIAS, grad_bias.view({nbk, bk}), "GradBias");
  libxsmm_dnn_fullyconnected_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, grad_output, "GradOutput");
  libxsmm_dnn_fullyconnected_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, grad_input, "GradInput");
  libxsmm_dnn_fullyconnected_set_ptr_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER, grad_weight, "GradWeight");

#if 0
{
RECORD_FUNCTION("xsmm_mm_bwd", std::vector<c10::IValue>(/*grad_output, weight*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
{
  int tid = omp_get_thread_num();
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid) );
}
}
{
RECORD_FUNCTION("xsmm_mm_upd", std::vector<c10::IValue>(/*grad_output, input*/), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
{
  int tid = omp_get_thread_num();
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid) );
}
}
#else
{
RECORD_FUNCTION("xsmm_mm_bwdupd", std::vector<c10::IValue>({grad_output, weight}), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
{
  int tid = omp_get_thread_num();
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_execute_st(libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWDUPD, 0, tid) );
}
}
#endif
  return {grad_input, grad_weight, grad_bias};
}

void destroy_handle( void* libxsmm_handle_ )
{
  libxsmm_dnn_err_t global_status;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_fullyconnected* libxsmm_handle = (libxsmm_dnn_fullyconnected*)libxsmm_handle_;
  //std::cout << "Destroy Handle = " << libxsmm_handle << std::endl;

  libxsmm_dnn_fullyconnected_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_RELU_MASK);
  libxsmm_dnn_fullyconnected_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT);
  libxsmm_dnn_fullyconnected_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER);
  libxsmm_dnn_fullyconnected_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT);
  libxsmm_dnn_fullyconnected_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT);
  libxsmm_dnn_fullyconnected_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER);
  libxsmm_dnn_fullyconnected_release_tensor_helper(libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT);
  size_t scratch_size = libxsmm_dnn_fullyconnected_get_scratch_size( libxsmm_handle, &status );
  if(scratch_size > 0) {
    void *scratch = libxsmm_dnn_fullyconnected_get_scratch_ptr( libxsmm_handle, &status );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_release_scratch( libxsmm_handle ) );
    if(scratch) libxsmm_free(scratch);
  }

  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_fullyconnected( libxsmm_handle ) );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mlp_forward, "Pcl libxsmm MLP forward");
  m.def("backward", &mlp_backward, "Pcl libxsmm MLP backward");
  m.def("create_handle", &create_handle, "Pcl libxsmm create MLP handle");
  m.def("set_relu_mask", &mlp_set_relu_mask, "Pcl libxsmm MLP Set ReLU Mask tensor");
  m.def("destroy_handle", &destroy_handle, "Pcl libxsmm destroy MLP handle");
  m.def("sparse_forward", &mlp_sparse_forward, "Pcl libxsmm MLP sparse forward");
  m.def("sparse_backward", &mlp_sparse_backward, "Pcl libxsmm MLP sparse backward");
  m.def("sparse_update", &mlp_sparse_update, "Pcl libxsmm MLP sparse update");
}
