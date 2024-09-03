#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cutlass/numeric_types.h>

#include "flash.h"
#include "static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must be of shape " #__VA_ARGS__)

// set_params_fprop
void set_params_fprop(
    Flash_fwd_params &params,
    const size_t b,
    const size_t seqlen_q, const size_t seqlen_k,
    const size_t h,
    const size_t d,
    const torch::Tensor q,
    const torch::Tensor k,
    const torch::Tensor v,
    torch::Tensor out,
    float softmax_scale,
    bool is_causal
)
{
    // reset params
    params = {};
    params.is_bf16 = q.dtype() == torch::kBFloat16;

    // set the pointers and stride
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = out.data_ptr();

    params.q_batch_stride = q.stride(0);
    params.q_head_stride  = q.stride(-3);
    params.q_row_stride   = q.stride(-2);

    params.k_batch_stride = k.stride(0);
    params.k_head_stride  = k.stride(-3);
    params.k_row_stride   = k.stride(-2);

    params.v_batch_stride = v.stride(0);
    params.v_head_stride  = v.stride(-3);
    params.v_row_stride   = v.stride(-2);

    params.o_batch_stride = out.stride(0);
    params.o_head_stride  = out.stride(-3);
    params.o_row_stride   = out.stride(-2);

    // softmax scale
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    // set the dimensions
    params.b = b;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.h = h;
    params.d = d;

    // causal
    params.is_causal = is_causal;
}

// run_mha_fwd
void run_mha_fwd(
    Flash_fwd_params &params,
    cudaStream_t stream
)
{
    FP16_SWITCH(!params.is_bf16, [&] {
        // These switches are to specify the template parameters
        HEADDIM_SWITCH(params.d, [&] {
            constexpr bool Is_causal = true;
            run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
            });
    });
}

// mha_fwd
std::vector<torch::Tensor>
mha_fwd(
    const torch::Tensor &q,
    const torch::Tensor &k,
    const torch::Tensor &v,
    const float softmax_scale,
    bool is_causal
)
{   
    // args shape
    // args shape
    // q: [batch_size, num_heads, seqlen_q, head_size]
    // k: [batch_size, num_heads, seqlen_k, head_size]
    // v: [batch_size, num_heads, seqlen_k, head_size]
    // simplified version: same num_heads (no MQA/GQA)

    // pre-conditions
    // CHECK_DEVICE
    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);

    // CHECK_CONTIGUOUS
    TORCH_CHECK(q.stride(-1) == 1, "q must be contiguous in last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "k must be contiguous in last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "v must be contiguous in last dimension");

    // init out
    auto out = torch::empty_like(q);

    // extract shape
    auto sizes = out.sizes();
    int batch_size = sizes[0];
    int num_heads = sizes[1];
    int seqlen_q = sizes[2];
    int head_size = sizes[3];

    int seqlen_k = k.sizes()[2];

    // set params
    Flash_fwd_params params;
    set_params_fprop(
        params,
        batch_size, seqlen_q, seqlen_k, num_heads, head_size,
        q, k, v,
        out,
        softmax_scale,
        is_causal
    );

    // call run_mha_fwd
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_mha_fwd(params, stream);

    // return result
    return {out};
}

// pybind
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mha_fwd", &mha_fwd, "FA multihead attention forward");
}