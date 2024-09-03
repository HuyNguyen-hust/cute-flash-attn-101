#include "flash_fwd_launch_template.h"

// Explicitly instantiate for bfloat16, hdim64
template<>
void run_mha_fwd_<cutlass::bfloat16_t, 64, true>(Flash_fwd_params &params, cudaStream_t stream)
{
    run_mha_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
}