#include <torch/extension.h>

torch::Tensor run_fused(
    std::vector<torch::Tensor> tensors,
    torch::Tensor program_tensor
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_fused", &run_fused);
}
