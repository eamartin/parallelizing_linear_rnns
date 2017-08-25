#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "../../linear_recurrence.h"

using namespace tensorflow;

REGISTER_OP("LinearRecurrence")
    .Input("decays: float")
    .Input("impulses: float")
    .Input("initial_state: float")
    .Output("response: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(1));
        return Status::OK();
    });

class GpuLinearRecurrenceOp : public OpKernel {
public:
  explicit GpuLinearRecurrenceOp(OpKernelConstruction *ctx): OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    const Tensor& decays_tensor = ctx->input(0);
    const Tensor& impulses_tensor = ctx->input(1);
    const Tensor& initial_state_tensor = ctx->input(2);

    int n_steps = impulses_tensor.dim_size(0);
    int n_dims = impulses_tensor.dim_size(1);

    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(impulses_tensor.shape()),
		errors::InvalidArgument("Impulses must be a matrix"));


    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(decays_tensor.shape()),
		errors::InvalidArgument("Decays must be a matrix"));

    OP_REQUIRES(ctx,
		decays_tensor.dim_size(0) == n_steps &&
		decays_tensor.dim_size(1) == n_dims,
		errors::InvalidArgument("Decay shape must match impulse shape"));

    OP_REQUIRES(ctx,
		TensorShapeUtils::IsVector(initial_state_tensor.shape()) &&
		initial_state_tensor.dim_size(0) == n_dims,
		errors::InvalidArgument("Initial state must be a vector of length n_dims"));

    Tensor *response_tensor = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, impulses_tensor.shape(), &response_tensor));

    auto decays = decays_tensor.flat<float>();
    auto impulses = impulses_tensor.flat<float>();
    auto initial_state = initial_state_tensor.flat<float>();
    auto response = response_tensor->template flat<float>();

    compute_linear_recurrence(decays.data(), impulses.data(),
			      initial_state.data(), response.data(),
			      n_dims, n_steps);
  }
};
REGISTER_KERNEL_BUILDER(Name("LinearRecurrence").Device(DEVICE_GPU), GpuLinearRecurrenceOp);
