import torch
import numpy as np


import torchvision.models as models
model = models.resnet50(pretrained=True)

model.eval()

input_batch = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
golden_confidences = output[0]
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
golden_probabilities = torch.nn.functional.softmax(
    golden_confidences, dim=0
).numpy()

golden_confidences = golden_confidences.numpy()

from shark.torch_mlir_lockstep_tensor import TorchMLIRLockstepTensor

input_detached_clone = input_batch.clone()
eager_input_batch = TorchMLIRLockstepTensor(input_detached_clone)

print("getting torch-mlir result")

output = model(eager_input_batch)

static_output = output.elem
confidences = static_output[0]
probabilities = torch.nn.functional.softmax(
    torch.from_numpy(confidences), dim=0
).numpy()

print("The obtained result via shark is: ", confidences)
print("The golden result is:", golden_confidences)

np.testing.assert_allclose(
    golden_confidences, confidences, rtol=1e-02, atol=1e-03
)
np.testing.assert_allclose(
    golden_probabilities, probabilities, rtol=1e-02, atol=1e-03
)
