import torch
from torch import Tensor, nn
from torch.autograd import Function

"""
Key Concepts:
x.view_as(x): This returns a view of the tensor x with the same shape and data. In this context, it's used to return the input tensor as is during the forward pass.
.neg(): This computes the element-wise negative of the tensor, effectively reversing the sign of the gradients.
ctx.lambd: Stores the lambd parameter in the context object to be accessed during the backward pass.
GradReverse.apply(x, lambd): This is the standard way to use a custom autograd function in PyTorch. It applies the custom forward and backward logic defined in GradReverse.

Use Case:
The GradReverse function is typically used in domain adversarial training, such as in domain adaptation tasks. It allows you to reverse the gradients for certain parts of the network during backpropagation, which can help to learn domain-invariant features.

For example:
In a domain adaptation setting, you might want to confuse a domain classifier by reversing the gradients, thus forcing the feature extractor to learn features that are not specific to any domain.
"""
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)
    '''
    ctx.lambd: Stores lambd in the context object for use in the backward pass.
    x.view_as(x): Returns a tensor with the same data and shape as x. Here, it's a no-op (doesn't change the input tensor), but is required for the custom function's structure.
    '''

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.neg() * ctx.lambd, None
    '''
    ctx: The context object containing lambd from the forward pass.
    grad_output: The gradient of the loss with respect to the output of the forward pass.
    grad_output.neg(): Computes the negative of grad_output.
    grad_output.neg() * ctx.lambd: Scales the negative gradient by lambd.
    return grad_output.neg() * ctx.lambd, None: Returns the scaled negative gradient for the input tensor and None for the lambd parameter (indicating no gradient computation needed for lambd).
    '''



def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)
    '''
    GradReverse.apply(x, lambd): Applies the GradReverse function to the input tensor x and scalar lambd.
    '''



class AdversarialDiscriminator(nn.Module):
    """
    Discriminator for the adversarial training for batch correction.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.LeakyReLU,
        reverse_grad: bool = False,
    ):
        super().__init__()
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)
        self.reverse_grad = reverse_grad

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0)
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)