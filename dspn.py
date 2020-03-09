import torch
import torch.nn as nn
import torch.nn.functional as F
import higher


class InnerSet(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask
    
    def forward(self):
        return self.mask

class DSPN(nn.Module):
    """ Deep Set Prediction Networks
    Yan Zhang, Jonathon Hare, Adam Prügel-Bennett
    https://arxiv.org/abs/1906.06565
    """

    def __init__(self, encoder, set_channels, iters, lr):
        """
        encoder: Set encoder module that takes a set as input and returns a representation thereof.
            It should have a forward function that takes two arguments:
            - a set: FloatTensor of size (batch_size, input_channels, maximum_set_size). Each set
            should be padded to the same maximum size with 0s, even across batches.
            - a mask: FloatTensor of size (batch_size, maximum_set_size). This should take the value 1
            if the corresponding element is present and 0 if not.

        channels: Number of channels of the set to predict.

        max_set_size: Maximum size of the set.

        iter: Number of iterations to run the DSPN algorithm for.

        lr: Learning rate of inner gradient descent in DSPN.
        """
        super().__init__()
        self.encoder = encoder
        self.iters = iters
        self.lr = lr

    def forward(self, target_repr, init):
        """
        Conceptually, DSPN simply turns the target_repr feature vector into a set.

        target_repr: Representation that the predicted set should match. FloatTensor of size (batch_size, repr_channels).
        This can come from a set processed with the same encoder as self.encoder (auto-encoder), or a different
        input completely (normal supervised learning), such as an image encoded into a feature vector.
        """
        # copy same initial set over batch
        current_set = nn.Parameter(init)
        inner_set = InnerSet(current_set)

        # info used for loss computation
        intermediate_sets = [current_set]
        # info used for debugging
        repr_losses = []
        grad_norms = []
 
        # optimise repr_loss for fixed number of steps
        with torch.enable_grad():
            opt = torch.optim.SGD(inner_set.parameters(), lr=self.lr, momentum=0.5)
            with higher.innerloop_ctx(inner_set, opt) as (fset, diffopt):
                for i in range(self.iters):
                    predicted_repr = self.encoder(fset())
                    # how well does the representation matches the target
                    repr_loss = ((predicted_repr- target_repr)**2).sum() 
                    diffopt.step(repr_loss)
                    intermediate_sets.append(fset.mask)
                    repr_losses.append(repr_loss)
                    grad_norms.append(())
        
        return intermediate_sets, repr_losses, grad_norms
