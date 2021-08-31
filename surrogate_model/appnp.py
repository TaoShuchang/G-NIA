from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SparseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        input_coal = input.coalesce()
        drop_val = F.dropout(input_coal._values(), self.p, self.training)
        return torch.sparse.FloatTensor(input_coal._indices(), drop_val, input.shape)


class MixedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Our fan_in is interpreted by PyTorch as fan_out (swapped dimensions)
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.bias is None:
            if input.is_sparse:
                res = torch.sparse.mm(input, self.weight)
            else:
                res = input.matmul(self.weight)
        else:
            if input.is_sparse:
                res = torch.sparse.addmm(self.bias.expand(input.shape[0], -1), input, self.weight)
            else:
                res = torch.addmm(self.bias, input, self.weight)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None)


class MixedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        if input.is_sparse:
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)

class PPRPowerIteration(nn.Module):
    def __init__(self, alpha: float, niter: int, drop_prob: float = None):
        super().__init__()
        self.alpha = alpha
        self.niter = niter

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, local_preds: torch.FloatTensor, nor_adj_tensor):
        preds = local_preds
        self.A_hat = (1 - self.alpha) * nor_adj_tensor
        for _ in range(self.niter):
            A_drop = self.dropout(self.A_hat)
            # preds = A_drop @ preds + self.alpha * local_preds
            preds = torch.sparse.mm(A_drop, preds) + self.alpha * local_preds
        return preds


class PPNP(nn.Module):
    def __init__(self, nfeatures: int, nclasses: int, hiddenunits: List[int], drop_prob: float,
                 propagation: nn.Module, bias: bool = False):
        super().__init__()

        fcs = [MixedLinear(nfeatures, hiddenunits[0], bias=bias)]
        for i in range(1, len(hiddenunits)):
            fcs.append(nn.Linear(hiddenunits[i - 1], hiddenunits[i], bias=bias))
        fcs.append(nn.Linear(hiddenunits[-1], nclasses, bias=bias))
        self.fcs = nn.ModuleList(fcs)

        self.reg_params = list(self.fcs[0].parameters())

        if drop_prob is 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)
        self.act_fn = nn.ReLU()

        self.propagation = propagation

    def _transform_features(self, attr_matrix: torch.sparse.FloatTensor):
        layer_inner = self.act_fn(self.fcs[0](self.dropout(attr_matrix)))
        for fc in self.fcs[1:-1]:
            layer_inner = self.act_fn(fc(layer_inner))
        res = self.fcs[-1](self.dropout(layer_inner))
        return res

    def forward(self, attr_matrix: torch.sparse.FloatTensor, nor_adj_tensor):
        local_logits = self._transform_features(attr_matrix)
        final_logits = self.propagation(local_logits,nor_adj_tensor)
        return final_logits
