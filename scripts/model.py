import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv

class scGAT_with_BN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.BN_nodefeat = nn.BatchNorm1d(dim_in)
        self.gat1 = GATConv(dim_in, out_channels=8,
                            heads=8, concat=True, negative_slope=0.2,
                            dropout=0.4, bias=True)
        self.gat2 = GATConv(8*8, dim_out,
                            heads=8, concat=False, negative_slope=0.2,
                            dropout=0.4, bias=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.BN_nodefeat(x) 
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
# class scGAT(nn.Module):
#     def __init__(self, dim_in, dim_out):
#         super().__init__()
#         self.gat1 = GATConv(dim_in, out_channels=8,
#                             heads=8, concat=True, negative_slope=0.2,
#                             dropout=0.4, bias=True)
#         self.gat2 = GATConv(8*8, dim_out,
#                             heads=8, concat=False, negative_slope=0.2,
#                             dropout=0.4, bias=True)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.gat1(x, edge_index)
#         x = F.elu(x)
#         x = self.gat2(x, edge_index)
#         return F.log_softmax(x, dim=1)

class yourecs(nn.Module):
    '''Updating with all design space for GNN recommendations.
    
    REF: 
      - You et al. NIPS 2020
    
    Results: 
      Several models do not train effectively,
        so go back to the simple stuff. See experiments210411.csv
    '''
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.BN_nodefeat = nn.BatchNorm1d(dim_in)
        self.gat1 = GATConv(dim_in, out_channels=64, 
                            heads=16, concat=True, 
                            negative_slope=0.2, 
                            dropout=0.0, bias=True)
        self.gat2 = GATConv(dim_in+64*16, out_channels=8, 
                            heads=8, concat=True, 
                            negative_slope=0.2, 
                            dropout=0.0, bias=True)
        # self.gat3??? add another layer (try 4 and 6)
        self.gat3 = GATConv(8*8, out_channels=dim_out, 
                            heads=8, concat=False, 
                            negative_slope=0.2, 
                            dropout=0.0, bias=True)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        
    def forward(self, data):
        x_in, edge_index = data.x, data.edge_index 
        x_in = self.BN_nodefeat(x_in)
        x = self.gat1(x_in, edge_index)
        x = self.prelu1(x) # F.elu(x) instead?
        x = self.gat2(torch.cat((x_in, x), 1), edge_index)
        x = self.prelu2(x)
        x = self.gat3(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    
# TabNet, Arix & Pfister. AAAI'21
# following implementation discussed here: <https://towardsdatascience.com/implementing-tabnet-in-pytorch-fc977c383279>
class GhostBatchNorm(nn.Module):
    def __init__(self, D_in, B_v=128, m_B=0.01):
        '''
        Arguments:
          B_v (int): (optional, Default=128) virtual batch size. Must be
            smaller than n_samples in mini-batch
          
        '''
        super().__init__()
        self.BN = nn.BatchNorm1d(D_in, momentum=m_B)
        self.B_v = B_v
        
    def forward(self, x):
        chunk = torch.chunk(x, x.size(0) // self.B_v, 0)
        res = [self.BN(x_sub) for x_sub in chunk]
        return torch.cat(res,0)
        
# implementation of SparseMax
# REF: https://github.com/aced125/sparsemax

def flatten_all_but_nth_dim(ctx, x: torch.Tensor):
    """
    Flattens tensor in all but 1 chosen dimension.
    Saves necessary context for backward pass and unflattening.
    """

    # transpose batch and nth dim
    x = x.transpose(0, ctx.dim)

    # Get and save original size in context for backward pass
    original_size = x.size()
    ctx.original_size = original_size

    # Flatten all dimensions except nth dim
    x = x.reshape(x.size(0), -1)

    # Transpose flattened dimensions to 0th dim, nth dim to last dim
    return ctx, x.transpose(0, -1)


def unflatten_all_but_nth_dim(ctx, x: torch.Tensor):
    """
    Unflattens tensor using necessary context
    """
    # Tranpose flattened dim to last dim, nth dim to 0th dim
    x = x.transpose(0, 1)

    # Reshape to original size
    x = x.reshape(ctx.original_size)

    # Swap batch dim and nth dim
    return ctx, x.transpose(0, ctx.dim)

class Sparsemax(nn.Module):
    __constants__ = ["dim"]

    def __init__(self, dim=-1):
        """
        Sparsemax class as seen in https://arxiv.org/pdf/1602.02068.pdf
        Parameters
        ----------
        dim: The dimension we want to cast the operation over. Default -1
        """
        super(Sparsemax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "dim"):
            self.dim = None

    def forward(self, input):
        return SparsemaxFunction.apply(input, self.dim)

    def extra_repr(self):
        return f"dim={self.dim}"


class SparsemaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, dim: int = -1):
        input_dim = input.dim()
        if input_dim <= dim or dim < -input_dim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [-{input_dim}, {input_dim - 1}], but got {dim})"
            )

        # Save operating dimension to context
        ctx.needs_reshaping = input_dim > 2
        ctx.dim = dim

        if ctx.needs_reshaping:
            ctx, input = flatten_all_but_nth_dim(ctx, input)

        # Translate by max for numerical stability
        input = input - input.max(-1, keepdim=True).values.expand_as(input)

        zs = input.sort(-1, descending=True).values
        range = torch.arange(1, input.size()[-1] + 1)
        range = range.expand_as(input).to(input)

        # Determine sparsity of projection
        bound = 1 + range * zs
        is_gt = bound.gt(zs.cumsum(-1)).type(input.dtype)
        k = (is_gt * range).max(-1, keepdim=True).values

        # Compute threshold
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (zs_sparse.sum(-1, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        output = torch.max(torch.zeros_like(input), input - taus)

        # Save context
        ctx.save_for_backward(output)

        # Reshape back to original shape
        if ctx.needs_reshaping:
            ctx, output = unflatten_all_but_nth_dim(ctx, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, *_ = ctx.saved_tensors

        # Reshape if needed
        if ctx.needs_reshaping:
            ctx, grad_output = flatten_all_but_nth_dim(ctx, grad_output)

        # Compute gradient
        nonzeros = torch.ne(output, 0)
        num_nonzeros = nonzeros.sum(-1, keepdim=True)
        sum = (grad_output * nonzeros).sum(-1, keepdim=True) / num_nonzeros
        grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        # Reshape back to original shape
        if ctx.needs_reshaping:
            ctx, grad_input = unflatten_all_but_nth_dim(ctx, grad_input)

        return grad_input, None
        
class AttentiveTransformer(nn.Module):
    def __init__(self, N_a, input_dim, gamma_relaxation, B_v=128):
        super().__init__()
        self.fc = nn.Linear(N_a, input_dim)
        self.BN = GhostBatchNorm(input_dim, B_v=B_v) # instead of inp_idm, out_dim? otherwise, error in the medium post
        self.sparsemax = Sparsemax()
        self.gamma_relaxation = gamma_relaxation
        
    # a := feature from previous decision step
    def forward(self, a, PriorScales): 
        a = self.BN(self.fc(a)) 
        mask = self.sparsemax(a * PriorScales) 
        PriorScales = PriorScales * (self.gamma_relaxation - mask)  #updating the prior
        return mask
    
class GLU(nn.Module):
    def __init__(self, input_dim, output_dim, fc=None, B_v=128):
        '''Internal block in feature transformer.
        
        Arguments:
          fc (nn.Module): (optional, Default=None) feed in nn.Module, allowing
            for shared feature processing across decision steps
        '''
        super().__init__()
        if fc is not None:
            self.fc = fc
        else:
            self.fc = nn.Linear(input_dim, 2*output_dim)
        self.BN = GhostBatchNorm(2*output_dim, B_v=B_v) 
        self.output_dim = output_dim
        
    def forward(self, x):
        x = self.BN(self.fc(x))
        return x[:, :self.output_dim]*torch.sigmoid(x[:, self.output_dim:])
    
class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, shared=None, n_independent=2, B_v=128):
        '''Concatenate blocks shared across decision steps and decision-step dependent
             feature processing.
             
        Notation:
          Based on Arik & Pfister:
            input_dim ~ D
            output_dim ~ N_d
        
        Arguments:
          shared (list(nn.Module(s))): feed in nn.Module list to allow
            for shared feature processing across decision steps
        '''
        super().__init__()
        first = True
        self.shared = nn.ModuleList()
        if shared is not None:
            self.shared.append(GLU(input_dim, output_dim, shared[0], B_v=B_v))
            first = False    
            for fc in shared[1:]:
                self.shared.append(GLU(output_dim, output_dim, fc, B_v=B_v))
        else:
            self.shared = None
        self.decstep_dep = nn.ModuleList() # Decision step dependent layers
        if first:
            self.decstep_dep.append(GLU(input_dim, output_dim, B_v=B_v))
        for x in range(first, n_independent):
            self.decstep_dep.append(GLU(output_dim, output_dim, B_v=B_v))
        self.scale = torch.sqrt(torch.tensor([0.5]))
        
    def forward(self, x):
        if self.shared is not None:
            x = self.shared[0](x)
            for glu in self.shared[1:]:
                x = torch.add(x, glu(x))
                x = x*self.scale.to(x.device)
        for glu in self.decstep_dep:
            x = torch.add(x, glu(x))
            x = x*self.scale.to(x.device)
        return x
    
class DecisionStep(nn.Module):
    def __init__(self, input_dim, N_d, N_a, shared, n_independent, gamma_relaxation, B_v=128, eps=1e-10):
        super().__init__()
        self.feat_transformer = FeatureTransformer(input_dim, N_d + N_a, shared=shared, n_independent=n_independent, B_v=B_v)
        self.attn_transformer =  AttentiveTransformer(N_a, input_dim, gamma_relaxation, B_v=B_v)
        self.eps = eps
        
    def forward(self, x, a, priors):
        mask = self.attn_transformer(a, priors)
        sparse_loss = ((-1) * mask * torch.log(mask + self.eps)).mean()
        x = self.feat_transformer(x * mask)
        return x, sparse_loss
    
class TabNet(nn.Module):
    def __init__(self, input_dim, output_dim,
                 N_d=256, N_a=256,
                 n_shared=2, n_ind=2,
                 n_steps=5, relax=1.2, vbs=128):
        super().__init__()
        
        if n_shared>0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(input_dim, 2*(N_d + N_a)))
            
            for x in range(n_shared-1):
                self.shared.append(nn.Linear(N_d + N_a, 2*(N_d + N_a)))
        else:
            self.shared = None
            
        self.first_step = FeatureTransformer(input_dim, N_d + N_a, shared=self.shared, n_independent=n_ind, B_v=vbs) 
        self.steps = nn.ModuleList()
        
        for x in range(n_steps - 1):
            self.steps.append(DecisionStep(input_dim, N_d, N_a, self.shared, n_ind, relax, B_v=vbs))
        
        self.fc = nn.Linear(N_d, output_dim)
        self.bn = nn.BatchNorm1d(input_dim)
        self.N_d = N_d
        
    def forward(self, x):
        x = self.bn(x)
        x_a = self.first_step(x)[:,self.N_d:]
        sparse_loss = torch.zeros(1).to(x.device)
        out = torch.zeros(x.size(0),self.N_d).to(x.device)
        priors = torch.ones(x.shape).to(x.device)
        for step in self.steps:
            x_te, l = step(x, x_a, priors)
            out += F.relu(x_te[:,: self.N_d])
            x_a = x_te[:, self.N_d:]
            sparse_loss += l
        return F.log_softmax(self.fc(out), dim=-1), sparse_loss
    

# multitasking 4 single-cell targeting data
class scGAT4multitasking(nn.Module):
    '''NOTE: this is also the modification for Captum.
    
    '''
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.gat1 = GATConv(dim_in, out_channels=8,
                            heads=8, concat=True, negative_slope=0.2,
                            dropout=0.4, bias=True)
        self.gat2 = GATConv(8*8, dim_out,
                            heads=8, concat=False, negative_slope=0.2,
                            dropout=0.4, bias=True)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
class scGAT_MT(nn.Module):
    def __init__(self, dim_in, scgat_dim_out, tabnet_dim_out):
        super().__init__()
        # self.fc1 = nn.Linear(dim_in, 2048) 
        # self.LN = nn.LayerNorm(dim_in)
        self.scgat = scGAT4multitasking(dim_in, scgat_dim_out)
        self.tabnet = TabNet(dim_in, tabnet_dim_out, 
                               N_d=256, N_a=256,
                               n_shared=3, n_ind=3,
                               n_steps=16, relax=1.5, vbs=128)
        
    def forward(self, data):
        x, edge_index = data.x[:, :-32], data.edge_index
        yhat_1 = self.scgat(x, edge_index)
        yhat_2, sparse_loss = self.tabnet(x)
        return yhat_1, yhat_2, sparse_loss
        
class TabNet_noinitBN(nn.Module):
    def __init__(self, input_dim, output_dim,
                 N_d=256, N_a=256,
                 n_shared=2, n_ind=2,
                 n_steps=5, relax=1.2, vbs=128):
        super().__init__()
        
        if n_shared>0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(input_dim, 2*(N_d + N_a)))
            
            for x in range(n_shared-1):
                self.shared.append(nn.Linear(N_d + N_a, 2*(N_d + N_a)))
        else:
            self.shared = None
            
        self.first_step = FeatureTransformer(input_dim, N_d + N_a, shared=self.shared, n_independent=n_ind, B_v=vbs) 
        self.steps = nn.ModuleList()
        
        for x in range(n_steps - 1):
            self.steps.append(DecisionStep(input_dim, N_d, N_a, self.shared, n_ind, relax, B_v=vbs))
        
        self.fc = nn.Linear(N_d, output_dim)
        self.N_d = N_d
        
    def forward(self, x):
        x_a = self.first_step(x)[:,self.N_d:]
        sparse_loss = torch.zeros(1).to(x.device)
        out = torch.zeros(x.size(0),self.N_d).to(x.device)
        priors = torch.ones(x.shape).to(x.device)
        for step in self.steps:
            x_te, l = step(x, x_a, priors)
            out += F.relu(x_te[:,: self.N_d])
            x_a = x_te[:, self.N_d:]
            sparse_loss += l
        return F.log_softmax(self.fc(out), dim=-1), sparse_loss
    
    
    

# try various normalizations    

## no ghost BN
class AttentiveTransformer_noGhostBN(nn.Module):
    def __init__(self, N_a, input_dim, gamma_relaxation):
        super().__init__()
        self.fc = nn.Linear(N_a, input_dim)
        self.BN = nn.BatchNorm1d(input_dim) 
        self.sparsemax = Sparsemax()
        self.gamma_relaxation = gamma_relaxation
        
    # a := feature from previous decision step
    def forward(self, a, PriorScales): 
        a = self.BN(self.fc(a)) 
        mask = self.sparsemax(a * PriorScales) 
        PriorScales = PriorScales * (self.gamma_relaxation - mask)  #updating the prior
        return mask
    
class GLU_noGhostBN(nn.Module):
    def __init__(self, input_dim, output_dim, fc=None):
        '''Internal block in feature transformer.
        
        Arguments:
          fc (nn.Module): (optional, Default=None) feed in nn.Module, allowing
            for shared feature processing across decision steps
        '''
        super().__init__()
        if fc is not None:
            self.fc = fc
        else:
            self.fc = nn.Linear(input_dim, 2*output_dim)
        self.BN = nn.BatchNorm1d(2*output_dim) 
        self.output_dim = output_dim
        
    def forward(self, x):
        x = self.BN(self.fc(x))
        return x[:, :self.output_dim]*torch.sigmoid(x[:, self.output_dim:])
    
class FeatureTransformer_noGhostBN(nn.Module):
    def __init__(self, input_dim, output_dim, shared=None, n_independent=2):
        '''Concatenate blocks shared across decision steps and decision-step dependent
             feature processing.
             
        Notation:
          Based on Arik & Pfister:
            input_dim ~ D
            output_dim ~ N_d
        
        Arguments:
          shared (list(nn.Module(s))): feed in nn.Module list to allow
            for shared feature processing across decision steps
        '''
        super().__init__()
        first = True
        self.shared = nn.ModuleList()
        if shared is not None:
            self.shared.append(GLU_noGhostBN(input_dim, output_dim, shared[0]))
            first = False    
            for fc in shared[1:]:
                self.shared.append(GLU_noGhostBN(output_dim, output_dim, fc))
        else:
            self.shared = None
        self.decstep_dep = nn.ModuleList() # Decision step dependent layers
        if first:
            self.decstep_dep.append(GLU_noGhostBN(input_dim, output_dim))
        for x in range(first, n_independent):
            self.decstep_dep.append(GLU_noGhostBN(output_dim, output_dim))
        self.scale = torch.sqrt(torch.tensor([0.5]))
        
    def forward(self, x):
        if self.shared is not None:
            x = self.shared[0](x)
            for glu in self.shared[1:]:
                x = torch.add(x, glu(x))
                x = x*self.scale.to(x.device)
        for glu in self.decstep_dep:
            x = torch.add(x, glu(x))
            x = x*self.scale.to(x.device)
        return x
    
class DecisionStep_noGhostBN(nn.Module):
    def __init__(self, input_dim, N_d, N_a, shared, n_independent, gamma_relaxation, eps=1e-10):
        super().__init__()
        self.feat_transformer = FeatureTransformer_noGhostBN(input_dim, N_d + N_a, shared=shared, n_independent=n_independent)
        self.attn_transformer =  AttentiveTransformer_noGhostBN(N_a, input_dim, gamma_relaxation)
        self.eps = eps
        
    def forward(self, x, a, priors):
        mask = self.attn_transformer(a, priors)
        sparse_loss = ((-1) * mask * torch.log(mask + self.eps)).mean()
        x = self.feat_transformer(x * mask)
        return x, sparse_loss
    
class TabNet_noGhostBN(nn.Module):
    def __init__(self, input_dim, output_dim,
                 N_d=256, N_a=256,
                 n_shared=2, n_ind=2,
                 n_steps=5, relax=1.2):
        super().__init__()
        
        if n_shared>0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(input_dim, 2*(N_d + N_a)))
            
            for x in range(n_shared-1):
                self.shared.append(nn.Linear(N_d + N_a, 2*(N_d + N_a)))
        else:
            self.shared = None
            
        self.first_step = FeatureTransformer_noGhostBN(input_dim, N_d + N_a, shared=self.shared, n_independent=n_ind) 
        self.steps = nn.ModuleList()
        
        for x in range(n_steps - 1):
            self.steps.append(DecisionStep_noGhostBN(input_dim, N_d, N_a, self.shared, n_ind, relax))
        
        self.fc = nn.Linear(N_d, output_dim)
        self.N_d = N_d
        
    def forward(self, x):
        x_a = self.first_step(x)[:,self.N_d:]
        sparse_loss = torch.zeros(1).to(x.device)
        out = torch.zeros(x.size(0),self.N_d).to(x.device)
        priors = torch.ones(x.shape).to(x.device)
        for step in self.steps:
            x_te, l = step(x, x_a, priors)
            out += F.relu(x_te[:,: self.N_d])
            x_a = x_te[:, self.N_d:]
            sparse_loss += l
        return F.log_softmax(self.fc(out), dim=-1), sparse_loss
    
## no batch norm
class AttentiveTransformer_noBN(nn.Module):
    def __init__(self, N_a, input_dim, gamma_relaxation):
        super().__init__()
        self.fc = nn.Linear(N_a, input_dim)
        self.sparsemax = Sparsemax()
        self.gamma_relaxation = gamma_relaxation
        
    # a := feature from previous decision step
    def forward(self, a, PriorScales): 
        a = self.fc(a) 
        mask = self.sparsemax(a * PriorScales) 
        PriorScales = PriorScales * (self.gamma_relaxation - mask)  #updating the prior
        return mask
    
class GLU_noBN(nn.Module):
    def __init__(self, input_dim, output_dim, fc=None):
        '''Internal block in feature transformer.
        
        Arguments:
          fc (nn.Module): (optional, Default=None) feed in nn.Module, allowing
            for shared feature processing across decision steps
        '''
        super().__init__()
        if fc is not None:
            self.fc = fc
        else:
            self.fc = nn.Linear(input_dim, 2*output_dim)
        self.output_dim = output_dim
        
    def forward(self, x):
        x = self.fc(x)
        return x[:, :self.output_dim]*torch.sigmoid(x[:, self.output_dim:])
    
class FeatureTransformer_noBN(nn.Module):
    def __init__(self, input_dim, output_dim, shared=None, n_independent=2):
        '''Concatenate blocks shared across decision steps and decision-step dependent
             feature processing.
             
        Notation:
          Based on Arik & Pfister:
            input_dim ~ D
            output_dim ~ N_d
        
        Arguments:
          shared (list(nn.Module(s))): feed in nn.Module list to allow
            for shared feature processing across decision steps
        '''
        super().__init__()
        first = True
        self.shared = nn.ModuleList()
        if shared is not None:
            self.shared.append(GLU_noBN(input_dim, output_dim, shared[0]))
            first = False    
            for fc in shared[1:]:
                self.shared.append(GLU_noBN(output_dim, output_dim, fc))
        else:
            self.shared = None
        self.decstep_dep = nn.ModuleList() # Decision step dependent layers
        if first:
            self.decstep_dep.append(GLU_noBN(input_dim, output_dim))
        for x in range(first, n_independent):
            self.decstep_dep.append(GLU_noBN(output_dim, output_dim))
        self.scale = torch.sqrt(torch.tensor([0.5]))
        
    def forward(self, x):
        if self.shared is not None:
            x = self.shared[0](x)
            for glu in self.shared[1:]:
                x = torch.add(x, glu(x))
                x = x*self.scale.to(x.device)
        for glu in self.decstep_dep:
            x = torch.add(x, glu(x))
            x = x*self.scale.to(x.device)
        return x
    
class DecisionStep_noBN(nn.Module):
    def __init__(self, input_dim, N_d, N_a, shared, n_independent, gamma_relaxation, eps=1e-10):
        super().__init__()
        self.feat_transformer = FeatureTransformer_noBN(input_dim, N_d + N_a, shared=shared, n_independent=n_independent)
        self.attn_transformer =  AttentiveTransformer_noBN(N_a, input_dim, gamma_relaxation)
        self.eps = eps
        
    def forward(self, x, a, priors):
        mask = self.attn_transformer(a, priors)
        sparse_loss = ((-1) * mask * torch.log(mask + self.eps)).mean()
        x = self.feat_transformer(x * mask)
        return x, sparse_loss
    
class TabNet_noBN(nn.Module):
    def __init__(self, input_dim, output_dim,
                 N_d=256, N_a=256,
                 n_shared=2, n_ind=2,
                 n_steps=5, relax=1.2):
        super().__init__()
        
        if n_shared>0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(input_dim, 2*(N_d + N_a)))
            
            for x in range(n_shared-1):
                self.shared.append(nn.Linear(N_d + N_a, 2*(N_d + N_a)))
        else:
            self.shared = None
            
        self.first_step = FeatureTransformer_noBN(input_dim, N_d + N_a, shared=self.shared, n_independent=n_ind) 
        self.steps = nn.ModuleList()
        
        for x in range(n_steps - 1):
            self.steps.append(DecisionStep_noBN(input_dim, N_d, N_a, self.shared, n_ind, relax))
        
        self.fc = nn.Linear(N_d, output_dim)
        self.N_d = N_d
        
    def forward(self, x):
        x_a = self.first_step(x)[:,self.N_d:]
        sparse_loss = torch.zeros(1).to(x.device)
        out = torch.zeros(x.size(0),self.N_d).to(x.device)
        priors = torch.ones(x.shape).to(x.device)
        for step in self.steps:
            x_te, l = step(x, x_a, priors)
            out += F.relu(x_te[:,: self.N_d])
            x_a = x_te[:, self.N_d:]
            sparse_loss += l
        return F.log_softmax(self.fc(out), dim=-1), sparse_loss
    

# layer norm    
class AttentiveTransformer_LN(nn.Module):
    def __init__(self, N_a, input_dim, gamma_relaxation):
        super().__init__()
        self.fc = nn.Linear(N_a, input_dim)
        self.LN = nn.LayerNorm(input_dim) 
        self.sparsemax = Sparsemax()
        self.gamma_relaxation = gamma_relaxation
        
    # a := feature from previous decision step
    def forward(self, a, PriorScales): 
        a = self.LN(self.fc(a)) 
        mask = self.sparsemax(a * PriorScales) 
        PriorScales = PriorScales * (self.gamma_relaxation - mask)  #updating the prior
        return mask
    
class GLU_LN(nn.Module):
    def __init__(self, input_dim, output_dim, fc=None):
        '''Internal block in feature transformer.
        
        Arguments:
          fc (nn.Module): (optional, Default=None) feed in nn.Module, allowing
            for shared feature processing across decision steps
        '''
        super().__init__()
        if fc is not None:
            self.fc = fc
        else:
            self.fc = nn.Linear(input_dim, 2*output_dim)
        self.LN = nn.LayerNorm(2*output_dim) 
        self.output_dim = output_dim
        
    def forward(self, x):
        x = self.LN(self.fc(x))
        return x[:, :self.output_dim]*torch.sigmoid(x[:, self.output_dim:])
    
class FeatureTransformer_LN(nn.Module):
    def __init__(self, input_dim, output_dim, shared=None, n_independent=2):
        '''Concatenate blocks shared across decision steps and decision-step dependent
             feature processing.
             
        Notation:
          Based on Arik & Pfister:
            input_dim ~ D
            output_dim ~ N_d
        
        Arguments:
          shared (list(nn.Module(s))): feed in nn.Module list to allow
            for shared feature processing across decision steps
        '''
        super().__init__()
        first = True
        self.shared = nn.ModuleList()
        if shared is not None:
            self.shared.append(GLU_LN(input_dim, output_dim, shared[0]))
            first = False    
            for fc in shared[1:]:
                self.shared.append(GLU_LN(output_dim, output_dim, fc))
        else:
            self.shared = None
        self.decstep_dep = nn.ModuleList() # Decision step dependent layers
        if first:
            self.decstep_dep.append(GLU_LN(input_dim, output_dim))
        for x in range(first, n_independent):
            self.decstep_dep.append(GLU_LN(output_dim, output_dim))
        self.scale = torch.sqrt(torch.tensor([0.5]))
        
    def forward(self, x):
        if self.shared is not None:
            x = self.shared[0](x)
            for glu in self.shared[1:]:
                x = torch.add(x, glu(x))
                x = x*self.scale.to(x.device)
        for glu in self.decstep_dep:
            x = torch.add(x, glu(x))
            x = x*self.scale.to(x.device)
        return x
    
class DecisionStep_LN(nn.Module):
    def __init__(self, input_dim, N_d, N_a, shared, n_independent, gamma_relaxation, eps=1e-10):
        super().__init__()
        self.feat_transformer = FeatureTransformer_LN(input_dim, N_d + N_a, shared=shared, n_independent=n_independent)
        self.attn_transformer =  AttentiveTransformer_LN(N_a, input_dim, gamma_relaxation)
        self.eps = eps
        
    def forward(self, x, a, priors):
        mask = self.attn_transformer(a, priors)
        sparse_loss = ((-1) * mask * torch.log(mask + self.eps)).mean()
        x = self.feat_transformer(x * mask)
        return x, sparse_loss
    
class TabNet_LN(nn.Module):
    def __init__(self, input_dim, output_dim,
                 N_d=256, N_a=256,
                 n_shared=2, n_ind=2,
                 n_steps=5, relax=1.2):
        super().__init__()
        
        if n_shared>0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(input_dim, 2*(N_d + N_a)))
            
            for x in range(n_shared-1):
                self.shared.append(nn.Linear(N_d + N_a, 2*(N_d + N_a)))
        else:
            self.shared = None
            
        self.first_step = FeatureTransformer_LN(input_dim, N_d + N_a, shared=self.shared, n_independent=n_ind) 
        self.steps = nn.ModuleList()
        
        for x in range(n_steps - 1):
            self.steps.append(DecisionStep_LN(input_dim, N_d, N_a, self.shared, n_ind, relax))
        
        self.fc = nn.Linear(N_d, output_dim)
        self.N_d = N_d
        
    def forward(self, x):
        x_a = self.first_step(x)[:,self.N_d:]
        sparse_loss = torch.zeros(1).to(x.device)
        out = torch.zeros(x.size(0),self.N_d).to(x.device)
        priors = torch.ones(x.shape).to(x.device)
        for step in self.steps:
            x_te, l = step(x, x_a, priors)
            out += F.relu(x_te[:,: self.N_d])
            x_a = x_te[:, self.N_d:]
            sparse_loss += l
        return F.log_softmax(self.fc(out), dim=-1), sparse_loss
    
    
# mtlv2
class scGAT_MTv2(nn.Module):
    def __init__(self, dim_in, scgat_dim_out, tabnet_dim_out):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_in) 
        self.scgat = scGAT4multitasking(dim_in, scgat_dim_out)
        self.tabnet = TabNet_noinitBN(15125, 32, 
                                      N_d=64, N_a=64,
                                      n_shared=2, n_ind=2,
                                      n_steps=4, relax=1.2, vbs=128)
        
    def forward(self, data):
        x, edge_index = data.x[:, :-32], data.edge_index
        x = self.fc1(x)
        yhat_1 = self.scgat(x, edge_index)
        yhat_2, sparse_loss = self.tabnet(x)
        return yhat_1, yhat_2, sparse_loss
    
class scGAT_MTv2_LN(nn.Module):
    def __init__(self, dim_in, scgat_dim_out, tabnet_dim_out):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_in) 
        self.LN = nn.LayerNorm(dim_in)
        self.scgat = scGAT4multitasking(dim_in, scgat_dim_out)
        self.tabnet = TabNet_noinitBN(15125, 32, 
                                      N_d=64, N_a=64,
                                      n_shared=2, n_ind=2,
                                      n_steps=4, relax=1.2, vbs=128)
        
    def forward(self, data):
        x, edge_index = data.x[:, :-32], data.edge_index
        x = self.LN(self.fc1(x))
        yhat_1 = self.scgat(x, edge_index)
        yhat_2, sparse_loss = self.tabnet(x)
        return yhat_1, yhat_2, sparse_loss
    
class scGAT_MTv2_PReLU(nn.Module):
    def __init__(self, dim_in, scgat_dim_out, tabnet_dim_out):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_in) 
        self.prelu = nn.PReLU()
        self.scgat = scGAT4multitasking(dim_in, scgat_dim_out)
        self.tabnet = TabNet_noinitBN(15125, 32, 
                                      N_d=64, N_a=64,
                                      n_shared=2, n_ind=2,
                                      n_steps=4, relax=1.2, vbs=128)
        
    def forward(self, data):
        x, edge_index = data.x[:, :-32], data.edge_index
        x = self.prelu(self.fc1(x))
        yhat_1 = self.scgat(x, edge_index)
        yhat_2, sparse_loss = self.tabnet(x)
        return yhat_1, yhat_2, sparse_loss
    
class scGAT_MTv2_tanh(nn.Module):
    def __init__(self, dim_in, scgat_dim_out, tabnet_dim_out):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_in) 
        self.tanh = nn.Tanh()
        self.scgat = scGAT4multitasking(dim_in, scgat_dim_out)
        self.tabnet = TabNet_noinitBN(15125, 32, 
                                      N_d=64, N_a=64,
                                      n_shared=2, n_ind=2,
                                      n_steps=4, relax=1.2, vbs=128)
        
    def forward(self, data):
        x, edge_index = data.x[:, :-32], data.edge_index
        x = self.tanh(self.fc1(x))
        yhat_1 = self.scgat(x, edge_index)
        yhat_2, sparse_loss = self.tabnet(x)
        return yhat_1, yhat_2, sparse_loss
        
        
class scGAT4guides(nn.Module):
    '''NOTE: this is also the modification for Captum.
    
    '''
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.gat1 = GATConv(dim_in, out_channels=8,
                            heads=8, concat=True, negative_slope=0.2,
                            dropout=0.4, bias=True)
        self.gat2 = GATConv(8*8, dim_out,
                            heads=8, concat=False, negative_slope=0.2,
                            dropout=0.4, bias=True)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    
class scGAT_PReLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.gat1 = GATConv(dim_in, out_channels=8,
                            heads=8, concat=True, negative_slope=0.2,
                            dropout=0.4, bias=True)
        self.gat2 = GATConv(8*8, dim_out,
                            heads=8, concat=False, negative_slope=0.2,
                            dropout=0.4, bias=True)
        self.prelu = nn.PReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = self.prelu(x)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
class scGAT_nodropout(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.gat1 = GATConv(dim_in, out_channels=8,
                            heads=8, concat=True, negative_slope=0.2,
                            dropout=0.0, bias=True)
        self.gat2 = GATConv(8*8, dim_out,
                            heads=8, concat=False, negative_slope=0.2,
                            dropout=0.0, bias=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)
    

class scGAT_skipcat(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.gat1 = GATConv(dim_in, out_channels=8, 
                            heads=8, concat=True, 
                            negative_slope=0.2, 
                            dropout=0.4, bias=True)
        self.gat2 = GATConv(dim_in+8*8, out_channels=8, 
                            heads=8, concat=True, 
                            negative_slope=0.2, 
                            dropout=0.4, bias=True)
        self.gat3 = GATConv(8*8, out_channels=dim_out, 
                            heads=8, concat=False, 
                            negative_slope=0.2, 
                            dropout=0.4, bias=True)
        
    def forward(self, data):
        x_in, edge_index = data.x, data.edge_index 
        x = self.gat1(x_in, edge_index)
        x = F.elu(x) # F.elu(x) instead?
        x = self.gat2(torch.cat((x_in, x), 1), edge_index)
        x = F.elu(x)
        x = self.gat3(x, edge_index)
        return F.log_softmax(x, dim=1)
    
# TODO: scGAT_skipcat_PReLU
class scGAT_skipcat_nodropout_PReLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.gat1 = GATConv(dim_in, out_channels=8, 
                            heads=8, concat=True, 
                            negative_slope=0.2, 
                            dropout=0.0, bias=True)
        self.gat2 = GATConv(dim_in+8*8, out_channels=8, 
                            heads=8, concat=True, 
                            negative_slope=0.2, 
                            dropout=0.0, bias=True)
        self.gat3 = GATConv(8*8, out_channels=dim_out, 
                            heads=8, concat=False, 
                            negative_slope=0.2, 
                            dropout=0.0, bias=True)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        
    def forward(self, data):
        x_in, edge_index = data.x, data.edge_index 
        x = self.gat1(x_in, edge_index)
        x = self.prelu1(x) # F.elu(x) instead?
        x = self.gat2(torch.cat((x_in, x), 1), edge_index)
        x = self.prelu2(x)
        x = self.gat3(x, edge_index)
        return F.log_softmax(x, dim=1)

class scGAT(nn.Module):
    # v3, updated 210423
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.gat1 = GATConv(dim_in, out_channels=8, 
                            heads=8, concat=True, 
                            negative_slope=0.2, 
                            dropout=0.4, bias=True)
        self.gat2 = GATConv(dim_in+8*8, out_channels=8, 
                            heads=8, concat=True, 
                            negative_slope=0.2, 
                            dropout=0.4, bias=True)
        self.gat3 = GATConv(8*8, out_channels=dim_out, 
                            heads=8, concat=False, 
                            negative_slope=0.2, 
                            dropout=0.4, bias=True)
        
    def forward(self, data):
        x_in, edge_index = data.x, data.edge_index 
        x = self.gat1(x_in, edge_index)
        x = F.elu(x) # F.elu(x) instead?
        x = self.gat2(torch.cat((x_in, x), 1), edge_index)
        x = F.elu(x)
        x = self.gat3(x, edge_index)
        return F.log_softmax(x, dim=1)
    
# scGAT.v3
class scGAT_customforward(nn.Module):
    # forward for Captum and multi-task learning
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.gat1 = GATConv(dim_in, out_channels=8, 
                            heads=8, concat=True, 
                            negative_slope=0.2, 
                            dropout=0.4, bias=True)
        self.gat2 = GATConv(dim_in+8*8, out_channels=8, 
                            heads=8, concat=True, 
                            negative_slope=0.2, 
                            dropout=0.4, bias=True)
        self.gat3 = GATConv(8*8, out_channels=dim_out, 
                            heads=8, concat=False, 
                            negative_slope=0.2, 
                            dropout=0.4, bias=True)
        
    def forward(self, x_in, edge_index):
        x = self.gat1(x_in, edge_index)
        x = F.elu(x) 
        x = self.gat2(torch.cat((x_in, x), 1), edge_index)
        x = F.elu(x)
        x = self.gat3(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    
# mtl.v3 (uses scGAT.v3 and TabNet_noBN)
class scGAT_MTL(nn.Module):
    # v3
    def __init__(self, dim_in, scgat_dim_out, tabnet_dim_out, 
                 tabnetv='noBN', shared_activation=None):
        '''Multitask learning for single-cell data, combining metadata label prediction and 
             perturbation prediction based on gene expression.
        
        Arguments:
          tabnetv (str): (optional, Default='noBN') select which variant of TabNet to use. 
            Options are 'noBN' or 'LN'
          shared_activation (str): (optional, Default=None) select the activation that comes after the linear
            projection of input that allows for hidden-rep sharing between tasks. Options None or 'PReLU'         
        '''
        super().__init__()
        self.shared_activation = shared_activation
        
        self.fc1 = nn.Linear(dim_in, dim_in) 
        if self.shared_activation is not None and self.shared_activation=='PReLU':
            self.prelu = nn.PReLU()
        self.scgat = scGAT_customforward(dim_in, scgat_dim_out)
        if tabnetv=='noBN':
            self.tabnet = TabNet_noBN(15125, 32, 
                                       N_d=64, N_a=64,
                                       n_shared=4, n_ind=4,
                                       n_steps=4, relax=1.2)
        elif tabnetv=='LN':
            self.tabnet = TabNet_LN(15125, 32, 
                                    N_d=64, N_a=64,
                                    n_shared=4, n_ind=4,
                                    n_steps=4, relax=1.2)
        else:
            print('Invalid TabNet model.')
        
    def forward(self, data):
        x, edge_index = data.x[:, :-32], data.edge_index
        if self.shared_activation is None:
            x = self.fc1(x)
        elif self.shared_activation is not None and self.shared_activation=='PReLU':
            x = self.prelu(self.fc1(x))
        else:
            print('Invalid activation for shared hidden representation.')
        yhat_1 = self.scgat(x, edge_index)
        yhat_2, sparse_loss = self.tabnet(x)
        return yhat_1, yhat_2, sparse_loss


# inference
class DecisionStep_noBN_inference(nn.Module):
    def __init__(self, input_dim, N_d, N_a, shared, n_independent, gamma_relaxation, eps=1e-10):
        super().__init__()
        self.feat_transformer = FeatureTransformer_noBN(input_dim, N_d + N_a, shared=shared, n_independent=n_independent)
        self.attn_transformer =  AttentiveTransformer_noBN(N_a, input_dim, gamma_relaxation)
        self.eps = eps
        
    def forward(self, x, a, priors):
        mask = self.attn_transformer(a, priors)
        sparse_loss = ((-1) * mask * torch.log(mask + self.eps)).mean()
        x = self.feat_transformer(x * mask)
        return x, sparse_loss, mask
    
class TabNet_noBN_inference(nn.Module):
    def __init__(self, input_dim, output_dim,
                 N_d=256, N_a=256,
                 n_shared=2, n_ind=2,
                 n_steps=5, relax=1.2):
        super().__init__()
        
        if n_shared>0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(input_dim, 2*(N_d + N_a)))
            
            for x in range(n_shared-1):
                self.shared.append(nn.Linear(N_d + N_a, 2*(N_d + N_a)))
        else:
            self.shared = None
            
        self.first_step = FeatureTransformer_noBN(input_dim, N_d + N_a, shared=self.shared, n_independent=n_ind) 
        self.steps = nn.ModuleList()
        
        for x in range(n_steps - 1):
            self.steps.append(DecisionStep_noBN_inference(input_dim, N_d, N_a, self.shared, n_ind, relax))
        
        self.fc = nn.Linear(N_d, output_dim)
        self.N_d = N_d
        
    def forward(self, x):
        all_masks = []
        x_a = self.first_step(x)[:,self.N_d:]
        sparse_loss = torch.zeros(1).to(x.device)
        out = torch.zeros(x.size(0),self.N_d).to(x.device)
        priors = torch.ones(x.shape).to(x.device)
        for step in self.steps:
            x_te, l, mask = step(x, x_a, priors)
            d_i = x_te[:, :self.N_d]
            all_masks.append((d_i, mask))
            out += F.relu(d_i)
            x_a = x_te[:, self.N_d:]
            sparse_loss += l
        return F.log_softmax(self.fc(out), dim=-1), sparse_loss, all_masks
    
class scGAT_MTL_inference(nn.Module):
    # v3.2 ## for interpretability
    def __init__(self, dim_in, scgat_dim_out, tabnet_dim_out, 
                 tabnetv='noBN', shared_activation=None):
        '''Multitask learning for single-cell data, combining metadata label prediction and 
             perturbation prediction based on gene expression.
        
        Arguments:
          tabnetv (str): (optional, Default='noBN') select which variant of TabNet to use. 
            Options are 'noBN' or 'LN'
          shared_activation (str): (optional, Default=None) select the activation that comes after the linear
            projection of input that allows for hidden-rep sharing between tasks. Options None or 'PReLU'         
        '''
        super().__init__()
        self.shared_activation = shared_activation
        
        self.fc1 = nn.Linear(dim_in, dim_in) 
        if self.shared_activation is not None and self.shared_activation=='PReLU':
            self.prelu = nn.PReLU()
        self.scgat = scGAT_customforward(dim_in, scgat_dim_out)
        if tabnetv=='noBN':
            self.tabnet = TabNet_noBN_inference(15125, 32, 
                                       N_d=64, N_a=64,
                                       n_shared=4, n_ind=4,
                                       n_steps=4, relax=1.2)
        elif tabnetv=='LN':
            # have not yet collecte4d x_a per step in a list for output
            print('Not implemented.')
        else:
            print('Invalid TabNet model.')
        
    def forward(self, data):
        x, edge_index = data.x[:, :-32], data.edge_index
        if self.shared_activation is None:
            x = self.fc1(x)
        elif self.shared_activation is not None and self.shared_activation=='PReLU':
            x = self.prelu(self.fc1(x))
        else:
            print('Invalid activation for shared hidden representation.')
        yhat_1 = self.scgat(x, edge_index)
        yhat_2, sparse_loss, all_masks = self.tabnet(x)
        return yhat_1, yhat_2, sparse_loss, all_masks
