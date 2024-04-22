import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedAdaptiveMixing(nn.Module):
    def __init__(self, num_stages, kernel_sizes, kernel_groups, mix_out_proj_ranks, 
                 in_dim, in_points, n_groups, query_dim=None, out_dim=None, out_points=None, sampling_rate=None):
        super(FactorizedAdaptiveMixing, self).__init__()
        self.num_stages = num_stages
        self.kernel_sizes = kernel_sizes
        self.kernel_groups = kernel_groups
        self.mix_out_proj_ranks = mix_out_proj_ranks
        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim
        sampling_rate = sampling_rate if sampling_rate is not None else 1

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points//sampling_rate
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = in_dim//n_groups
        self.eff_out_dim = out_dim//n_groups        

        self.ml_parameters = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.n_groups*k_group, self.eff_in_dim, k_size)) 
            for k_size, k_group in zip(self.kernel_sizes, self.kernel_groups)]) 
        
        self.mr_parameters = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.n_groups*k_group, k_size, self.eff_out_dim))
            for k_size, k_group in zip(self.kernel_sizes, self.kernel_groups)]) 

        self.sl_parameters = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.n_groups*k_group, self.out_points, k_size))
            for k_size, k_group in zip(self.kernel_sizes, self.kernel_groups)]) 

        self.sr_parameters = nn.ParameterList([
        nn.Parameter(torch.Tensor(self.n_groups*k_group, k_size, self.in_points))
            for k_size, k_group in zip(self.kernel_sizes, self.kernel_groups)]) 

        self.common_parameter_generator = nn.Linear(self.query_dim, 
            self.n_groups*(self.eff_in_dim * self.eff_out_dim + self.in_points * self.out_points))

        self.private_parameter_generator = nn.ModuleList([
            nn.Linear(self.query_dim, 2 * self.n_groups * k_group * k_size**2)
                for k_size,k_group in zip(self.kernel_sizes, self.kernel_groups)])

        self.base_out_proj = nn.Linear(self.eff_out_dim*self.out_points*self.n_groups, 
                                       self.query_dim, bias=False)   

        self.out_weight_add_A = nn.ParameterList([
                nn.Parameter(torch.Tensor(rank, self.eff_out_dim*self.out_points*self.n_groups))
                for rank in self.mix_out_proj_ranks])

        self.out_weight_add_B = nn.ParameterList([
                nn.Parameter(torch.zeros(self.query_dim, rank))
                for rank in self.mix_out_proj_ranks])

        self.out_weight_bias = nn.ParameterList([      
            nn.Parameter(torch.zeros(self.query_dim))                              
            for _ in range(num_stages)])

        self.out_norm = nn.ModuleList([
            nn.LayerNorm(query_dim)
            for _ in range(num_stages)])

        self.act = nn.ReLU(inplace=True)

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.common_parameter_generator.weight)
        
        for private_parameter_generator in self.private_parameter_generator:
            nn.init.zeros_(private_parameter_generator.weight)
            nn.init.zeros_(private_parameter_generator.bias)            

        for ml_parameters in self.ml_parameters:
            nn.init.kaiming_uniform_(ml_parameters)
        for mr_parameters in self.mr_parameters:
            nn.init.kaiming_uniform_(mr_parameters)
        for sl_parameters in self.sl_parameters:
            nn.init.kaiming_uniform_(sl_parameters)
        for sr_parameters in self.sr_parameters:        
            nn.init.kaiming_uniform_(sr_parameters)
        
        for out_weight_add_A in self.out_weight_add_A:
            nn.init.kaiming_uniform_(out_weight_add_A)

        # # import math; nn.init.kaiming_uniform_(a=math.sqrt(5))   
        # bound = 1 / math.sqrt(self.base_out_proj.weight.shape[1])
        # for out_weight_bias in self.out_weight_bias:
        #     nn.init.uniform_(out_weight_bias, -bound, bound)
       

    def forward(self, stage, x, query):
        B, N, G, P, C = x.size()
        # batch, num_query, group, point, channel
        assert G == self.n_groups
        # assert C*g == self.in_dim

        # query: B, N, C
        # x: B, N, G, Px, Cx

        '''generate mixing parameters'''
        Mc, Sc = self.common_parameter_generator(query).reshape(B*N, G, -1).split(
            [self.eff_in_dim * self.eff_out_dim, self.in_points * self.out_points], 2)

        Mc = Mc.reshape(
            B*N, G, 1, self.eff_in_dim, self.eff_out_dim)
        Sc = Sc.reshape(
            B*N, G, 1, self.out_points, self.in_points)

        k_size = self.kernel_sizes[stage]
        k_group = self.kernel_groups[stage]
                
        params = self.private_parameter_generator[stage](query)
        params = params.reshape(B*N, 2, G*k_group, k_size, k_size)
        
        ml = self.ml_parameters[stage][None]
        mz = params[:, 0, ...]
        mr = self.mr_parameters[stage][None]
        
        sl = self.sl_parameters[stage][None]
        sz = params[:, 1, ...]
        sr = self.sr_parameters[stage][None]
        
        Mp = ml @ mz @ mr
        Sp = sl @ sz @ sr
        
        Mp = Mp.view(B*N, G, k_group, self.eff_in_dim, self.eff_out_dim)
        Sp = Sp.view(B*N, G, k_group, self.out_points, self.in_points)
        
        M = torch.cat([Mc, Mp], dim=2).sum(axis=2)
        S = torch.cat([Sc, Sp], dim=2).sum(axis=2)

        out = x.reshape(B*N, G, P, C)

        '''adaptive channel mixing'''
        out = torch.matmul(out, M)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''adaptive spatial mixing'''
        out = torch.matmul(S, out)  # implicitly transpose and matmul
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''linear transfomation to query dim'''        
        _A , _B = self.out_weight_add_A[stage], self.out_weight_add_B[stage]
        base_weight = self.base_out_proj.weight
        bias = self.out_weight_bias[stage]
        
        out = out.reshape(B, N, -1)
        out = F.linear(out, base_weight + _B @ _A, bias)        
        out = self.out_norm[stage](query + out)

        return out