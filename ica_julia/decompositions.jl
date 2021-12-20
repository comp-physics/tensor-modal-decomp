# Input variables:
# p: The input three-way tensor 
# k: The number of components
# ds_i: The offset for subsampling on first dim
# ds_j: The offset for subsampling on second dim

function compute_ica(p::AbstractArray{<:Any,3}, k, ds_i=1, ds_j=1)
    i_inds = 1 : ds_i : size(p, 2)
    j_inds = 1 : ds_j : size(p, 3)
    p = p[:, i_inds, j_inds]
    p = Matrix(transpose(reshape(p, (size(p, 1), :))))
    return fit(ICA, p, k, maxiter=10000).W, i_inds, j_inds
end 

function compute_pod_svd(p::AbstractArray{<:Any, 3}, ds_i=1, ds_j=1)
    i_inds = 1 : ds_i : size(p, 2)
    j_inds = 1 : ds_j : size(p, 3)
    p = p[:, i_inds, j_inds]
    p = Matrix(transpose(reshape(p, (size(p, 1), :))))
    return POD(p)[1].modes, i_inds, j_inds
end
