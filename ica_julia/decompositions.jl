# Input variables:
# p: The input three-way tensor 
# k: The number of components
# ds_i: The offset for subsampling on first dim
# ds_j: The offset for subsampling on second dim

function compute_ica(p::AbstractArray{<:Any,3}, k, ds_i=1, ds_j=1)
    i_inds = 1 : ds_i : size(p, 2)
    j_inds = 1 : ds_j : size(p, 3)
    p = p[:, i_inds, j_inds]
    @show size(p)
    p = transpose(reshape(p, (5000, :)))
    return fit(ICA, p, k, maxiter=10000), i_inds, j_inds
end 
