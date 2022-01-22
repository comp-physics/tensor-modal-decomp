# Input variables:
# p: The input three-way tensor, first two dimensions are spatial, third is time
# k: The number of components
# ds_i: The offset for subsampling on first dim
# ds_j: The offset for subsampling on second dim

function compute_ica(p::AbstractArray{<:Any,3}, k, ds_i=1, ds_j=1)
    i_inds = 1 : ds_i : size(p, 1)
    j_inds = 1 : ds_j : size(p, 2)
    p = p[i_inds, j_inds, :]
    @show size(p)
    p = reshape(p, (:, size(p, 3)))
    @show size(p)
    modes = fit(ICA, p, k; maxiter=100000, tol=0.00001).W
    modes = reshape(modes, (length(i_inds), length(j_inds), size(modes, 2)))
    return modes, i_inds, j_inds
end 

function compute_pod_svd(p::AbstractArray{<:Any, 3}, ds_i=1, ds_j=1)
    i_inds = 1 : ds_i : size(p, 1)
    j_inds = 1 : ds_j : size(p, 2)
    p = p[i_inds, j_inds, :]
    p = reshape(p, (:, size(p, 3)))
    @show size(p)
    # returns a number of modes times number of dof matrix
    modes = POD(p)[1].modes
    @show size(modes)
    # reshape the modes again so that they return the 
    modes = reshape(modes, (length(i_inds), length(j_inds), size(modes, 2)))
    return modes, i_inds, j_inds
end

# STILL NOT FINISHED! 
function compute_spod_svd(p::AbstractArray{<:Any, 3}, ds_i=1, ds_j=1)
    i_inds = 1 : ds_i : size(p, 1)
    j_inds = 1 : ds_j : size(p, 2)
    p = p[i_inds, j_inds, :]
    # Applying time-domain fft
    p = fft(p, (1))
    p = Matrix(transpose(reshape(p, (size(p, 1), :))))
    @show size(p)
    modes = POD(fftp)[1].modes
    # reshape the modes again so that they return the 
    fftmodes = reshape(modes, (length(i_inds), length(j_inds), size(modes, 2)))
    modes = ifft(fftmodes, (1,2))
    return modes, i_inds, j_inds
end
