using MAT

using MultivariateStats
using GLMakie
using ProperOrthogonalDecomposition
include("utils.jl")
include("decompositions.jl")

data = matread("../spod/jet_data/jetLES.mat")
p = data["p"]
# switches the spatial dimensions to the leading dimension
p = permutedims(p, [2, 3, 1])

# The fastica algorithm used by the packages doesn't seem to converge
# reliably. Convergence seems to depend on the initialization, so we try 
# it until it works
ica = nothing
i_indices = nothing
j_indices = nothing
k = 1
while isnothing(ica)
    try 
        global ica, i_indices, j_indices = compute_ica(p, 20, 1, 1)
    catch
        @warn "ica not converged in $k-th attempt"
        global k = k + 1
    end
    if k >20 break end
end

# This becomes a three-way tensor; the k-th component is given by components[:, :, k]
# now included in the ica implementation
# independent_components = reshape(ica, (length(i_indices), length(j_indices), size(ica, 2)))

# Plots the independent components
figs_per_row = 4
fig_ica = Figure()
for (k, ind) in enumerate(CartesianIndices((figs_per_row, size(ica, 3)))[:])
    if k <= size(ica, 3)
        heatmap(fig_ica[ind[2], ind[1]], ica[:, :, k])
    end
end
display(fig_ica)

pod, i_indices, j_indices = compute_pod_svd(p, 1, 1)
# This becomes a three-way tensor; the k-th component is given by components[:, :, k]
#proper_components = reshape(pod, (length(i_indices), length(j_indices), size(pod, 2)))
pod = pod[:, :, 1:size(ica, 3)]

# Plots the independent components
figs_per_row = 4
fig_pod = Figure()
for (k, ind) in enumerate(CartesianIndices((figs_per_row, size(ica, 3)))[:])
    if k <= size(ica, 3)
        heatmap(fig_pod[ind[2], ind[1]], pod[:, :, k])
    end
end
display(fig_pod)
