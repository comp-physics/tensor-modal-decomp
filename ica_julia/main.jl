using MAT
using MultivariateStats
using GLMakie
include("utils.jl")
include("decompositions.jl")

data = matread("../spod/jet_data/jetLES.mat")

p = data["p"]

# The fastica algorithm used by the packages doesn't seem to converge
# reliably. Convergence seems to depend on the initialization, so we try 
# it until it works
ica = nothing
i_indices = nothing
j_indices = nothing
k = 1
while isnothing(ica)
    try 
        ica, i_indices, j_indices = compute_ica(p, 10, 3, 3)
    catch
        @warn "ica not converged in $k-th attempt"
        k = k + 1
    end
end


# This becomes a three-way tensor; the k-th component is given by components[:, :, k]
components = reshape(ica.W, (length(i_indices), length(j_indices), size(ica.W, 2)))

# Plots the independent components
figs_per_row = 4
fig = Figure()
for (k, ind) in enumerate(CartesianIndices((figs_per_row, size(components, 3)))[:])
    if k <= size(components, 3)
        @show (ind[2], ind[1])
        heatmap(fig[ind[2], ind[1]], components[:, :, k])
    end
end
display(fig)
