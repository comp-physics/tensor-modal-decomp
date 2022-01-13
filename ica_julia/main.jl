using MAT

using MultivariateStats
using GLMakie
using ProperOrthogonalDecomposition
using Statistics

include("utils.jl")
include("decompositions.jl")

data = matread("../spod/jet_data/jetLES.mat")
p = data["p"]
x = data["x"][1,:]
r = data["r"][:,1]

ifica = true
ifpod = false
ifplotdata = false

# Remove mean
p = p .- mean(p[:])

# switches the spatial dimensions to the leading dimension
p = permutedims(p, [2, 3, 1])

# Strides in i and j
ds_i = 1 
ds_j = 1

# Get arrays
i_arr = 1 : ds_i : size(p, 1)
j_arr = 1 : ds_j : size(p, 2)

i_length = length(i_arr)
j_length = length(j_arr)

# Number of POD/ICA components to compute and/or plot
num_comp = 16

if ifplotdata
    fig_data = heatmap(x[j_arr], r[i_arr], Matrix(p[i_arr, j_arr, 10]'),colormap =:bluesreds)
    display(fig_data)
    save("figures/data.png",fig_data)
end


if ifica
    # The fastica algorithm used by the packages doesn't seem to converge
    # reliably. Convergence seems to depend on the initialization, so we try 
    # it until it works

    # When to stop ICA if it never gets a valid one
    stop_ica_it = 20

    ica = nothing
    i_indices = nothing
    j_indices = nothing
    k = 1
    while isnothing(ica)
        try 
            global ica, i_indices, j_indices = compute_ica(p, num_comp, ds_i, ds_j)
        catch
            @warn "ica not converged in $k-th attempt"
            global k = k + 1
        end
        if k > stop_ica_it break end
    end

    # This becomes a three-way tensor; the k-th component is given by components[:, :, k]
    # now included in the ica implementation
    # independent_components = reshape(ica, (length(i_indices), length(j_indices), size(ica, 2)))

    # Plots the independent components
    figs_per_row = 4
    fig_ica = Figure()
    for (k, ind) in enumerate(CartesianIndices((figs_per_row, num_comp))[:])
        if k <= num_comp
            heatmap(fig_ica[ind[2], ind[1]], x[j_arr], r[i_arr], Matrix(ica[i_arr, j_arr, k]'),colormap =:bluesreds)
        end
    end
    display(fig_ica)
    save("figures/ica.png",fig_ica)
end

if ifpod
    pod, i_indices, j_indices = compute_pod_svd(p, ds_i, ds_j)
    # This becomes a three-way tensor; the k-th component is given by components[:, :, k]
    #proper_components = reshape(pod, (length(i_indices), length(j_indices), size(pod, 2)))
    pod = pod[:, :, 1:num_comp]

    # Plots the independent components
    figs_per_row = 4
    fig_pod = Figure()
    for (k, ind) in enumerate(CartesianIndices((figs_per_row, num_comp))[:])
        if k <= num_comp
            heatmap(fig_pod[ind[2], ind[1]], x[j_arr], r[i_arr], Matrix(pod[i_arr, j_arr, k]'),colormap =:bluesreds)
        end
    end
    display(fig_pod)
    save("figures/pod.png",fig_pod)
end
