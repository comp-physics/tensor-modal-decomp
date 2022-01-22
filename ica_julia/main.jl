using MAT

using MultivariateStats
using GLMakie
using ProperOrthogonalDecomposition
using Statistics
using Random

include("utils.jl")
include("decompositions.jl")

data = matread("../spod/jet_data/jetLES.mat")
p = data["p"]
x = data["x"][1,:]
r = data["r"][:,1]

ifica = true
ifpod = false
ifplotdata = false

# x = (1:length(x))*maximum(x)/length(x)
# r = (1:length(r))*maximum(r)/length(r)

# switches the spatial dimensions to the leading dimension
p = permutedims(p, [2, 3, 1])

# Strides in i (x) and j (r) and k (t)
ds_i = 1 
ds_j = 1
ds_k = 1

# Get arrays
i_arr = 1 : ds_i : size(p, 1)
j_arr = 1 : ds_j : size(p, 2)
k_arr = 1 : ds_k : size(p, 3)

i_length = length(i_arr)
j_length = length(j_arr)
k_length = length(k_arr)

# 
# p = p[:,:,k_arr]

# Remove mean
p = p .- mean(p[:])

# Normalize p
amp_p = mean(abs.(p))
p = p/amp_p

@show maximum(p)
@show minimum(p)

# Add inverted jet
size_fac = 1.0
num_subsamples = 2
# p = p[end:-1:1,end:-1:1,randperm(size(p,3))]*size_fac + p

# p = p[:,:,rand(randperm(size(p,3))[1:num_subsamples],size(p,3))] + 
#     p[end:-1:1,end:-1:1,rand(randperm(size(p,3))[1:num_subsamples],size(p,3))]

p1 = repeat(p[:,:,1234],1,1,size(p,3))
p2 = repeat(p[end:-1:1,end:-1:1,2345],1,1,size(p,3))

p1[:,:,randperm(size(p,3))[1:1000]] .= 0.0
p2[:,:,randperm(size(p,3))[1:1000]] .= 0.0
p = p1 + p2

# p = p[end:-1:1,end:-1:1,:]*size_fac

# Try adding random component 
# rand_mat = rand(size(p)...) * amp_p
# p = p + rand_mat

# Number of POD/ICA components to compute and/or plot
num_comp = 4

if ifplotdata
    println("Display Data")
    display_ts = 5000
    fig_data = heatmap(x[j_arr], r[i_arr], Matrix(p[i_arr, j_arr, display_ts]'),colormap =:summer)
    display(fig_data)
    save("figures/data.png",fig_data)
end


if ifica
    # The fastica algorithm used by the packages doesn't seem to converge
    # reliably. Convergence seems to depend on the initialization, so we try 
    # it until it works
    println("ICA")

    # When to stop ICA if it never gets a valid one
    stop_ica_it = 5

    ica = nothing
    i_indices = nothing
    j_indices = nothing
    k = 1

    while isnothing(ica) 
        # @code_warntype compute_ica(p, num_comp, ds_i, ds_j)
        @time try 
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
    println("POD")
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
