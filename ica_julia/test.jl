using MultivariateStats
using GLMakie
using Statistics
using Random
using Distributions
using LinearAlgebra

include("utils.jl")
include("decompositions.jl")

n_latent = 1
n_out = 1
n_samples = 10000
s_noise = 0.00001
sigma = 0.1

# Atest = rand(n_out,n_latent)
Atest = Matrix(I,n_out,n_latent)
data = Atest*rand(Laplace(0,sigma),n_latent,n_samples) + randn(n_out,n_samples)*s_noise

ica = fit(ICA, data, n_latent; maxiter=100000, tol=0.001)
