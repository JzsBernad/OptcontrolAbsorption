# With parallelisation
using Distributed
# how many threads does processor have
ncores = Sys.CPU_THREADS
addprocs(ncores - nprocs() - 6)
println("Processes: ", nprocs())

# import scripts_photon_catching.jl
@everywhere include("scripts_photon_catching_full.jl")
# Test
#do_gauss = true
do_achievable = true
redo = false
#g_samples, k_samples, T_samples = 10, 10, 10
g_samples, k_samples, T_samples = 100, 100, 100

if do_achievable
    g_s::Vector{Float64} = range(0.01, stop=1.0, length=g_samples)
    kappa_s::Vector{Float64} = range(0.01, stop=1.0, length=k_samples)
    T_s::Vector{Float64} = range(0.5, stop=50.0, length=T_samples)
else ### large g, but only short T
    g_s::Vector{Float64} = range(0.1, stop=4.0, length=g_samples)
    kappa_s::Vector{Float64} = range(0.1, stop=6.0, length=k_samples)
    T_s::Vector{Float64} = range(0.25, stop=20.0, length=T_samples)
end

omega = 10.0
param = Parameters(1.0, 1.0, omega, 10.0)
n = 3
n_zsolt = 11
reltol = 1e-7
max_delta = 100.0
target_step = 1
coeff = ones(n)

for do_gauss in [true, false]
    if !do_gauss
        filename = functionname(n, g_s, kappa_s, T_s)
    else
        filename = functionname(n_zsolt, g_s, kappa_s, T_s; prefix="gaussian_pulses_n_")
    end
    # if filename exists load it, else optimize
    if isfile(filename) && !redo
        dict = load(filename, "dict")
        println("Loaded")
    else
        if !isfile(filename)
            println("File does not exist, Optimizing")
        else
            println("Redoing optimization")
        end
        if !do_gauss
            dict = optimize_catching_photons_T_g_kappa(T_s, g_s, kappa_s, coeff, param, reltol, max_delta, target_step)
        else
            dict = gaussian_catch_photons_T_g_kappa(T_s, g_s, kappa_s, param, n_zsolt, reltol, max_delta, target_step)
        end
        println("Done")
        save(filename, "dict", dict)
        println("Saved")
    end
end