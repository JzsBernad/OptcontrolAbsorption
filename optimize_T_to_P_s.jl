using Distributed
# how many threads does processor have
ncores = Sys.CPU_THREADS
addprocs(ncores - nprocs() - 1)
println("Processes: ", nprocs())

# import scripts_photon_catching.jl
@everywhere include("scripts_photon_catching_full.jl")


# Test
redo = false
g_samples = 90
P_s_samples = 49
g_s::Vector{Float64} = range(0.1, stop=1.0, length=g_samples)
target_P_s_s::Vector{Float64} = range(0.1, stop=0.99, length=P_s_samples)
omega = 10.0
T = 1.0
min_T = 0.01
max_T = 10000.0
reltol = 1e-9
n = 3
n_zsolt = 10
param = Parameters(1.0, 1.0, omega, T)
coeff = ones(n)

filename = joinpath("Results", "time_2_" * params_to_str("P_s", target_P_s_s) * "_n_" * string(n) * "_" * string(n_zsolt) * "_o_" * string(omega) * "_" * params_to_str("g", g_s) * ".jld")
if isfile(filename) && !redo
    println("Loading cached results")
    dict = load(filename)
else
    println("Calculating results")
    T_target = zeros(length(g_s), length(target_P_s_s))
    Zsolt_T_target = zeros(length(g_s), length(target_P_s_s))
    kappa_opt = zeros(length(g_s), length(target_P_s_s))
    for (i, target_P_s) in enumerate(target_P_s_s)
        println("Calculating for target P_s: ", target_P_s, " (", i, "/", length(target_P_s_s), ")")
        T_target[:, i], Zsolt_T_target[:, i], kappa_opt[:, i] = Both_T_g_to_target(target_P_s, n, n_zsolt, g_s, coeff, param, min_T, max_T, reltol=reltol)
    end
    println("Done Calculating")
    dict = Dict("T_target" => T_target, "Zsolt_T_target" => Zsolt_T_target, "kappa_opt" => kappa_opt, "g_s" => g_s, "target_P_s_s" => target_P_s_s)
    save(filename, dict)
end
