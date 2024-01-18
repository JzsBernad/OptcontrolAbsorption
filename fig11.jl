# To install packages (if needed)
# using Pkg 
# elem_list = ["Distributed", "LsqFit", "Formatting", "JLD", "LaTeXStrings", "Optimization", "Ipopt", "OptimizationOptimJL", "OptimizationOptimisers", "ForwardDiff", "Zygote", "BenchmarkTools", "QuadGK", "Roots", "SharedArrays", "Plots", "PGFPlotsX"]
# for elem in elem_list
#   Pkg.add(elem)
# end

# With parallelisation
using Distributed
# how many threads does processor have
ncores = Sys.CPU_THREADS
addprocs(max(ncores - nprocs() - 6, 1))
println("Processes: ", nprocs())

@everywhere include("scripts_photon_catching_full.jl")
@everywhere using LsqFit
@everywhere using Formatting


# a function that saves a plt to a file via pgfplotsx backend. then modifies the .tex file by removing comments at the beginning and adding a header, and begin and end document
function plt2pdf(plt, filename::String="filename"; dpi=96, width=10, height=8, preview::Bool=true, directory::String="Plots", header::String=raw"\documentclass[groupedaddress,amsmath,amssymb,amsfonts,nofootinbib,a4paper, 10pt]{standalone}" * "\n" * raw"\input{../tikz_header}")
    # width and height are in cm
    pixel_width = width / 2.54 * dpi
    pixel_height = height / 2.54 * dpi
    plot!(plt, framestyle=:box, size=(pixel_width, pixel_height))
    # get the extent of plt (xlims and ylims)
    x_lims = xlims(plt)
    y_lims = ylims(plt)
    # construct a box at the border
    box = [(x_lims[1], y_lims[1]), (x_lims[1], y_lims[2]), (x_lims[2], y_lims[2]), (x_lims[2], y_lims[1]), (x_lims[1], y_lims[1])]
    plot!(plt, box, color=:black, linewidth=1, label="")
    # plot the box
    plot!(plt, xlims=x_lims, ylims=y_lims)
    # create completed filename
    dir_name = joinpath(directory, filename)
    # if .tex missing add it
    if !endswith(dir_name, ".pdf")
        dir_name = dir_name * ".pdf"
    end
    # if directory missing create it
    if !isdir(directory)
        mkdir(directory)
    end
    # save plt to file, then load it
    savefig(plt, dir_name)
    if preview
        display(plt)
    end
end
function stringx(x, N=2)
    return sprintf1("%1.$(N)f", x)
end

# expand function into sinus basis functions
omega = 10.0
T = 10.0
g = 0.5
n = 3
n_zsolt = 10
max_delta = 100.0
reltol = 1e-9
kappa = 4 * g^2 / omega
param = Parameters(g, kappa, omega, T)
kappa_opt, P_s_val = find_optimal_k(param, n; max_delta=max_delta, reltol=reltol)
println("kappa_opt = ", kappa_opt)
println("Optimized Cooperativity: ", 4 * g^2 / (kappa_opt * omega))
param = Parameters(g, kappa_opt, omega, T)

kappa_g = 2 * pi / param.T

coeffs = expand_Zsolt2coeff_full(n_zsolt, kappa_g, param, reltol=reltol)
integral_cache = analytical_P_s_cache_integrals_full_gaussian(param, n_zsolt, max_delta, reltol)
constraint_matrix = create_harmonic_f_constraint_cache_full(n_zsolt, param, reltol)
# normalize coeffs
coeffs = normalize_coeff(coeffs, param.kappa, constraint_matrix)
Zsolt_P_s = P_s_cached(coeffs, integral_cache)
println("Zsolt_P_s: ", Zsolt_P_s)


integral_cache = analytical_P_s_cache_integrals_full(param, n, max_delta, reltol)
constraint_matrix = create_harmonic_f_constraint_cache_full(n, param, reltol)
coeff = ones(2)# deepcopy(coeffs)
opt_fun = x -> -P_s_cached_normalized(x, integral_cache, constraint_matrix, param.kappa)
opt_fun_diffed = OnceDifferentiable(opt_fun, coeff, autodiff=:forward)
results = optimize(opt_fun_diffed, coeff, BFGS())#, Optim.Options(callback=cb))
coeff_opt = normalize_coeff(Optim.minimizer(results), param.kappa, constraint_matrix)
optim_result = P_s_cached_normalized(coeff_opt, integral_cache, constraint_matrix, param.kappa)
println("Optimized P_s: ", optim_result)

# plot the pulse of coeffs
plt = plot()
t_arr = Vector(range(0.0, param.T, length=1000))
gauss_vals = full_pulse(t_arr, coeffs, param.T)
pulse_vals = full_pulse(t_arr, coeff_opt, param.T)
plot!(plt, t_arr, gauss_vals, label=L"\mathrm{Gaussian}")
plot!(plt, t_arr, pulse_vals, label=L"\mathrm{Optimal}")
## is the pulse normalized?
c2 = [0.0]
# add horizontal line to y = 0 to plt then display
plot!(plt, [0.0, param.T], [0.0, 0.0], color="black", label="", linestyle=:dash, xlims=(0.0, param.T), ylims=(min(0.0, minimum(pulse_vals)), maximum(gauss_vals) * 1.2))
plot!(plt, xlabel=L"t\; [\mu \mathrm{s}]", ylabel=L"f(t)\; [\mathrm{MHz}]")
title!(plt, "")
# save as pdf
plt2pdf(plt, "fig11") #optimized_pulse