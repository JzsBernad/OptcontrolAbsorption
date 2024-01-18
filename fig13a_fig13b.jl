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

# Test
redo = false
g_samples = 90
P_s_samples = 89
g_s::Vector{Float64} = range(0.1, stop=1.0, length=g_samples)
target_P_s_s::Vector{Float64} = range(0.1, stop=0.99, length=P_s_samples)
omega = 10.0
T = 1.0
min_T = 0.01
max_T = 10000.0
reltol = 1e-10
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
    dict = Dict("T_target" => T_target, "Zsolt_T_target" => Zsolt_T_target, "kappa_opt" => kappa_opt, "g_s" => g_s, "target_P_s_s" => target_P_s_s)
    save(filename, dict)
end

# heatmap of times to P_s logarithmic, with contour lines
# load data from dict
T_target = dict["T_target"]
Zsolt_T_target = dict["Zsolt_T_target"]
g_s = dict["g_s"]
target_P_s_s = dict["target_P_s_s"]
# make color axis logarithmic 
# calculate loagrithmic version of T_target and ticks
T_target_log = log10.(T_target)
exponents = Vector{Int64}(Int64(round(minimum(T_target_log))):Int64(round(maximum(T_target_log))))
ticks = [10.0^i for i in exponents]
tick_labels = [latexstring(raw"10^{" * string(i) * "}") for i in exponents]
plt = heatmap(g_s, target_P_s_s, T_target_log', xlabel=L"g_\mathrm{ens}/2\pi\;[\mathrm{MHz}]", ylabel=L"P_{s, \mathrm{tar}}", colorbar_title=" \n" * L"\log_{10}[T(\mu s)]", clim=(minimum(T_target_log), maximum(T_target_log)), colorbar_ticks=(exponents, tick_labels), right_margin=5Plots.mm)
# add contour lines
#contour!(plt, g_s, target_P_s_s, T_target_log', levels=[-1.0, 0.0, 1.0, 2.0, 3.0], linewidth=1, clabels=true, color=:black, label="")
plt2pdf(plt, "fig13a") #Time_2_P_s_heatmap_optim

# Speedup
speedup = Zsolt_T_target ./ T_target
plt = heatmap(g_s, target_P_s_s, speedup', xlabel=L"g_\mathrm{ens}/2\pi\;[\mathrm{MHz}]", ylabel=L"P_{s, \mathrm{tar}}", colorbar_title=" \n" * L"T_\mathrm{Gauss}/T_\mathrm{Optim}", right_margin=10Plots.mm, clim=(minimum(speedup), maximum(speedup)))
plt2pdf(plt, "fig13b") #Speedup_heatmap_optim