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
do_achievable = true
redo = false
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

# To calculate 
omega = 10.0
param = Parameters(1.0, 1.0, omega, 10.0)
n = 3
n_zsolt = 10
reltol = 1e-7
max_delta = 100.0
target_step = 100
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

# Plot Heatmap
filename = filename = functionname(n, g_s, kappa_s, T_s)
dict = load(filename, "dict")
filename_gauss = filename = functionname(n_zsolt, g_s, kappa_s, T_s; prefix="gaussian_pulses_n_")
dict_gauss = load(filename_gauss, "dict")

# find first T at which max value is 99% of max value
max_vals = [maximum(dict["objective"][i, :, :]) for i in 1:length(T_s)]
max_vals_gaussian = [maximum(dict_gauss["objective"][i, :, :]) for i in 1:length(T_s)]
#T_ind = findfirst(max_vals .>= 0.99)
T_ind = findmin(abs.(T_s .- 20.0))[2]
g_s::Vector{Float64} = dict["param"]["g"]
kappa_s::Vector{Float64} = dict["param"]["kappa"]

P_s_values = dict["objective"][T_ind, :, :]
P_s_values_gaussian = dict_gauss["objective"][T_ind, :, :]
# max values
max_val = max_vals[T_ind]
println("Max value: ", max_val)
max_val_gaussian = max_vals_gaussian[T_ind]
println("Max value gaussian: ", max_val_gaussian)

#### Optimization ###################
plt = heatmap(g_s, kappa_s, P_s_values', xlabel=L"g_\mathrm{ens}/2\pi\;[\mathrm{MHz}]", ylabel=L"\kappa/2\pi\;[\mathrm{MHz}]", clim=(0, 1), colorbar_title=" \n" * L"P_s", right_margin=5Plots.mm) #title=latexstring(raw"P_s(T="*string(T_s[T_ind])*raw"\mu s)"), 
contour!(plt, g_s, kappa_s, P_s_values', levels=[0.68, 0.87, 0.95, 0.99], linewidth=1, clabels=true, color=:black, label="")
plot!(plt, xlims=(minimum(g_s), maximum(g_s)), ylims=(minimum(kappa_s), maximum(kappa_s)))
g0, kappa0 = 0.350, 0.050
plot!(plt, [g0], [kappa0], seriestype=:scatter, markersize=3, color=:white, label="", markerstype=:circle)
plt2pdf(plt, "fig10b") #P_s_of_kappa_g_contour_optim

#### Gaussian #######################
plt2 = heatmap(g_s, kappa_s, P_s_values_gaussian', xlabel=L"g_\mathrm{ens}/2\pi\;[\mathrm{MHz}]", ylabel=L"\kappa/2\pi\;[\mathrm{MHz}]", clim=(0, 1), colorbar_title=" \n" * L"P_s", right_margin=5Plots.mm) #title=latexstring(raw"P_s(T="*string(T_s[T_ind])*raw"\mu s)"), 
contour!(plt2, g_s, kappa_s, P_s_values_gaussian', levels=[0.68, 0.87, 0.95, 0.99], linewidth=1, clabels=true, color=:black, label="")
plot!(plt2, xlims=(minimum(g_s), maximum(g_s)), ylims=(minimum(kappa_s), maximum(kappa_s)))
plot!(plt2, [g0], [kappa0], seriestype=:scatter, markersize=3, color=:white, label="", markerstype=:circle)
plt2pdf(plt2, "fig10a") #P_s_of_kappa_g_contour_Zsolt