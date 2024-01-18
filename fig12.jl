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

# optimize kappa for different g and T values
omega = 10.0
n = 3
redo = false
n_g, n_T = 91, 99
g_s::Vector{Float64} = range(0.1, 1.0, length=n_g)
T_s::Vector{Float64} = range(1.0, 50.0, length=n_T)
kappa_opt_s = SharedArray{Float64}(n_g, n_T)
C = SharedArray{Float64}(n_g, n_T)
filename = joinpath("Results", "optimal_kappa_" * string(n) * "_" * params_to_str("g", g_s) * "_" * params_to_str("T", T_s) * "_w_" * string(omega) * ".jld")
if isfile(filename) && !redo
    dict = load(filename, "dict")
    println("Loaded")
else
    if !isfile(filename)
        println("File does not exist, Optimizing")
    else
        println("Redoing optimization")
    end
    for i in 1:n_g
        g = g_s[i]
        if i % 10 == 0
            println("g = ", g, " (", i, "/", n_g, ")")
        end
        @sync @distributed for j in 1:n_T
            T = T_s[j]
            param = Parameters(g, 1.0, omega, T)
            kappa_opt, P_s_val = find_optimal_k(param, n, max_delta=100.0, reltol=1e-6)
            kappa_opt_s[i, j] = kappa_opt
            # Cooperativity
            C[i, j] = 4.0 * g^2 / (kappa_opt * omega)
        end
    end
    dict = Dict("g_s" => g_s, "T_s" => T_s, "kappa_opt_s" => Array(kappa_opt_s), "C" => Array(C))
    println("Done")
    save(filename, "dict", dict)
    println("Saved")
end

g_s = dict["g_s"]
T_s = dict["T_s"]
C = dict["C"]
plt = plot()
heatmap!(plt, g_s, T_s, C', xlabel=L"g_\mathrm{ens}/2\pi \; [\mathrm{MHz}]", ylabel=L"T \; [\mu s]", colorbar_title=" \n" * L"C", clims=(0, 1.0), right_margin=5Plots.mm)
# add a hightmap at points:
#height_vals = [0.682, 0.954, 0.997]
height_vals = [0.2, 0.6, 0.8, 0.9]#, 0.95]
contour!(plt, g_s, T_s, C', levels=height_vals, color=:black, linewidth=1, label="", right_margin=5Plots.mm)
plt2pdf(plt, "fig12") #C_of_g_T