# With parallelisation
## create macro @everywhere that simply returns the function unchanged (to use everywhere without having loaded Distributed.jl)
#macro everywhere(ex)
#    return ex
#end
using LaTeXStrings
@everywhere using JLD
@everywhere using Optimization, Ipopt, OptimizationOptimJL, OptimizationOptimisers
@everywhere using ForwardDiff, Zygote
using BenchmarkTools
@everywhere using Random
using Plots
@everywhere using QuadGK
using SharedArrays
using Roots
rng = MersenneTwister(1234)
# Install packages
# add JLD, Distributed, LaTeXStrings, Optimization, Ipopt, OptimizationOptimJL, OptimizationOptimisers, ForwardDiff, Zygote, BenchmarkTools, QuadGK, Roots, SharedArrays, Plots


@everywhere mutable struct Parameters
    g::Float64
    kappa::Float64
    omega::Float64
    T::Float64
    omega_dash::ComplexF64

    function Parameters(g=1.0, kappa=1.0, omega=10.0, T=10.0) # in MHz (T in microsec)
        g = g * 2 * pi
        kappa = kappa * 2 * pi
        omega = omega * 2 * pi
        omega_dash = sqrt(ComplexF64(-16 * g^2 + (kappa - omega)^2))
        new(g, kappa, omega, T, omega_dash)
    end
end

# pulse generator_real
@everywhere function full_pulse(t::Float64, coeff::Vector{Float64}, T::Float64=10.0; min_ind::Int64=1)::Float64
    i::Int64 = 0
    val::Float64 = 0.0
    #if 0 <= t <= T
    ind = min_ind
    # split coeff into constant term and then sin and cos terms
    for c in coeff
        # ceil(ind/2) is the index of the cosine/sine term
        i = floor(ind / 2)
        if ind % 2 == 0
            val += c * sin(pi / T * i * t)
        else
            val += c * cos(pi / T * i * t)
        end
        ind += 1
    end
    return val
end
@everywhere function full_pulse(t::Vector{Float64}, coeff::Vector{Float64}, T::Float64=10.0; min_ind::Int64=1)::Vector{Float64}
    return [full_pulse(t[i], coeff, T, min_ind=min_ind) for i in 1:length(t)]
end


function plot_full_pulse(coeff, param::Parameters; res::Int64=1000, optim=0.0)
    if typeof(coeff) <: SciMLBase.OptimizationSolution
        objective = coeff.objective
        coeff = coeff.u
        is_optimized = true
    else
        is_optimized = false
    end
    t_arr::Vector{Float64} = range(0, stop=param.T, length=res)
    pulse_vals = full_pulse(t_arr, coeff, param.T)
    plt = plot(t_arr, pulse_vals, label=L"f(t)")
    plot!(xlabel=L"t\;[\mu \mathrm{s}]", ylabel=L"f(t)\;[\mathrm{MHz}]")
    if is_optimized
        title!("Objective: $(-objective)")
    elseif optim != 0.0
        title!("Objective: $(-optim)")
    end
    return plt, pulse_vals, t_arr
end

# Use function handles for more general functions! fun=fun(t)
@everywhere function weighted_pulse_integral_analytical_full(i::Int64, do_sinh::Bool, kappa, omega, omega_dash, T)
    a = -(kappa + omega) / 4
    b = omega_dash / 4
    do_sin::Bool = i % 2 == 0
    j::Int64 = floor(i / 2)
    term = (sign_a, sign_b) -> sign_b^do_sinh * sign_a^do_sin * 1 / 4 * ((-1)^j - exp((a - sign_b * b) * T)) / (-a + sign_b * b + sign_a * im * pi * j / T)
    if do_sin
        return 1 / im * (term(1, 1) + term(1, -1) + term(-1, 1) + term(-1, -1))
    else
        return term(1, 1) + term(1, -1) + term(-1, 1) + term(-1, -1)
    end
end

@everywhere function analytical_P_s_cache_integrals_full(param::Parameters, how_many::Int64, max_Delta::Float64=100.0, reltol::Float64=1e-4)
    if abs(param.omega_dash) < 1e-12
        param.g += 0.000000005
        param.omega_dash = sqrt(ComplexF64(-16 * param.g^2 + (param.kappa - param.omega)^2))
    end
    T = param.T
    g = param.g
    kappa = param.kappa
    omega = param.omega
    omega_dash = param.omega_dash
    p_1 = delta -> omega / (2 * pi) * 1 / (delta^2 + omega^2 / 4)
    A = delta -> -(4 * delta + 2im * omega) / ((2 * delta + im * omega) * (2 * delta + im * kappa) - 4 * g^2)
    B = delta -> 1 / omega_dash * ((-4 * delta - 2im * omega) * (omega - kappa) + 16im * g^2) / ((2 * delta + im * omega) * (2 * delta + im * kappa) - 4 * g^2)

    Nj::Array{ComplexF64} = zeros(how_many)
    Mj::Array{ComplexF64} = zeros(how_many)

    Dj_sin = (j, delta) -> if !(abs(abs(delta) - pi * j / T) < 10^-12)
        pi * T * j * ((-1)^j - exp(-im * T * delta)) / (T^2 * delta^2 - pi^2 * j^2)
    else
        im * pi * j * (-1)^j / (2 * delta)
    end
    Dj_cos = (j, delta) ->
        if j > 0
            if !(abs(abs(delta) - pi * j / T) < 10^-12)
                -im * T^2 * delta * ((-1)^j - exp(-im * T * delta)) / (T^2 * delta^2 - pi^2 * j^2)
            else
                -(-T * (-1)^j) / 2
            end
        else
            # if Delta = 0 ==> T * exp(-im * delta * T), else:
            im / delta * (exp(-im * delta * T) - 1)
        end
    # if j % 2 = 0 -> cos, else sin
    Dj(j, delta) = j % 2 == 1 ? Dj_cos(floor(j / 2), delta) : Dj_sin(floor(j / 2), delta)
    for i in 1:how_many
        # Calculate N_j, M_j
        Nj[i] = weighted_pulse_integral_analytical_full(i, true, kappa, omega, omega_dash, T)
        Mj[i] = weighted_pulse_integral_analytical_full(i, false, kappa, omega, omega_dash, T)
    end
    integral_cache::Array{Float64} = zeros(how_many, how_many)
    Fi(delta, i) = A(delta) * Mj[i] + B(delta) * Nj[i] - A(delta) * Dj(i, delta)
    for i in 1:how_many
        for j in i:how_many
            # abs(x+y)^2 = (x+y)(x+y)' = real(x*x') + real(x*y') + real(y*x') + real(y*y') = abs(x)^2 + abs(y)^2 + 2*real(x*y')
            if i == j
                abs_int_prod_ii(delta) = abs(Fi(delta, i))^2
                integral_cache[i, j] = g^2 * quadgk(delta -> p_1(delta) * abs_int_prod_ii(delta), -max_Delta, -max_Delta / 2, 0, max_Delta / 2, max_Delta, rtol=reltol)[1]
            else
                abs_int_prod_ij(delta) = 2 * real(Fi(delta, i) * conj(Fi(delta, j)))
                integral_cache[i, j] = g^2 * quadgk(delta -> p_1(delta) * abs_int_prod_ij(delta), -max_Delta, -max_Delta / 2, 0, max_Delta / 2, max_Delta, rtol=reltol)[1]
            end
        end
    end
    return integral_cache
end
@everywhere function analytical_P_s_cache_integrals_full_gaussian(param::Parameters, how_many::Int64, max_Delta::Float64=100.0, reltol::Float64=1e-4)
    if abs(param.omega_dash) < 1e-12
        param.g += 0.000000005
        param.omega_dash = sqrt(ComplexF64(-16 * param.g^2 + (param.kappa - param.omega)^2))
    end
    T = param.T
    g = param.g
    kappa = param.kappa
    omega = param.omega
    omega_dash = param.omega_dash
    p_1 = delta -> omega / (2 * pi) * 1 / (delta^2 + omega^2 / 4)
    A = delta -> -(4 * delta + 2im * omega) / ((2 * delta + im * omega) * (2 * delta + im * kappa) - 4 * g^2)
    B = delta -> 1 / omega_dash * ((-4 * delta - 2im * omega) * (omega - kappa) + 16im * g^2) / ((2 * delta + im * omega) * (2 * delta + im * kappa) - 4 * g^2)

    Nj::Array{ComplexF64} = zeros(how_many)
    Mj::Array{ComplexF64} = zeros(how_many)

    Dj_sin = (j, delta) -> if !(abs(abs(delta) - pi * j / T) < 10^-12)
        pi * T * j * ((-1)^j - exp(-im * T * delta)) / (T^2 * delta^2 - pi^2 * j^2)
    else
        im * pi * j * (-1)^j / (2 * delta)
    end
    Dj_cos = (j, delta) ->
        if j > 0
            if !(abs(abs(delta) - pi * j / T) < 10^-12)
                -im * T^2 * delta * ((-1)^j - exp(-im * T * delta)) / (T^2 * delta^2 - pi^2 * j^2)
            else
                -(-T * (-1)^j) / 2
            end
        else
            # if Delta = 0 ==> T * exp(-im * delta * T), else:
            im / delta * (exp(-im * delta * T) - 1)
        end
    # if j % 2 = 0 -> cos, else sin
    Dj(j, delta) = j % 2 == 1 ? Dj_cos(floor(j / 2), delta) : Dj_sin(floor(j / 2), delta)
    for i in 1:how_many
        # Calculate N_j, M_j
        Nj[i] = weighted_pulse_integral_analytical_full(i, true, kappa, omega, omega_dash, T)
        Mj[i] = weighted_pulse_integral_analytical_full(i, false, kappa, omega, omega_dash, T)
    end
    integral_cache::Array{Float64} = zeros(how_many, how_many)
    Fi(delta, i) = A(delta) * Mj[i] + B(delta) * Nj[i] - A(delta) * Dj(i, delta)
    indexes = Zsolt_indexes(how_many)
    n = length(indexes)
    for i0 in 1:n
        i = indexes[i0]
        for j0 in i0:n
            j = indexes[j0]
            # abs(x+y)^2 = (x+y)(x+y)' = real(x*x') + real(x*y') + real(y*x') + real(y*y') = abs(x)^2 + abs(y)^2 + 2*real(x*y')
            if i == j
                abs_int_prod_ii(delta) = abs(Fi(delta, i))^2
                integral_cache[i, j] = g^2 * quadgk(delta -> p_1(delta) * abs_int_prod_ii(delta), -max_Delta, -max_Delta / 2, 0, max_Delta / 2, max_Delta, rtol=reltol)[1]
            else
                abs_int_prod_ij(delta) = 2 * real(Fi(delta, i) * conj(Fi(delta, j)))
                integral_cache[i, j] = g^2 * quadgk(delta -> p_1(delta) * abs_int_prod_ij(delta), -max_Delta, -max_Delta / 2, 0, max_Delta / 2, max_Delta, rtol=reltol)[1]
            end
        end
    end
    return integral_cache
end

@everywhere function create_harmonic_f_constraint_cache_full(n::Int64, param::Parameters, reltol::Float64=1e-4)
    # int_0^T dt sin(w*j*t/T) = T/2
    # diagonal T/2 matrix 
    T = param.T
    T_2 = T / 2
    ind_i::Int64 = 0
    ind_j::Int64 = 0
    integral_matrix::Array{Float64} = zeros(n, n)
    # first the constant terms
    integral_matrix[1, 1] = T # const. * const. 
    for i in 2:n # First the diagonal (T/2) sin_i*sin_i and cos_i*cos_i terms
        integral_matrix[i, i] = T_2  # s_i * s_i or c_i * c_i
        is_sin_i = i % 2 == 0
        ind_i = floor(i / 2)
        if is_sin_i # if sin -> T*(1-(-1)^j)/(pi*j)
            integral_matrix[1, i] = 2 * T * (1 - (-1)^ind_i) / (pi * ind_i)
        end
        for j in i+1:n
            is_sin_j = j % 2 == 0
            ind_j = floor(j / 2)
            if !(ind_i == ind_j)
                if is_sin_i && !is_sin_j   # s_i * c_j
                    integral_matrix[i, j] = 2 * T * ind_i * (1 - (-1)^(ind_i + ind_j)) / (pi * (ind_i^2 - ind_j^2))
                elseif !is_sin_i && is_sin_j  # c_i * s_j
                    integral_matrix[i, j] = 2 * T * ind_j * (1 - (-1)^(ind_i + ind_j)) / (pi * (ind_j^2 - ind_i^2))
                end
            end
        end
    end
    return integral_matrix
end

# check the create_harmonic_f_constraint_cache_full function
@everywhere function create_f_constraint_cache_full(n::Int64, param::Parameters, reltol::Float64=1e-4)
    function_integral_ij(i, j) = 2 * quadgk(t -> real(full_pulse(t, [1.0], param.T; min_ind=i) * conj(full_pulse(t, [1.0], param.T; min_ind=j))), 0, param.T, rtol=reltol)[1]
    function_integral_i(i) = quadgk(t -> abs(full_pulse(t, [1.0], param.T; min_ind=i))^2, 0, param.T, rtol=reltol)[1]
    integral_matrix::Array{Float64} = zeros(n, n)
    for i in 1:n
        for j in i:n
            if i == j
                integral_matrix[i, j] = function_integral_i(i)
            else
                integral_matrix[i, j] = function_integral_ij(i, j)
            end
        end
    end
    return integral_matrix
end
@everywhere function P_s_cached(coeff, integral_cache::Array{Float64})
    # use simd operations, inbounds to quickly calculate the sum( [coeff[i]*coeff[j]*integral_cache[i,j] for i in 1:length(coeff), j in 1:length(coeff)] )
    sum = 0.0
    for i in 1:length(coeff)
        @simd for j in i:length(coeff)
            sum += coeff[i] * coeff[j] * integral_cache[i, j]
        end
    end
    return sum
end

@everywhere function f_constraint_cached(res, coeff, integral_matrix::Array{Float64}, param::Parameters)
    # use simd operations, inbounds to quickly calculate the sum( [coeff[i]*coeff[j]*integral_matrix[i,j] for i in 1:length(coeff), j in 1:length(coeff)] )
    sum = 0.0
    @inbounds for i in 1:length(coeff)
        @simd for j in i:length(coeff)
            sum += coeff[i] * coeff[j] * integral_matrix[i, j]
        end
    end
    res[1] = sum - param.kappa
    return nothing  # optional
end
@everywhere function f_constraint_cached_returns(coeff, integral_matrix::Array{Float64}, kappa::Float64)
    # use simd operations, inbounds to quickly calculate the sum( [coeff[i]*coeff[j]*integral_matrix[i,j] for i in 1:length(coeff), j in 1:length(coeff)] )
    sum = 0.0
    @inbounds for i in 1:length(coeff)
        @simd for j in i:length(coeff)
            sum += coeff[i] * coeff[j] * integral_matrix[i, j]
        end
    end
    return sum - kappa
end

@everywhere function normalize_coeff(coeff, kappa::Float64, constraint_matrix::Array{Float64})
    res = f_constraint_cached_returns(coeff, constraint_matrix, kappa)
    how_big = (res + kappa) / kappa
    coeff2 = coeff / sqrt(how_big)
    return coeff2
end
@everywhere function P_s_cached_normalized(coeff, integral_cache::Array{Float64}, constraint_cache::Array{Float64}, kappa::Float64)
    # first calculate constraint value
    coeff2 = normalize_coeff(coeff, kappa, constraint_cache)
    # use simd operations, inbounds to quickly calculate the sum( [coeff[i]*coeff[j]*integral_cache[i,j] for i in 1:length(coeff), j in 1:length(coeff)] )
    sum = 0.0
    for i in 1:length(coeff2)
        @simd for j in i:length(coeff2)
            sum += coeff2[i] * coeff2[j] * integral_cache[i, j]
        end
    end
    return sum
end

@everywhere function mrange(lengths::Vector{Int64})
    curr_index::Vector{Int64} = ones(Int64, length(lengths))
    max_iterations = prod(lengths)
    counter = 0
    chnl = Channel() do channel
        while counter < max_iterations
            put!(channel, copy(curr_index))
            counter += 1
            for i in length(lengths):-1:1
                curr_index[i] += 1
                if curr_index[i] > lengths[i]
                    curr_index[i] = 1
                    if i == 1
                        return
                    end
                else
                    break
                end
            end
        end
    end
    return chnl
end
@everywhere function vector_iterator(args...)
    n = length(args)
    lengths = [length(arg) for arg in args]
    chnl = Channel() do channel
        for inds in mrange(lengths)
            put!(channel, [args[i][inds[i]] for i in 1:n])
        end
    end
    return chnl
end
@everywhere function optimize_catching_photons(coeff, param::Parameters, reltol::Float64=1e-6, max_delta::Float64=100.0)
    n::Int64 = length(coeff)
    # Fast optimization 
    integral_cache = analytical_P_s_cache_integrals_full(param, n, max_delta, reltol)
    constraint_matrix = create_harmonic_f_constraint_cache_full(n, param, reltol)

    opt_fun = (x, p) -> -P_s_cached_normalized(x, integral_cache, constraint_matrix, param.kappa)
    optimization_problem = OptimizationFunction(opt_fun, Optimization.AutoForwardDiff())
    problem = OptimizationProblem(optimization_problem, coeff, param)
    sol = solve(problem, BFGS(), reltol=1e-10)

    coeff_optim = normalize_coeff(sol.u, param.kappa, constraint_matrix)
    objective = -sol.objective
    return coeff_optim, objective
end
function optimize_catching_photons_T_g_kappa(Ts::Vector, gs::Vector, kappas::Vector, coeff::Vector{Float64}, parameters::Parameters, reltol::Float64=1e-6, max_delta::Float64=100.0, target_step::Int64=100)
    # all combinations of g and kappa
    how_many = length(coeff)
    n_T = length(Ts)
    n_g = length(gs)
    n_k = length(kappas)
    n_total = n_g * n_k * n_T
    # make a vector of all combinations of g and kappa
    function new_param(parameters, T, g, kappa)
        param = deepcopy(parameters)
        param.T = T
        param.kappa = kappa * 2 * pi
        param.g = g * 2 * pi
        param.omega_dash = sqrt(ComplexF64(-16 * param.g^2 + (param.kappa - param.omega)^2))  # important to update this too!
        return param
    end
    combinations = [out for out in vector_iterator(Ts, gs, kappas)]
    index_combinations = [indexes for indexes in mrange([n_T, n_g, n_k])]
    n_batches::Int64 = 20
    n_total_batch::Int64 = Int64(ceil(n_total / n_batches))
    steps::Int64 = min(target_step, Int64(floor(n_total_batch / nprocs())))

    objective_array = SharedArray{Float64}(n_T, n_g, n_k)
    coeff_array = SharedArray{Float64}(how_many, n_T, n_g, n_k)
    println("Calculating in parallel")
    for batch in 0:n_batches-1
        curr_max = min((batch + 1) * n_total_batch, n_total)
        int_min = (1 + batch * n_total_batch)
        @sync @distributed for i in int_min:steps:curr_max
            step_max = min(i + steps - 1, curr_max)
            for j in i:step_max
                ind = index_combinations[j]
                (T, g, k) = combinations[j]
                curr_param = new_param(parameters, T, g, k)
                coeff_array[:, ind...], objective_array[ind...] = optimize_catching_photons(coeff, curr_param, reltol, max_delta)
            end
        end
        println("  Progress: ", Int(ceil((batch + 1) / n_batches * 100)), "%")
    end
    dict = Dict()
    dict["objective"] = Array(objective_array)
    dict["coeff"] = Array(coeff_array)
    dict["param"] = Dict("T" => Ts, "g" => gs, "kappa" => kappas)
    return dict
end
function gaussian_catch_photons_T_g_kappa(Ts::Vector, gs::Vector, kappas::Vector, parameters::Parameters, how_many::Int64=20, reltol::Float64=1e-6, max_delta::Float64=100.0, target_step::Int64=100)
    # all combinations of g and kappa
    n_T = length(Ts)
    n_g = length(gs)
    n_k = length(kappas)
    n_total = n_g * n_k * n_T
    # make a vector of all combinations of g and kappa
    function new_param(parameters, T, g, kappa)
        param = deepcopy(parameters)
        param.T = T
        param.kappa = kappa * 2 * pi
        param.g = g * 2 * pi
        param.omega_dash = sqrt(ComplexF64(-16 * param.g^2 + (param.kappa - param.omega)^2))  # important to update this too!
        return param
    end
    combinations = [out for out in vector_iterator(Ts, gs, kappas)]
    index_combinations = [indexes for indexes in mrange([n_T, n_g, n_k])]
    n_batches::Int64 = 20
    n_total_batch::Int64 = Int64(ceil(n_total / n_batches))
    steps::Int64 = min(target_step, Int64(floor(n_total_batch / nprocs())))

    objective_array = SharedArray{Float64}(n_T, n_g, n_k)
    coeff_array = SharedArray{Float64}(how_many, n_T, n_g, n_k)
    println("Calculating in parallel")
    for batch in 0:n_batches-1
        curr_max = min((batch + 1) * n_total_batch, n_total)
        @sync @distributed for i in (1+batch*n_total_batch):steps:curr_max
            step_max = min(i + steps - 1, curr_max)
            for j in i:step_max
                ind = index_combinations[j]
                (T, g, k) = combinations[j]
                kappa_g = 2 * pi / T
                curr_param = new_param(parameters, T, g, k)
                integral_cache = analytical_P_s_cache_integrals_full(curr_param, how_many, max_delta, reltol)
                constraint_matrix = create_harmonic_f_constraint_cache_full(how_many, curr_param, reltol)
                curr_coeff = expand_Zsolt2coeff_full(how_many, kappa_g, curr_param; reltol=reltol)
                curr_coeff = normalize_coeff(curr_coeff, curr_param.kappa, constraint_matrix)
                curr_P_s_val = P_s_cached_normalized(curr_coeff, integral_cache, constraint_matrix, curr_param.kappa)
                coeff_array[:, ind...] = curr_coeff
                objective_array[ind...] = curr_P_s_val
            end
        end
        println("  Progress: ", Int(ceil((batch + 1) / n_batches * 100)), "%")
    end
    dict = Dict()
    dict["objective"] = Array(objective_array)
    dict["coeff"] = Array(coeff_array)
    dict["param"] = Dict("T" => Ts, "g" => gs, "kappa" => kappas)
    return dict
end

function params_to_str(prefix, param)
    # make a string of the parameters (min, max, length)
    min_param = minimum(param)
    max_param = maximum(param)
    len_param = length(param)
    param_string = prefix * "_" * string(min_param) * "_" * string(max_param) * "_" * string(len_param)
    return param_string
end
function functionname(n, g_s, kappa_s, T_s, omega_s::Vector{Float64}=Float64[]; prefix="optimized_pulses_n_")
    path_str = joinpath("Results", prefix * string(n) * "_" * params_to_str("g", g_s) * "_" * params_to_str("k", kappa_s))
    if length(omega_s) > 0
        path_str *= "_" * params_to_str("o", omega_s)
    end
    path_str *= "_" * params_to_str("T", T_s) * ".jld"
    return path_str
end

function Polynomial_Fit(x_vals, y_vals, degree::Int64=2; first_degree::Int64=0)
    # Finds the best linear regression f(x) = c[1]+ c[2]*x + c[3]*x^2 + ... + c[degree+1]*x^degree
    # returns coefficients c
    if length(x_vals) != length(y_vals)
        error("x_vals and y_vals must have same length")
    end
    how_many_degrees = degree + 1 - first_degree
    if how_many_degrees < 0
        error("degree must be greater than first_degree")
    end
    if length(x_vals) < how_many_degrees
        error("Not enough data points")
    end
    cs::Vector{Float64} = zeros(how_many_degrees)
    # Construct matrix A, where y = Ac
    A = zeros(length(x_vals), how_many_degrees)
    for i in 1:length(x_vals)
        for j in 1:how_many_degrees
            A[i, j] = x_vals[i]^(first_degree + j - 1)
        end
    end
    # best c is found via (A'A)^-1 A'y
    cs = inv(A' * A) * A' * y_vals
    # also construct average error and max error
    approx_y_vals = A * cs
    avg_error = sum(abs.(approx_y_vals - y_vals)) / length(y_vals)
    max_error = maximum(abs.(approx_y_vals - y_vals))
    return cs, avg_error, max_error
end
function find_maximum_in_interval(x_vals, y_vals, degree=-1)
    # Finds the maximum of y_vals in the interval
    # returns (x_max, y_max)
    n = length(x_vals)
    if degree == -1
        degree = n - 2
    end
    if n < degree + 1
        error("Not enough data points")
    end
    cs, evg_error, max_error = Polynomial_Fit(x_vals, y_vals, degree)
    # find maximum of polynomial between x_vals[1] and x_vals[end]
    # use newtons method to find maximum
    poly_fun = x -> sum([cs[i+1] * x^i for i in 0:length(cs)-1])
    result = optimize(x -> -poly_fun(x), x_vals[1], x_vals[end], Brent())
    max_point = Optim.minimizer(result)
    max_value = -Optim.minimum(result)
    return max_point, max_value, degree
end

# find index of maximum for every g
function find_maximum_for_every_g(P_s_values::Array{Float64}, kappa_s::Vector{Float64}; degree::Int64=5, extra_points::Int64=0)
    # Approximates the maximum of P_s_values for every g by finding the maximum in an interval around the maximum 
    #and approximating the intermediate function values via polynomial interpolation
    max_inds = [findmax(P_s_values[i, :])[2] for i in 1:length(g_s)]
    max_i = size(P_s_values)[2]
    # find first max_ind = max_i and cut off the rest 
    for (i, max_ind) in enumerate(max_inds)
        if max_ind == max_i
            max_inds = max_inds[1:i-1]
            break
        end
    end
    max_kappas::Vector{Float64} = zeros(length(max_inds))
    max_P_s_values::Vector{Float64} = zeros(length(max_inds))
    how_many = degree + 1 + extra_points
    for (i, max_ind) in enumerate(max_inds)
        curr_min_i = max(1, max_ind - how_many)
        curr_max_i = min(length(kappa_s), max_ind + how_many)
        # if curr_min_i:curr_max_i is too small, then increase whtever can be increased
        if curr_max_i - curr_min_i < degree + 1
            if curr_min_i == 1
                curr_max_i = minimum(curr_min_i + how_many, length(kappa_s))
            elseif curr_max_i == length(kappa_s)
                curr_min_i = maximum(curr_max_i - how_many, 1)
            else
                # raise exception
                println("Error: Not enough points in interval, need at least " * string(degree + 1) * " points, but have " * string(curr_max_i - curr_min_i) * " points")
            end
        end
        max_kappas[i], max_P_s_values[i] = find_maximum_in_interval(kappa_s[curr_min_i:curr_max_i], P_s_values[i, curr_min_i:curr_max_i], degree)[1:2]
    end
    return max_kappas, max_P_s_values
end

function find_zero_in_interval(x_vals, y_vals, degree=-1)
    # Finds the maximum of y_vals in the interval
    # returns (x_max, y_max)
    n = length(x_vals)
    if degree == -1
        degree = n - 2
    end
    if n < degree + 1
        error("Not enough data points")
    end
    cs, avg_error, max_error = Polynomial_Fit(x_vals, y_vals, degree)
    # find maximum of polynomial between x_vals[1] and x_vals[end]
    # use newtons method to find maximum
    delta = x_vals[2] - x_vals[1]
    poly_fun = x -> sum([cs[i+1] * x^i for i in 0:length(cs)-1])^2
    result = optimize(x -> poly_fun(x), x_vals[1] - delta, x_vals[end], Brent())
    max_point = Optim.minimizer(result)
    return max_point
end


#### Zsolt Comparison
@everywhere function update_T(param, T)
    param.T = T
    return param
end
# make index list of relevant indexes
@everywhere function Zsolt_indexes(n)
    n_arr::Vector{Int64} = []
    for i in 1:n
        ind = Int64(ceil(i / 2))
        if ind % 2 == 1
            push!(n_arr, i)
        end
    end
    return n_arr
end

@everywhere function expand_Zsolt2coeff_full(n, kappa_g, param; reltol=1e-4)
    # the sinuses are sin(pi*n*t/T) for n=1,2,3,...
    # the coefficients are given by the integral of the product of the function and the sinus
    t0 = param.T / 2
    pref = sqrt(param.kappa * kappa_g) / pi^(1 / 4)
    fun = (t) -> exp(-kappa_g^2 * (t - t0)^2 / 2) # defined from 0 to T, with t0=T/2
    coeffs = zeros(n)
    # only sinus terms
    coeffs[1] = pref / 2 / param.T * quadgk(t -> fun(t), 0, param.T; rtol=reltol)[1]
    count = 0.0
    ind = 0
    for i in 2:2:n
        if ind % 2 == 0
            count += pi / param.T
            curr_term = pref / param.T * quadgk(t -> fun(t) * exp(im * count * t), 0, param.T; rtol=reltol)[1]
            coeffs[i] = imag(curr_term)
            if i + 1 <= n
                coeffs[i+1] = real(curr_term)
            end
        end
    end
    return coeffs
end

@everywhere function find_optimal_k(param, n; max_delta=100.0, reltol=1e-6)
    # find optimal kappa
    # guess C = 1.0 -> kappa = 4 g^2 / omega
    init_kappa_g = 4 * param.g^2 / param.omega / (2 * pi)
    # find optimal kappa
    function update_param(param, kappa)
        new_param = deepcopy(param)
        new_param.kappa = kappa * 2 * pi
        new_param.omega_dash = sqrt(complex(-16 * new_param.g^2 + (new_param.kappa - new_param.omega)^2, 0))
        return new_param
    end
    coeff = [1.0 for i in 1:n]
    function opt_fun(kappa)
        new_param = update_param(param, abs(kappa[1]))
        integral_cache = analytical_P_s_cache_integrals_full(new_param, n, max_delta, reltol)
        constraint_matrix = create_harmonic_f_constraint_cache_full(n, new_param, reltol)

        opt_fun = x -> -P_s_cached_normalized(x, integral_cache, constraint_matrix, new_param.kappa)
        results = optimize(opt_fun, coeff, BFGS(), Optim.Options(x_tol=reltol), autodiff=:forward)
        curr_coeff = normalize_coeff(Optim.minimizer(results), new_param.kappa, constraint_matrix)
        curr_P_s_val = P_s_cached_normalized(curr_coeff, integral_cache, constraint_matrix, new_param.kappa)
        return -curr_P_s_val
    end
    # start at init_kappa_g, in interval [kappa_min, kappa_max], with BFGS, autodiff=:forward
    init = [init_kappa_g]
    # use line search in 1D using Golden Section Search
    result = optimize(opt_fun, init, Newton(), Optim.Options(x_tol=reltol))
    kappa_opt = abs(Optim.minimizer(result)[1])
    P_s_val = -Optim.minimum(result)
    #println("Initial kappa: ", init_kappa_g, ", optimal kappa: ", kappa_opt, ", P_s: ", P_s_val)
    return kappa_opt, P_s_val
end

@everywhere function Zsolt_optimize_to_T(target_P_s, n, param, min_T, max_T; reltol=1e-6, max_delta=100.0)
    curr_P_s_val = 0.0
    curr_coeff = ones(n)
    function opt_fun(T)
        kappa_g = 2 * pi / T
        new_param = update_T(deepcopy(param), T)
        integral_cache = analytical_P_s_cache_integrals_full_gaussian(new_param, n, max_delta, reltol)
        constraint_matrix = create_harmonic_f_constraint_cache_full(n, new_param, reltol)
        curr_coeff = expand_Zsolt2coeff_full(n, kappa_g, new_param; reltol=reltol)
        curr_coeff = normalize_coeff(curr_coeff, new_param.kappa, constraint_matrix)
        curr_P_s_val = P_s_cached_normalized(curr_coeff, integral_cache, constraint_matrix, new_param.kappa)
        #println("Zsolt: T=", T[1], ", P_s=", curr_P_s_val)
        return curr_P_s_val
    end
    max_P_s = opt_fun(max_T)
    if max_P_s <= target_P_s
        return max_T, max_P_s, curr_coeff
    end
    min_P_s = opt_fun(min_T)
    #println("Zsolt min_P_s: ", min_P_s)
    if min_P_s >= target_P_s
        return min_T, min_P_s, curr_coeff
    else
        max_point = find_zero(T -> target_P_s - opt_fun(T), (min_T, max_T), rtol=reltol)
        return max_point, curr_P_s_val, curr_coeff
    end
end
@everywhere function optimize_optimization_T(target_P_s, n, coeff, param, min_T, max_T; max_delta=100.0, reltol=1e-6) # at optimal kappa
    curr_P_s_val = 0.0
    kappa_opt = 0.0
    function meta_opt_fun(T)
        new_param = update_T(deepcopy(param), T)
        kappa_opt, curr_P_s_val = find_optimal_k(new_param, n; max_delta=max_delta, reltol=reltol)
        return curr_P_s_val
    end
    max_P_s = meta_opt_fun(max_T)
    if max_P_s < target_P_s
        return max_T, max_P_s, kappa_opt
    end
    min_P_s = meta_opt_fun(min_T)
    #println("min_P_s: ", min_P_s)
    if min_P_s >= target_P_s
        return min_T, min_P_s, kappa_opt
    else
        max_point = find_zero(T -> target_P_s - meta_opt_fun(T), (min_T, max_T), rtol=reltol)
        return max_point, curr_P_s_val, kappa_opt
    end
end

@everywhere function param_2_new_g(param, g)
    kappa = 4 * g^2 / param.omega * 2 * pi
    my_param = deepcopy(param)
    my_param.g = g * 2 * pi
    my_param.kappa = kappa * 2 * pi
    my_param.omega_dash = sqrt(complex(-16 * my_param.g^2 + (my_param.kappa - my_param.omega)^2, 0))  # important to update this too!
    return my_param
end
@everywhere function param_2_new_k(param, k)
    my_param = deepcopy(param)
    my_param.kappa = k * 2 * pi
    my_param.omega_dash = sqrt(complex(-16 * my_param.g^2 + (my_param.kappa - my_param.omega)^2, 0))  # important to update this too!
    return my_param
end


function Both_T_g_to_target(target_P_s, n, n_zsolt, g_s, coeff, param, min_T, max_T; reltol=1e-6, max_delta=100.0)
    T_target = SharedArray{Float64}(length(g_s))
    Zsolt_T_target = SharedArray{Float64}(length(g_s))
    kappa_opt = SharedArray{Float64}(length(g_s))
    g_vals = [g for g in g_s]
    @sync @distributed for i in 1:length(g_vals)
        my_param = param_2_new_g(param, g_vals[i])
        curr_T, curr_P_s_val, kappa = optimize_optimization_T(target_P_s, n, coeff, my_param, min_T, max_T; reltol=reltol, max_delta=max_delta)
        T_target[i] = curr_T
        kappa_opt[i] = kappa
        # decrease search range give error if reach that target
        my_param = param_2_new_k(my_param, kappa_opt[i])
        new_min_T = max(curr_T / 3, min_T)
        Zsolt_T_target[i] = Zsolt_optimize_to_T(target_P_s, n_zsolt, my_param, new_min_T, max_T; reltol=reltol, max_delta=max_delta)[1]
        if Zsolt_T_target[i] == max_T
            Zsolt_T_target[i] = NaN
        end
        if T_target[i] == max_T
            T_target[i] = NaN
        end
    end
    return Array(T_target), Array(Zsolt_T_target), Array(kappa_opt)
end

#### Rewrite these!!!!
function optimize_T_g_to_target(target_P_s, n, g_s, coeff, param, min_T, max_T; reltol=1e-6, max_delta=100.0)
    T_target = SharedArray{Float64}(length(g_s))
    g_vals = [g for g in g_s]
    @sync @distributed for i in 1:length(g_vals)
        my_param = param_2_new_g(param, g_vals[i])
        T_target[i] = optimize_optimization_T(target_P_s, n, coeff, my_param, min_T, max_T; reltol=reltol, max_delta=max_delta)[1]
    end
    return Array(T_target)
end
function Zsolt_T_g_to_target(target_P_s, n, g_s, param, min_T, max_T; reltol=1e-6, max_delta=100.0)
    T_target = SharedArray{Float64}(length(g_s))
    g_vals = [g for g in g_s]
    @sync @distributed for i in 1:length(g_vals)
        my_param = param_2_new_g(param, g_vals[i])
        T_target[i] = Zsolt_optimize_to_T(target_P_s, n, my_param, min_T, max_T; reltol=reltol, max_delta=max_delta)[1]
    end
    return Array(T_target)
end
