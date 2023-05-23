using LinearAlgebra, Random, SparseArrays

function randomized_resolvent(Lq, ω, k; Φ=nothing)
    # RANDOMIZED RESOLVENT ANALYSIS
    #
    # Ref: J.H.M. Ribeiro, C.-A. Yeh and K. Taira, 
    #      "Randomized Resolvent Analysis," 
    #       Physical Review Fluids, 5, 033902, 2020.
    #       (doi:10.1103/PhysRevFluids.5.033902)
    #
    # Inputs:
    # -- L   : Linearized Navier-Stokes operator (n x n)
    # -- w   : Radial frequency (real value)
    # -- k   : Width of test (random) matrix (integer); optional but users 
    #          should check for convergence
    # -- Φ : Physics-based scaling matrix (n x 1); optional
    #   NOTE : the resolvent operator is defined as inv( -iw eye(n) - L )
    #
    # Outputs:
    # -- U    : Response modes  (n x k)
    # -- Sigma: Singular values (k x 1)
    # -- V    : Forcing modes   (n x k)

    n = size(Lq, 2)

    ## STEP 1 -- BUILD RANDOM MATRIX

    # Generate test matrix with random normal Gaussian distribution (default)
    Ω = randn(n, k) # Algorithm 1, line 1

    # # Optional: scale test vectors with physics-based weighting.
    if isdefined(Main, :Φ)
        if size(Φ, 1) != n
            error("number of rows in Φ and cols in L must be the same")
        end

        Ωp = zeros(n, k)
        for i in 1:k
            Ωp[:, i] = Ω[:, i] .* Φ # physics-based random matrix
        end
    else
        Ωp = Ω # use randn (default)
    end

    ## STEP 2 -- Solve linear system for Y such that Y ← [−iωI - Lq]\Ωp
    dL = (-1im * ω * Matrix{ComplexF64}(I, n, n) - Lq)
    Y = dL\Ωp

    ## STEP 3 -- Economical QR decomposition of Y
    Q, R = qr(Y)

    ## STEP 4 -- Solve linear system for B, B ← Q'/[−iωI − Lq]
    B = Q' / dL

    ## STEP 5 -- SVD of B
    U, Σ, V = svd!(B)

    ## STEP 6 -- Solve linear system to recover UΣ
    U_to_Σ = dL\V

    ## STEP 7 -- Compute U and Σ
    for i in 1:k
        Σ[i] = norm(U_to_Σ[:, i], 2) # Algorithm 1, line 8
        U_to_Σ[:, i] = U_to_Σ[:, i] / Σ[i] # Algorithm 1, line 9
    end

    ## OPTIONAL : Recovor U, Σ
    U, Σ, Ṽ = svd!(U_to_Σ, full=false)
    V = V * Ṽ

    return U, Σ, V
end

# using NPZ
# using LaTeXStrings
# using Plots; pythonplot()
# using LinearAlgebra
# using Statistics
# include("utils.jl")
# include("DMD.jl")

# ts = npzread(raw"data/fluid.npy")

# x,y,z = collect(ts[:, 1, :, :, :]), collect(ts[:, 2, :, :, :]), collect(ts[:, 3, :, :, :])
# u = (collect(ts[:, 4, :, :, :]))
# v = collect(ts[:, 5, :, :, :])
# w = collect(ts[:, 6, :, :, :])
# p = collect(ts[:, 7, :, :, :])

# # ∂u_∂x(t::Int64) = derivative_wrapper(collect(u[t, :, :, 1]), 2, 4)
# ∂u_∂y(t::Int64) = derivative_wrapper(collect(u[t, :, :, 1]), 1, 2)
# ∂v_∂x(t::Int64) = derivative_wrapper(collect(v[t, :, :, 1]), 2, 4)
# # ∂v_∂y(t::Int64) = derivative_wrapper(collect(v[t, :, :, 1]), 1, 2)
# @inline ω_z(t::Int64) = ∂v_∂x(t) - ∂u_∂y(t)

# # Create the concatanated snapshots
# t = collect(1:size(x, 1))
# size_ω = size(vec(ω_z(1)), 1)

# tgrid = repeat(t, 1, size_ω)'
# ωgrid = similar(tgrid, Float64)
# xgrid = similar(ωgrid)

# for i in t
#     ωgrid[:, i] = vec(ω_z(i))
#     xgrid[:, i] = vec(x[i,:,:,1])
# end

# Φ, A = fbDMD(ωgrid, 5)
# println(size(Φ), size(A))

# # # resolvant operator
# # n = size(Φ, 2); k = 6; ω = 0.5;
# # k = minimum((k, n))
# # Ω = rand(n, k)
# # dL = (-1im * ω * Matrix{ComplexF64}(I, n, n) - A)
# # Y = dL\Ω
# # Q, R = qr(Y)
# # B = Q' / dL
# # U, Σ, V = svd!(B, full=false)
# # # println(size(U), size(Σ), size(V))
# # U_to_Σ = dL\V

# # Σ = zeros(k)
# # for i in 1:k
# #     Σ[i] = norm(U_to_Σ[:, i], 2) # Algorithm 1, line 8
# #     U_to_Σ[:, i] = U_to_Σ[:, i] / Σ[i]     # Algorithm 1, line 9
# # end

# # U, Σ, Ṽ = svd!(U_to_Σ, full=false)
# # V = V * Ṽ

# # U, Σ, V
# # println(size(U), size(Σ), size(V))

# # full resolvant
# n = size(Φ, 2); k = 6; ω = 0.5;
# k = minimum((k, n))
# dL = (-1im * ω * Matrix{ComplexF64}(I, n, n) - A)
# U, Σ, V = svd!(dL)

# # resolvent_modes = V[:, 1:5]
# # diagm(Σ[1:5])

# forcing_modes = Φ * V
# response_modes = Φ * U

# color_levels = range(-0.02, stop=0.02, length=44)
# for i in 1:5
#     snap = reshape(real(forcing_modes[:, i]), length(y[1,:,1,1]), length(x[1,1,:,1]))

#     # freq = round(omega[i], digits=4)

#     contourf(x[1,:, :, 1],
#             y[1,:, :, 1],
#             snap,
#             xlabel="x", ylabel="y", title="mode=$i ",
#             clims=(-0.02, 0.02),
#             levels=color_levels,
#             cmap=:seismic,
#             aspect_ratio=:equal,
#     )

#     savefig("../figures/forcing_mode$i.png")
# end

# color_levels = range(-0.02, stop=0.02, length=44)
# for i in 1:5
#     snap = reshape(real(response_modes[:, i]), length(y[1,:,1,1]), length(x[1,1,:,1]))

#     # freq = round(omega[i], digits=4)

#     contourf(x[1,:, :, 1],
#             y[1,:, :, 1],
#             snap,
#             xlabel="x", ylabel="y", title="mode=$i ",
#             clims=(-0.02, 0.02),
#             levels=color_levels,
#             cmap=:seismic,
#             aspect_ratio=:equal,
#     )

#     savefig("../figures/response_mode$i.png")
# end