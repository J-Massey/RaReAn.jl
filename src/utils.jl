using LinearAlgebra
using Statistics
using Random, SparseArrays


@inline CI(a...) = CartesianIndex(a...)
δ(i,::Val{N}) where N = CI(ntuple(j -> j==i ? 1 : 0, N))
δ(i,I::CartesianIndex{N}) where N = δ(i, Val{N}())
@inline ∂(a,I::CartesianIndex{d},f::AbstractArray{T,d}) where {T,d} = @inbounds f[I]-f[I-δ(a,I)]
@inline ∂(a,I::CartesianIndex{m},u::AbstractArray{T,n}) where {T,n,m} = @inbounds u[I+δ(a,I),a]-u[I,a]

function derivative_wrapper(field, dim, stretch_factor=1)

    # Define the size of the field
    size_x, size_y = size(field)

    # Create an array to store the derivatives
    derivatives = similar(field)
    # Compute the derivative for each point in the field
    for i in 1:size_x
        for j in 1:size_y
            # Create the Cartesian index object I for each point
            I = CartesianIndex(i, j)
            # Compute the derivative using the ∂ function
            derivatives[I] = ∂(dim, I, field)
        end
    end
    # fill edges with zeros
    derivatives[1:2,:] .= 0
    derivatives[end-1:end,:] .= 0
    derivatives[:,1:2] .= 0
    derivatives[:,end-1:end] .= 0
    return derivatives/stretch_factor
end


function fbDMD(X, r)
    X1 = X
    X2 = circshift(X, 1)
    # Perform SVD
    U2, Σ2, V2 = svd(X1)
    U = U2[:, 1:r]
    Σ = Diagonal(Σ2)[1:r, 1:r]
    V = V2[:, 1:r]

    Ãf = U' * X2 * V / Σ

    # Backwards step
    U2, Σ2, V2 = svd(X2)
    U = U2[:, 1:r]
    Σ = Diagonal(Σ2)[1:r, 1:r]
    V = V2[:, 1:r]

    Ãb = U' * X1 * V / Σ

    Ã  = sqrt(Ãf / Ãb)

    D, W = eigen(Ã)
    Phi = X2 * V / Σ * W
    return Phi, Ã
end


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
        U[:, i] = U[:, i] / Σ[i] # Algorithm 1, line 9
    end

    ## OPTIONAL : Recovor U, Σ

    U, Σ, Ṽ = svd!(U_to_Σ, full=false)

    V = V * Ṽ

    return U, Σ, V
end
