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

using ProperOrthogonalDecomposition

# res, singularvals  = POD(Y)
# reconstructFirstMode = res.modes[:,1:1]*res.coefficients[1:1,:]
