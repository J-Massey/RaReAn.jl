using LinearAlgebra
using Random
using Test
# using BenchmarkTools

# """
# rSVD(X,k)

# Randomised SVD using sketching.

# # Arguments
# - `A`: Data matrix of shape (m, n) representing the data.
# - `k`: Number of desired singular values/vectors.

# # Returns
# - `M`: Matrix of shape (m, k) containing the left singular vectors or modes.
# - `Σ`: Array of length k containing the initial singular values.
# """

# function rSVD(A,k)
#     (m,n)=size(A)
#     Phi=rand(n,k)
#     Y=A*Phi
#     Q,R = qr(Y)
#     B=transpose(Matrix(Q))*A
#     Û,Σ,Vt=svd(B)
#     Uk=Û[:, 1:k]
#     Vk=Vt'[1:k, :]
#     U=Matrix(Q)*Uk
#     M = U * diagm(Σ[1:k]) * Vk
#     return M, Σ
# end

"""
    initialize(A, k)

Initializes the data matrix and computes the initial left singular vectors and singular values.

# Arguments
- `A`: Data matrix of shape (m, n) representing the first batch of data.
- `k`: Number of desired singular values/vectors.

# Returns
- `M`: Initial matrix of shape (m, k) containing the left singular vectors.
- `Σ`: Array of length k containing the initial singular values.
"""
function initialize(A, k)
    q, r = qr(A)
    ui, Σ, vit = svd(r)
    M = q * ui[:, 1:k]
    Σ = Σ[1:k]
    return M, Σ
end

"""
    incorporate_data(A, M, Σ, k)

Incorporates new data into the existing data matrix using singular value decomposition (SVD).

# Arguments
- `A`: Existing data matrix of shape (m, n).
- `M`: Matrix of shape (m, k) containing the left singular vectors.
- `Σ`: Array of length k containing the singular values.
- `k`: Number of singular values to incorporate.

# Returns
- `M`: Updated matrix of shape (m, k) containing the left singular vectors.
- `Σ`: Updated array of length k containing the singular values.
"""

function incorporate_data(A, M, Σ, k)
    m_ap = M * Diagonal(Σ)
    m_ap = hcat(m_ap, A)
    ui, di = qr(m_ap)
    ũi, d̃i, ṽti = svd(di)
    max_idx = sortperm(d̃i, rev=true)[1:k]
    Σ = d̃i[max_idx]
    ũi = ũi[:, max_idx]
    M = ui * ũi
    return M, Σ
end

@testset "Test the SVDs" begin
    m=1000; n=50; k=10
    A=rand(m,n)
    # @testset "rSVD" begin
    #     M, Σ = rSVD(A,k)
        
    #     println(size(M))
    #     # @test length(Σ) == k
    # end
    @testset "Streaming parts" begin
        Mi, Σi = initialize(A, k)
        @test size(Mi) == (size(A, 1), k)
        @test length(Σi) == k
        A2 = rand(m, n+10)
        M, Σ = incorporate_data(A2, Mi, Σi, k)
        @test size(M) == (size(A, 1), k)
        @test length(Σ) == k
    end
    @testset "Streaming" begin
        M, Σ = initialize(A, k)
        @test size(M) == (size(A, 1), k)
        @test length(Σ) == k
        A2 = rand(m, n+10)
        M, Σ = incorporate_data(A2, M, Σ, k)
        @test size(M) == (size(A, 1), k)
        @test length(Σ) == k
    end
end

function BenchmarkStreaming(m,n,k)
    A=rand(m,n)
    M1, Σ1 = initialize(A, k)
    A2 = rand(m, n+10)
    M, Σ = incorporate_data(A2, M1, Σ1, k)
    return M, Σ
end

# bmark = @benchmarkable BenchmarkStreaming(1000, 50, 10)
# tune!(bmark)
# run(bmark)
# dump(bmark)