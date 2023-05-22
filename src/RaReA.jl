using LinearAlgebra, Random, SparseArrays
function randomized_resolvent(L, w, k, Phi)
    # RANDOMIZED RESOLVENT ANALYSIS
    #
    # Authors: Jean Helder Marques Ribeiro, Chi-An Yeh, & Kunihiko Taira
    # Email: jeanmarques@ucla.edu
    # University of California, Los Angeles
    # Department of Mechanical and Aerospace Engineering 
    #
    # This code is provided as-is and intended for academic research use only. 
    # No commercial use is allowed without permission.  
    # For details, please see:
    #
    # Ref: J.H.M. Ribeiro, C.-A. Yeh and K. Taira, 
    #      "Randomized Resolvent Analysis," 
    #       Physical Review Fluids, 5, 033902, 2020.
    #       (doi:10.1103/PhysRevFluids.5.033902)
    #
    # The code is written for educational clarity and not for speed.  
    #
    # -- Started: Dec 12, 2018 
    # -- Revised: Aug 26, 2020
    #
    # Inputs:
    # -- L   : Linearized Navier-Stokes operator (n x n)
    # -- w   : Radial frequency (real value)
    # -- k   : Width of test (random) matrix (integer); optional but users 
    #          should check for convergence
    # -- Phi : Physics-based scaling matrix (n x 1); optional
    #   NOTE : the resolvent operator is defined as inv( -iw eye(n) - L )
    #
    # Outputs:
    # -- U    : Response modes  (n x k)
    # -- Sigma: Singular values (k x 1)
    # -- V    : Forcing modes   (n x k)
    
    if size(L, 1) != size(L, 2)
        error("L must be square")
    end
    n = size(L, 2)
    
    if !isdefined(Main, :k)
        k = min(20, n) # Default if not provided
    end
    
    ## STEP 1 -- BUILD RANDOM MATRIX
    
    # Generate test matrix with random normal Gaussian distribution (default)
    Omega = randn(n, k) # Algorithm 1, line 1
    
    # Optional: scale test vectors with physics-based weighting.
    if isdefined(Main, :Phi)
        if size(Phi, 1) != n
            error("number of rows in Phi and cols in L must be the same")
        end
        
        Omegap = zeros(n, k)
        for i in 1:k
            Omegap[:, i] = Omega[:, i] .* Phi # physics-based random matrix
        end
    else
        Omegap = Omega # use randn (default)
    end
    
    ## STEP 2 -- QB DECOMPOSITION
    
    # Decompose matrix for linear system solvers.
    dL = lufact!(-1im * w * Matrix{ComplexF64}(I, n, n) - L)
    
    # Sketch resolvent operator: Eq. (6) in PRF paper.
    Y = dL \ Omegap # Algorithm 1, line 2
    
    # Orthogonalize sketch using QR decomposition.
    Q = qr(Y).Q # Algorithm 1, line 3
    
    # Project Resolvent Opt. into reduced-basis. Eq (7) in PRF paper.
    B = Q' / dL # Algorithm 1, line 4
    
    ## STEP 3 -- COMPUTATION OF SINGULAR VALUES AND VECTORS
    
    # SVD of Resolvent Opt. Projection:
    U, Sigma, V = svd(B, thin=false) # Algorithm 1, line 5
    
    # Solve linear system to compute U vectors. Eq. (9) in PRF paper.
    U = dL \ V # Algorithm 1, line 6
    
    # Option 1: obtain left singular vectors and singular values.
    # Sigma = zeros(k, 1)
    # for i in 1:k
    #     Sigma[i] = norm(U[:, i], 2) # Algorithm 1, line 8
    #     U[:, i] = U[:, i] / Sigma[i] # Algorithm 1, line 9
    # end
    
    # Option 2: obtain singular vectors and singular values
    # -- Slightly more accurate.  Enforces orthogonality of singular 
    # vectors, however, requires one additional SVD.
    U, Sigma, Vt = svd(U, thin=false) # Algorithm 1, line 11
    Sigma = diagm(Sigma)
    V = V * Vt # Algorithm 1, line 12
    
    return U, Sigma, V
end
