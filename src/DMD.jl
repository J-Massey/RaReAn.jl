using LinearAlgebra

function DMD(X, r)
    X1 = X
    X2 = circshift(X, 1)
    # Perform SVD
    U2, Σ2, V2 = svd(X1)
    U = U2[:, 1:r]
    Σ = Diagonal(Σ2)[1:r, 1:r]
    V = V2[:, 1:r]

    # DMD J-Tu decomposition
    Ã = U' * X2 * V / Σ

    D, W = eigen(Ã)
    Phi = X2 * V / Σ * W
    return Phi
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

function fbDMD_time(X, r)
    # fbDMD with time dynamics
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
    
    # Extract time dynamics using a fourier decomposition to get a phase space with dominant fequencies
    mu = D
    omega = log.(mu)
    println(maximum(real(omega)))
    
    # Generate time vector, this needs to be adjusted for the time step
    t = range(1, size(X, 2))
    
    # Calculate time dynamics
    u0 = X[:, 1]
    y0 = Phi \ u0
    u_modes = zeros(ComplexF64, r, length(t))
    
    for iter in 1:length(t)
        u_modes[:, iter] = y0 .* exp.(omega .* t[iter]) # This is the fourier transform
    end
    u_dmd = Phi * u_modes
    return Phi, Ã, u_dmd
end


################################################
#            --Testing Functions--             #
################################################

####   Plot the first 5 DMD modes  ####
# r=5
# Phi = real(DMD(ωgrid, r))

# for i in 1:r
#     snap = reshape(Phi[:, i], length(y[1,:,1,1]), length(x[1,1,:,1]))

#     contourf(x[1,:, :, 1],
#             y[1,:, :, 1],
#             snap,
#             xlabel="x", ylabel="y", title="ω_z contour",
#             #  clims=(-0.1, 0.1),
#             #  levels=color_levels,
#             cmap=:seismic,
#             aspect_ratio=:equal,
#     )

#     savefig("../figures/DMD$i.png")
# end

####   Plot the first 5 fbDMD modes ####
# r=5
# Phi, A = fbDMD(ωgrid, r)

# D, W = eigen(A)
# omega = angle.(D) ./ (2π*0.1/3)

# color_levels = range(-0.02, stop=0.02, length=51)

# for i in 1:r
#     snap = reshape(real(Phi[:, i]), length(y[1,:,1,1]), length(x[1,1,:,1]))

#     freq = round(omega[i], digits=4)

#     contourf(x[1,:, :, 1],
#             y[1,:, :, 1],
#             snap,
#             xlabel="x", ylabel="y", title="ω=$freq ",
#             clims=(-0.02, 0.02),
#             levels=color_levels,
#             cmap=:seismic,
#             aspect_ratio=:equal,
#     )

#     savefig("../figures/fbDMD$i.png")
# end

####   Plot the spectrum ####

# theta = range(0, stop=2π, length=100)
# Λ, W = eigen(A)
# plot!(cos.(theta), sin.(theta), color=:black, label="", aspect_ratio=:equal)
# scatter!(real(Λ), imag(Λ), color=:red, label="")
# savefig("../figures/spectrum.png")

####   Plot the time dynamics ####
# r=7
# Phi, A, u_dmd = fbDMD(ωgrid, r)

# contourf(xgrid, tgrid, real(u_dmd), )
# # Save the plot to a file
# savefig("../figures/dynamics_plot.png")

