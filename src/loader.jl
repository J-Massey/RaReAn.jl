using NPZ
using Plots; pythonplot()
using Zygote
using CoupledFields: ndgrid

# create a class to hold of the fluid properties
mutable struct Fluid
    # t::Array{Float32, 4}
    x::Array{Float32, 4}
    y::Array{Float32, 4}
    z::Array{Float32, 4}
    u::Array{Float32, 4}
    v::Array{Float32, 4}
    w::Array{Float32, 4}
    p::Array{Float32, 4}
end

# initialise the fluid properties
function Fluid()
    ts = npzread("./data/fluid.npy")

    x,y,z = collect(ts[:, 1, :, :, :]), collect(ts[:, 2, :, :, :]), collect(ts[:, 3, :, :, :])
    u = vec(collect(ts[:, 4, :, :, :]))
    v = collect(ts[:, 5, :, :, :])
    w = collect(ts[:, 6, :, :, :])
    p = collect(ts[:, 7, :, :, :])

    println(typeof(u))

    return Fluid(x,y,z,u,v,w,p)
end

# function ω_z(x, y, u,v)
#     ∂u_∂y = jacobian(u, y)[2]
#     ∂v_∂x = jacobian(v, x)[1]
#     return ∂v_∂x - ∂u_∂y
# end

# simple plot of the flow
flow = Fluid()

# ω = ω_z(flow.x[1,:,:,1], flow.y[1,:,:,1], flow.u[1,:,:,1], flow.v[1,:,:,1])

plot = contour(flow.x[1,:,:,1], flow.y[1,:,:,1], flow.v[1,:,:,1])
savefig("./flow.png")