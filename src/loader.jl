using NPZ
# using Plots; pythonplot()

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
    ts = npzread(raw"../data/fluid.npy")

    x,y,z = collect(ts[:, 1, :, :, :]), collect(ts[:, 2, :, :, :]), collect(ts[:, 3, :, :, :])
    u = (collect(ts[:, 4, :, :, :]))
    v = collect(ts[:, 5, :, :, :])
    w = collect(ts[:, 6, :, :, :])
    p = collect(ts[:, 7, :, :, :])

    return Fluid(x,y,z,u,v,w,p)
end

