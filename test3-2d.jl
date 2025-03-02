# Testing this method
# https://www.youtube.com/watch?v=juTzmSDZm8Q

cd(@__DIR__)
import Pkg
Pkg.activate(".")
Pkg.instantiate()

using Lux
using Random
using Zygote
using Optimisers
using CairoMakie
using Distributions
using Statistics

# Initialize psuedo-random number generator
rng = Xoshiro(42)

function mynorm(x)
    # Compute mean and standard deviation
    x_mean = mean(x)
    x_std  = std(x)

    # Standardize x values
    x_norm = (x .- x_mean) ./ x_std
    return x_norm
end

begin
    # Draw a toy dataset with x1 and x2
    N_SAMPLES = 200
    X = Float32.(rand(rng, Uniform(0.0, 2 * π), (2, N_SAMPLES)))
    y_noise   = Float32.(rand(rng, Normal(0.0, 0.3), (1, N_SAMPLES)))
    Y = Float32.(y_noise .* 0.0)
    Y[1,:] = sin.(X[1, :]) .+ cos.(X[2, :]) .+ y_noise[1,:]

    X_norm = mapslices(mynorm,X,dims=2)
end

# Define the model architecture
model = Chain(
    Dense(2, 10, Lux.relu),
    Dense(10, 10, Lux.relu),
    Dense(10, 10, Lux.relu),
    Dense(10, 1, identity)
)

# Initialize the parameters 
ps, ls = Lux.setup(rng, model)

Y_pred_initial, ls = model(X, ps, ls)

# The forward function
function loss_fn_mse(p, ls)
    y_prediction, new_ls = model(X,p,ls)
    loss = 0.5 * mean( (y_prediction .- Y).^2)
    return loss, new_ls
end

function loss_fn_huber(p, ls; delta=1.0)
    y_prediction, new_ls = model(X,p,ls)
    residual = abs.(y_prediction .- Y)
    mask = residual .< delta
    loss = mean(mask .* (0.5 * residual .^ 2) .+ .!mask .* (delta .* (residual .- 0.5 * delta)))
    return loss, new_ls
end

#loss_fn = loss_fn_mse
loss_fn = loss_fn_huber

# Use plain gradient Descent, or use Adam
opt = Adam(0.01)
opt_state = Optimisers.setup(opt, ps)

function plot_it(x,y,y_pred)
    fig = Figure(size=(800,800))
    
    ax1 = Axis(fig[1,1:2])
    scatter!(ax1,x[1,:],y[:], color=:grey60, markersize=8, label="data")
    scatter!(ax1,x[1,:],y_pred[:], color=:green, label="prediction")
    
    ax2 = Axis(fig[1,3:4])
    scatter!(ax2,x[2,:],y[:], color=:grey60, markersize=8, label="data")
    scatter!(ax2,x[2,:],y_pred[:], color=:green, label="prediction")
    
    ax3 = Axis(fig[2,2:3])
    scatter!(ax3,y[:], y_pred[:], label="Predicted vs True")
    ablines!(ax3,0,1)
    ax3.xlabel = "True values"
    ax3.ylabel = "Predicted values"

    return fig, ax1, ax2, ax3
end

fig, ax1, ax2 = plot_it(X,Y,Y_pred_initial)
fig

# Train loop
N_EPOCHS = 30_000

loss_history = []
for epoch = 1:N_EPOCHS
    (loss, ls,), back = pullback(loss_fn, ps, ls)
    grad, _ = back((1.0,nothing))

    opt_state, ps = Optimisers.update(opt_state, ps, grad)

    push!(loss_history, loss)

    if epoch % 100 == 0
        println("Epoch: $epoch, Loss: $loss")
    end

end

begin
    fig, ax = lines(loss_history)
    #ax.yscale = log10
    fig
end

Y_pred_final, ls = model(X, ps, ls)

fig, ax1, ax2 = plot_it(X,Y,Y_pred_final)
fig
save("test.png",fig)