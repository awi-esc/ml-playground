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

begin
    N_SAMPLES = 200
    LAYERS    = [2, 10, 10 ,10, 1]
    LEARNING_RATE = 0.1
    N_EPOCHS = 30_000

    # Our psuedo-random number generator
    rng = Xoshiro(42)
end

begin
    # Draw a toy dataset with x1 and x2
    x_samples = rand(rng, Uniform(0.0, 2 * π), (2, N_SAMPLES))  # Now (2, N_SAMPLES)
    y_noise = rand(rng, Normal(0.0, 0.3), (1, N_SAMPLES))
    y_samples = y_noise .* 0.0
    y_samples[1,:] = sin.(x_samples[1, :]) .+ cos.(x_samples[2, :]) .+ y_noise[1,:]
    #y_samples[1,:] = sin.(x_samples[1, :]) .+ y_noise[1,:]
end

# Define the model architecture
model = Chain(
    [Dense(fan_in => fan_out, Lux.sigmoid) for (fan_in,fan_out) in zip(LAYERS[1:end-2], LAYERS[2:end-1])]...,
    Dense(LAYERS[end-1] => LAYERS[end], identity)
)

# Initialize the parameters 
parameters, layer_states = Lux.setup(rng, model)

y_initial_prediction, layer_states = model(x_samples, parameters, layer_states)

# The forward function
function loss_fn(p, ls)
    y_prediction, new_ls = model(x_samples,p,ls)
    loss = 0.5 * mean( (y_prediction .- y_samples).^2)
    return loss, new_ls
end

# Use plain gradient Descent, or use Adam
opt = Descent(LEARNING_RATE)
opt_state = Optimisers.setup(opt, parameters)

function plot_it(y_pred,y_samples)
    fig = Figure(size=(800,800))
    
    ax1 = Axis(fig[1,1:2])
    scatter!(ax1,x_samples[1,:],y_samples[:], color=:grey60, markersize=8, label="data")
    scatter!(ax1,x_samples[1,:],y_pred[:], color=:green, label="prediction")
    
    ax2 = Axis(fig[1,3:4])
    scatter!(ax2,x_samples[2,:],y_samples[:], color=:grey60, markersize=8, label="data")
    scatter!(ax2,x_samples[2,:],y_pred[:], color=:green, label="prediction")
    
    ax3 = Axis(fig[2,2:3])
    scatter!(ax3,y_samples[:], y_final_prediction[:], label="Predicted vs True")
    ablines!(ax3,0,1)
    ax3.xlabel = "True values"
    ax3.ylabel = "Predicted values"

    return fig, ax1, ax2, ax3
end

fig, ax1, ax2 = plot_it(y_initial_prediction,y_samples)
fig

# Train loop
loss_history = []
for epoch = 1:N_EPOCHS
    (loss, layer_states,), back = pullback(loss_fn, parameters, layer_states)
    grad, _ = back((1.0,nothing))

    opt_state, parameters = Optimisers.update(opt_state, parameters, grad)

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

y_final_prediction, layer_states = model(x_samples, parameters, layer_states)

fig, ax1, ax2 = plot_it(y_final_prediction,y_samples)
fig