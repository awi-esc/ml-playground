cd(@__DIR__)
import Pkg
Pkg.activate(".")
Pkg.instantiate()

Pkg.add(["Lux", "Random", "Optimisers", "Zygote"])
Pkg.add(["NCDatasets","CSV", "DataFrames", "Statistics", "CairoMakie","JLD2","Printf"])

using Lux
using Random
using Optimisers
using Zygote

using NCDatasets
using CSV
using DataFrames
using Statistics
using CairoMakie
using JLD2
using Printf

# Set random seed for reproducibility
rng = Random.default_rng()
Random.seed!(rng, 0)

### 1. Load Data ###
if false
    nc = NCDataset(homedir()*"/data/MAR/MARv3.14/Greenland/ERA5-1km-monthly/MARv3.14-monthly-ERA5-2023.nc")

    #SRF3D = repeat(nc["SRF"][:,:,:], 1, 1, 12)
    nx, ny = size(nc["SRF"])
    SRF3D = Array{Float64}(undef, nx, ny, 12);
    for k in 1:12
        SRF3D[:,:,k] .= nc["SRF"][:,:]
    end

    # Find indices of non-missing values
    ii = findall(!ismissing, nc["SMBcorr"][:])

    data = DataFrame(   mdot=nc["SMBcorr"][ii],
                        I=nc["SWD"][ii],
                        z=SRF3D[ii],
                        p=nc["SF"][ii] .+ nc["RF"][ii],
                        T=nc["T2Mcorr"][ii]  )

    # Save the DataFrame to "mydata.jld2"
    @save "mar-smb-2023.jld2" data

else
    @load "mar-smb-2023.jld2" data
end

#data = CSV.read("smb_data.csv", DataFrame)  # Assumes columns: T, p, I, z, smb

data = dropmissing!(data)
data_now = mapcols(x -> Float32.(x), data[1:1000:end,:])

begin
    # Extract inputs and outputs
    X = hcat(data_now.T, data_now.p, data_now.I, data_now.z)'  # Features as a 4×N matrix
    y = data.mdot                              # Target variable

    # Train-Test Split (80-10-10): Training (80%), Validation (10%), Test (10%)
    N = size(X, 2)
    idx = randperm(N)
    train_size = Int(round(0.8 * N))
    val_size   = Int(round(0.1 * N))

    train_idx = idx[1:train_size]
    val_idx   = idx[train_size+1:train_size+val_size]
    test_idx  = idx[train_size+val_size+1:end]

    X_train, X_val, X_test = X[:, train_idx], X[:, val_idx], X[:, test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    ### 2. Normalize Data ###
    μ = mean(X_train, dims=2)
    σ = std(X_train, dims=2) .+ 1e-6  # Avoid division by zero

    X_train_norm = (X_train .- μ) ./ σ
    X_val_norm   = (X_val .- μ) ./ σ
    X_test_norm  = (X_test .- μ) ./ σ  # Use training stats for normalization
end

# Define the model with Dropout
model = Lux.Chain(
    Lux.Dense(4, 32, Lux.relu),  # Input layer
    Lux.Dropout(0.2),             # Dropout: Randomly disables 20% of neurons
    Lux.Dense(32, 32, Lux.relu), # Hidden layer
    Lux.Dropout(0.2),             # Dropout again to prevent overfitting
    Lux.Dense(32, 1)             # Output layer
)

# Get the device determined by Lux
dev = gpu_device()

# Parameter and State Variables
ps, st = Lux.setup(rng, model) |> dev

# Input
x = X_train_norm |> dev

# Run the model
y, st = Lux.apply(model, x, ps, st)

# Gradients
## First construct a TrainState
train_state = Lux.Training.TrainState(model, ps, st, Adam(0.0001f0))

## We can compute the gradients using Training.compute_gradients
gs, loss, stats, train_state = Lux.Training.compute_gradients(
    AutoZygote(), MSELoss(),
    (x, dev(y_train')), train_state
)

lossfn = MSELoss()

function train_model!(model, ps, st, opt, nepochs::Int)
    tstate = Training.TrainState(model, ps, st, opt)
    for i in 1:nepochs
        grads, loss, _, tstate = Training.single_train_step!(
            AutoZygote(), lossfn, (x, dev(y_train')), tstate)
        if i % 100 == 1 || i == nepochs
            @printf "Loss Value after %6d iterations: %.8f\n" i loss
        end
    end
    return tstate.model, tstate.parameters, tstate.states
end

model, ps, st = train_model!(model, ps, st, Descent(0.01f0), 1000)

println("Loss Value after training: ", lossfn(first(model(x, ps, st)), y_train'))

y_pred = first(model(x, ps, st))

# Plot Predictions vs True Values using CairoMakie
begin
    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1], xlabel="True SMB", ylabel="Predicted SMB", title="SMB Predictions")

    scatter!(ax, y_train, Float32.(y_pred[1,:]), color=:blue, markersize=8, label="Predictions")
    lines!(ax, y_train, y_train, color=:red, linestyle=:dash, linewidth=2, label="Perfect Fit")

    axislegend(ax)
    save("smb_predictions.png", fig)
    fig
end