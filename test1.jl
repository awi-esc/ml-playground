cd(@__DIR__)
import Pkg
Pkg.activate(".")
Pkg.instantiate()

#Pkg.add(["Revise","CairoMakie", "Statistics"])
#Pkg.add(["Flux", "NeuralOperators", "DifferentialEquations"])
#Pkg.add("FFTW")
Pkg.add(["Zygote","Optim"])

using Revise
using CairoMakie
using Flux, NeuralOperators
using FFTW

using Flux, DifferentialEquations, Optim, Zygote

# Define the neural network (a simple feedforward NN)
nn = Chain(Dense(1, 10, relu), Dense(10, 10, relu), Dense(10, 1))

# Parameters of the network
ps = Flux.params(nn)

# Function for computing the second derivative using automatic differentiation
function laplacian(x, nn, ps)
    u = x -> nn([x])[1]  # Neural network output
    du = x -> Zygote.gradient(u, x)[1]  # First derivative
    d2u = x -> Zygote.gradient(du, x)[1]  # Second derivative
    return d2u(x)
end

# Physics-informed loss function
function loss(nn, ps)
    # Sample points in the domain
    x = range(0, 1, length=20)  # 20 collocation points
    f(x) = sin(pi * x)  # Right-hand side forcing function

    # PDE loss (enforcing the governing equation)
    pde_loss = sum(abs2.(laplacian(xi, nn, ps) - f(xi)) for xi in x)

    # Boundary conditions loss
    bc_loss = abs2(nn([0])[1] - 0) + abs2(nn([1])[1] - 0)

    return pde_loss + bc_loss
end

# Define the training process
opt = Flux.ADAM(0.01)  # Optimizer

# Training loop
for epoch in 1:1000
    l = loss(nn, ps)  # Compute loss
    Flux.Optimise.update!(opt, ps, Flux.gradient(() -> loss(nn, ps), ps))  # Update parameters
    if epoch % 100 == 0
        println("Epoch $epoch, Loss: $l")
    end
end

# Evaluate the trained network on a fine grid
x_test = range(0, 1, length=100)
u_pred = [nn([xi])[1] for xi in x_test]

# Plot the solution
plot(x_test, u_pred, label="PINN Approximation", lw=2)
