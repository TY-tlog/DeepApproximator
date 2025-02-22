using Plots, JLD2, SavitzkyGolay, Plots

function runge_kutta4(f, t0, tf, y0, N)
    t = range(t0, stop=tf, length=N+1)
    y = zeros(length(y0), length(t))
    y[:, 1] = y0
    
    h = (tf - t0) / N

    for i = 1:N
        k1 = h * f(t[i], y[:, i])
        k2 = h * f(t[i] + 0.5*h, y[:, i] + 0.5*k1)
        k3 = h * f(t[i] + 0.5*h, y[:, i] + 0.5*k2)
        k4 = h * f(t[i] + h, y[:, i] + k3)

        y[:, i+1] = y[:, i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    end
    return t, y
end

# Lorenz system parameters
sigma = 10
rho = 28
beta = 8 / 3

# Lorenz system equations
lorenz = (t, Y) -> [sigma * (Y[2] - Y[1]); 
                    Y[1] * (rho - Y[3]) - Y[2]; 
                    Y[1] * Y[2] - beta * Y[3]]

# Initial conditions
Y0 = [1.0, 1.0, 1.0]  # Julia uses more general floating-point notation

# Time span and steps
t0 = 0.0
tf = 50.0
N = 10000  # High N for a smoother curve

# Solve the ODEs
t, Y = runge_kutta4(lorenz, t0, tf, Y0, N)

# Apply Savitzky-Golay filter to the data
window_size = 7
order = 4

x_dot = savitzky_golay(Y[1,:], window_size, order, deriv=1, rate = N / (tf - t0))
y_dot = savitzky_golay(Y[2,:], window_size, order, deriv=1, rate = N / (tf - t0))
z_dot = savitzky_golay(Y[3,:], window_size, order, deriv=1, rate = N / (tf - t0))

plot(t, Y[1,:], label="Original x", title="Lorenz System", xlabel="Time", ylabel="x(t)", legend=:outertopright)
plot!(t, Y_filtered.y, label="Filtered x", title="Lorenz System", xlabel="Time", ylabel="x(t)", legend=:outertopright)

Derivative_data = [x_dot.y, y_dot.y, z_dot.y]

Θ = hcat(
    ones(length(t)),  # Constant term
    Y[1,:], Y[2,:], Y[3,:],          # Linear terms
    Y[1,:].^2, Y[1,:].*Y[2,:], Y[1,:].*Y[3,:], Y[2,:].^2, Y[2,:].*Y[3,:], Y[3,:].^2  # Quadratic terms
)

# Sparse regression function for SINDy
function sindy(Θ, d_dt, λ=0.1)
    Ξ = Θ \ d_dt  # Initial least squares solution
    for _ in 1:10  # Iterate to enforce sparsity
        small_inds = abs.(Ξ) .< λ  # Identify small coefficients
        Ξ[small_inds] .= 0         # Set small coefficients to zero
        big_inds = .!small_inds    # Keep larger coefficients
        if sum(big_inds) > 0
            Ξ[big_inds] = Θ[:, big_inds] \ d_dt  # Recompute with active terms
        end
    end
    return Ξ
end

# Apply SINDy to each derivative
λ = 0.05  # Sparsity threshold
Ξx = sindy(Θ, x_dot.y, λ)  # Coefficients for dx/dt
Ξy = sindy(Θ, y_dot.y, λ)  # Coefficients for dy/dt
Ξz = sindy(Θ, z_dot.y, λ)  # Coefficients for dz/dt

# Define the terms for printing
terms = ["1", "x", "y", "z", "x²", "xy", "xz", "y²", "yz", "z²"]

# Print the identified equations
println("Identified equations:")
println("dx/dt = ", join([string(round(c, digits=3)) * "*" * t for (c, t) in zip(Ξx, terms) if abs(c) > 1e-3], " + "))
println("dy/dt = ", join([string(round(c, digits=3)) * "*" * t for (c, t) in zip(Ξy, terms) if abs(c) > 1e-3], " + "))
println("dz/dt = ", join([string(round(c, digits=3)) * "*" * t for (c, t) in zip(Ξz, terms) if abs(c) > 1e-3], " + "))