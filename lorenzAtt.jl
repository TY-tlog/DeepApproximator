#####################################################################
# Lorenz Attractor
#####################################################################
using Plots, JLD2

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

# Create 4x1 subplot layout
# l = @layout [a; b; c; d]

# # x(t) plot
# p1 = plot(t, Y[1, :], label="x(t)", xlabel="Time t", ylabel="x(t)")

# # y(t) plot
# p2 = plot(t, Y[2, :], label="y(t)", xlabel="Time t", ylabel="y(t)")

# # z(t) plot
# p3 = plot(t, Y[3, :], label="z(t)", xlabel="Time t", ylabel="z(t)")

# # x-y-z 3D plot
# p4 = plot(Y[1, :], Y[2, :], Y[3, :], label="Lorenz Attractor", xlabel="x", ylabel="y", zlabel="z", legend=false)

# # Display subplots
# plot(p1, p2, p3, p4, layout = l, size=(800, 1200))

file_dir = "/Volumes/YD/DeepApproximator"
file_name = "lorenz_data.jld2"
myfile = joinpath(file_dir, file_name)

@save myfile t Y

#####################################################################
# Lyapunov Exponent ##Example Code(x + r - x^2)
#####################################################################

using Plots

# Functions
function f(x, r)
    # return cos(r*x)^2
    return x + r - x^2
end

function df(x, r)
    # return -2*r*cos(r*x)*sin(r*x)
    return 1 - 2*x
end

# Main parameters
T = 100
dt = 0.01
N = Int(T/dt)
x1 = 0.1

# Create plot
plot()

# Loop over r values
for r in 0:0.01:2
    # Initialize arrays
    dy_dt = zeros(N)
    df_dt = zeros(N)
    
    # Main iteration loop
    for i in 1:N
        dy_dt[i] = x1
        df_dt[i] = df(x1, r)
        global x1 = f(x1, r)  # Need global because x1 is modified in loop
    end
    
    # Calculate Lyapunov exponent
    LyaE = (1/N) * sum(log.(abs.(df_dt)))
    
    # Plot point
    scatter!([r], [LyaE], color=:blue, markersize=1, label="")
end

# Add labels and show plot
xlabel!("r")
ylabel!("Lyapunov exponent")
display(plot!)