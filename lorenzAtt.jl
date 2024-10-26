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

# 4x1 서브플롯 레이아웃 생성
l = @layout [a; b; c; d]

# x(t) 그래프
p1 = plot(t, Y[1, :], label="x(t)", xlabel="Time t", ylabel="x(t)")

# y(t) 그래프
p2 = plot(t, Y[2, :], label="y(t)", xlabel="Time t", ylabel="y(t)")

# z(t) 그래프
p3 = plot(t, Y[3, :], label="z(t)", xlabel="Time t", ylabel="z(t)")

# x-y-z 3D 그래프
p4 = plot(Y[1, :], Y[2, :], Y[3, :], label="Lorenz Attractor", xlabel="x", ylabel="y", zlabel="z", legend=false)

# 서브플롯으로 그래프 표시
plot(p1, p2, p3, p4, layout = l, size=(800, 1200))

display(plot)

@save "lorenz_data.jld2" t Y
