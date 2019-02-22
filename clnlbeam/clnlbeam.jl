using JuMP
using Ipopt

function clnlbeam(N, optimizer, max_iter)
    ni    = N
    alpha = 350
    h     = 1/ni

    m = Model(with_optimizer(optimizer, max_iter = max_iter))

    @variable(m, -1 <= t[i=1:(ni+1)] <= 1, start = 0.05*cos((i-1)*h))
    @variable(m, -0.05 <= x[i=1:(ni+1)] <= 0.05, start = 0.05*cos((i-1)*h))
    @variable(m, u[1:(ni+1)], start = 0.01)

    @NLobjective(m, Min, sum(0.5 * h * (u[i+1]^2 + u[i]^2) + 0.5 * alpha * h * (cos(t[i+1]) + cos(t[i])) for i in 1:ni))

    # boundary conditions
    set_lower_bound(x[1], 0.0)
    set_upper_bound(x[1], 0.0)
    set_lower_bound(x[ni+1], 0.0)
    set_upper_bound(x[ni+1], 0.0)

    set_lower_bound(t[1], 0.0)
    set_upper_bound(t[1], 0.0)
    set_lower_bound(t[ni+1], 0.0)
    set_upper_bound(t[ni+1], 0.0)

    # cons1
    for i in 1:ni
        @NLconstraint(m, x[i+1] - x[i] - 0.5*h*(sin(t[i+1])+sin(t[i])) == 0)
    end
    # cons2
    for i in 1:ni
        @constraint(m, t[i+1] - t[i] - (0.5h)*u[i+1] - (0.5h)*u[i] == 0)
    end

    optimize!(m)

end

clnlbeam(parse(Int,ARGS[1]), Ipopt.Optimizer, 3)
