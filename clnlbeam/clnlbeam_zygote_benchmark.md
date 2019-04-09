# Benchmarking JuMP Automatic Differentiation with Zygote
 
```julia
using JuMP
using Ipopt
using Zygote
using BenchmarkTools
```

## Create Problem
### Initialize Problem Parameters

```julia
ni = 500; 

function create(ni)
    alpha = 350; 
    h     = 1/ni; 

    u = fill(0.01, ni+1);
    x = map(i -> 0.5*cos((i-1)*h), 1:(ni+1));
    t = map(i -> 0.5*cos((i-1)*h), 1:(ni+1));
	
	values = vcat(u, x, t);

    return alpha, h, u, x, t, values
end

alpha, h, u, x, t, values = create(ni);

```

### Create JuMP Model

```julia
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

    return(m)

end
```

```julia
m = clnlbeam(ni, Ipopt.Optimizer, 3); 

d = NLPEvaluator(m);
MOI.initialize(d, [:Grad, :Hess])
```

## Evaluate Gradients
### Evalute objective gradient

#### JuMP

```julia
grad_f = zeros(length(all_variables(m)));

@benchmark MOI.eval_objective_gradient(d, grad_f, values)
```

```
julia> @benchmark MOI.eval_objective_gradient(d, grad_f, values)
BenchmarkTools.Trial:
  memory estimate:  16 bytes
  allocs estimate:  1
  --------------
  minimum time:     7.586 μs (0.00% GC)
  median time:      7.656 μs (0.00% GC)
  mean time:        7.761 μs (0.00% GC)
  maximum time:     22.813 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     4
```

#### Zygote
```julia
@benchmark Zygote.gradient((u, t) -> sum(0.5 * h * (u[i+1]^2 + u[i]^2) + 0.5 * alpha * h * (cos(t[i+1]) + cos(t[i])) for i in 1:ni), u, t)
```

```
julia> @benchmark Zygote.gradient((u, t) -> sum(0.5 * h * (u[i+1]^2 + u[i]^2) + 0.5 * alpha * h * (cos(t[i+1]) + cos(t[i])) for i in 1:ni), u, t)
BenchmarkTools.Trial:
memory estimate:  17.58 MiB
  allocs estimate:  61774
  --------------
  minimum time:     7.243 ms (0.00% GC)
  median time:      10.542 ms (27.09% GC)
  mean time:        10.194 ms (23.04% GC)
  maximum time:     15.622 ms (24.40% GC)
  --------------
  samples:          491
  evals/sample:     1 
```

### Evalute the Jacobian of the constraint matrix

#### JuMP

```julia
J = zeros(length(MOI.jacobian_structure(d)));

@benchmark MOI.eval_constraint_jacobian(d, J, values)
```

```
julia> @benchmark MOI.eval_constraint_jacobian(d, J, values)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     21.638 μs (0.00% GC)
  median time:      22.675 μs (0.00% GC)
  mean time:        22.845 μs (0.00% GC)
  maximum time:     65.368 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     1
```

#### Zygote
```julia
function eval_constraint_jacobian_zygote(u, x, t)
    for i in 1:ni
        Zygote.gradient((x, t) -> x[i+1] - x[i] - 0.5*h*(sin(t[i+1]) + sin(t[i])), x, t)
        Zygote.gradient((t, u) -> t[i+1] - t[i] - 0.5*h*(u[i+1] - u[i]), t, u)
    end
end


@benchmark eval_constraint_jacobian_zygote(u, x, t)
```

```
julia> @benchmark eval_constraint_jacobian_zygote(u, x, t)
BenchmarkTools.Trial:
  memory estimate:  25.68 MiB
  allocs estimate:  64001
  --------------
  minimum time:     9.671 ms (17.76% GC)
  median time:      10.500 ms (19.17% GC)
  mean time:        10.785 ms (21.82% GC)
  maximum time:     13.568 ms (32.51% GC)
  --------------
  samples:          464
  evals/sample:     1
```


### Evaluate the Hessian of the Lagrangian 
#### Compose Lagrangian

```julia
lambda = ones(500);
rho = ones(500);
uxt = vcat(u, x, t);

L(uxt) = (sum(0.5 * h * (uxt[i+1]^2 + uxt[i]^2) + 0.5 * alpha * h * (cos(uxt[i+1003]) + cos(uxt[i+1002])) for i in 1:ni)
           + sum(lambda[i]*(uxt[i+502] - uxt[i+501] - 0.5*h*(sin(uxt[i+1003])+sin(uxt[i+1002]))) for i in 1:ni)
           + sum(rho[i]*(uxt[i+1003] - uxt[i+1002] - (0.5h)*uxt[i+1] - (0.5h)*uxt[i]) for i in 1:ni))
```

#### JuMP
```julia
H = zeros(size(MOI.hessian_lagrangian_structure(d), 1));
mu = vcat(lambda, rho);

@benchmark MOI.eval_hessian_lagrangian(d, H, uxt, 1.0, mu)
```
```
julia> @benchmark MOI.eval_hessian_lagrangian(d, H, uxt, 1.0, mu)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     214.040 μs (0.00% GC)
  median time:      220.200 μs (0.00% GC)
  mean time:        223.998 μs (0.00% GC)
  maximum time:     444.838 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     1
```

#### ReverseDiff
I'm having some trouble getting Zygote to cooperate with getting second order derivatives 
of `sum(...)` generators, so I'm using ReverseDiff in the meantime. 

```julia
@benchmark ReverseDiff.hessian(L, uxt)
```

```
julia> @benchmark ReverseDiff.hessian(L, uxt)
BenchmarkTools.Trial:
  memory estimate:  43.06 MiB
  allocs estimate:  758497
  --------------
  minimum time:     3.748 s (0.32% GC)
  median time:      3.776 s (0.40% GC)
  mean time:        3.776 s (0.40% GC)
  maximum time:     3.804 s (0.48% GC)
  --------------
  samples:          2
  evals/sample:     1
```
