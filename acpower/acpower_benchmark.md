# Benchmarking JuMP Automatic Differentiation with Zygote
 
```julia
using JuMP
using Ipopt
using Compat
using DelimitedFiles
using BenchmarkTools
```

## Create Problem
### Initialize Problem Parameters

```julia
numbuses = 66200;

branch, bus = prepdata(readdlm("IEEE$numbuses.bus"),readdlm("IEEE$numbuses.branch"));

bus_voltage_min = @compat Dict(0 => 0.85, 1 => 0.85, 2 => 0.92, 3 => 0.99)
bus_voltage_max = @compat Dict(0 => 1.15, 1 => 1.15, 2 => 1.08, 3 => 1.01)
branch_tap_min = 0.85
branch_tap_max = 1.15

p_gen_upper = 1.10
p_gen_lower = 0.90

nbus = length(bus)
nbranch = length(branch)

in_lines = [Int[] for i in 1:nbus];
out_lines = [Int[] for i in 1:nbus];

bus_voltage = ones(nbus);
bus_b_shunt = map(i -> bus[i].b_shunt0, 1:nbus);
bus_angle = zeros(nbus);
branch_tap = ones(nbranch);
branch_def = map(i -> branch[i].def0, 1:nbranch);

values = vcat(bus_voltage, bus_angle, bus_b_shunt, branch_tap, branch_def);
```

### Create JuMP Model

```julia
model = opf(bus, branch, bus_voltage_min, bus_voltage_max, branch_tap_min, branch_tap_max, p_gen_upper, p_gen_lower)

d = NLPEvaluator(model)
MOI.initialize(d, [:Grad, :Hess])
```

## Evaluate Derivatives
### Evalute objective gradient

#### JuMP
```julia
grad_f = zeros(length(all_variables(model)));

@benchmark MOI.eval_objective_gradient(d, grad_f, values)
```

```
julia> @benchmark MOI.eval_objective_gradient(d, grad_f, values)
BenchmarkTools.Trial:
  memory estimate:  31.65 MiB
  allocs estimate:  2074046
  --------------
  minimum time:     577.060 ms (2.16% GC)
  median time:      587.063 ms (2.12% GC)
  mean time:        590.536 ms (2.74% GC)
  maximum time:     614.835 ms (3.60% GC)
  --------------
  samples:          9
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
  memory estimate:  31.65 MiB
  allocs estimate:  2074045
  --------------
  minimum time:     751.162 ms (1.45% GC)
  median time:      774.065 ms (1.52% GC)
  mean time:        774.848 ms (2.11% GC)
  maximum time:     796.790 ms (1.40% GC)
  --------------
  samples:          7
  evals/sample:     1
```

### Evaluate the Hessian of the Lagrangian 

#### JuMP
```julia
H = zeros(length(MOI.hessian_lagrangian_structure(d)));
mu = ones(length(d.constraints));
values = vcat(bus_voltage, bus_angle, bus_b_shunt, branch_tap, branch_def);

@benchmark MOI.eval_hessian_lagrangian(d, H, values, 1.0, mu)
```

```
julia> @benchmark MOI.eval_hessian_lagrangian(d, H, values, 1.0, mu)
BenchmarkTools.Trial:
  memory estimate:  4.04 MiB
  allocs estimate:  264802
  --------------
  minimum time:     1.938 s (0.00% GC)
  median time:      1.953 s (0.00% GC)
  mean time:        1.951 s (0.00% GC)
  maximum time:     1.963 s (0.00% GC)
  --------------
  samples:          3
  evals/sample:     1
```
