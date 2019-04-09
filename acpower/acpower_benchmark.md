# Benchmarking JuMP Automatic Differentiation with Zygote
 
```julia
using JuMP
using Ipopt
using Compat
using DelimitedFiles
using BenchmarkTools
using Zygote
```

## Create Problem
### Initialize Problem Parameters

```julia
numbuses = 662;

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

Gself(k) =
    bus[k].g_shunt +
    if !isempty(out_lines[k])
        sum(branch[i].g*branch_tap[i]^2 for i in out_lines[k])
    else 0 end  +
    if !isempty(in_lines[k]) sum(branch[i].g for i in in_lines[k])
    else 0 end

Gout(i) = (-branch[i].g*cos(branch_def[i])+branch[i].b*sin(branch_def[i]))*branch_tap[i]
Gin(i)  = (-branch[i].g*cos(branch_def[i])-branch[i].b*sin(branch_def[i]))*branch_tap[i]

Bself(k) =
    bus_b_shunt[k] +
    if !isempty(out_lines[k])
        sum(branch[i].b*branch_tap[i]^2 + branch[i].c/2 for i in out_lines[k])
    else 0 end +
    if !isempty(in_lines[k])
        sum(branch[i].b + branch[i].c/2 for i in in_lines[k])
    else 0 end 

Bout(i) = (-branch[i].g*sin(branch_def[i])-branch[i].b*cos(branch_def[i]))*branch_tap[i]
Bin(i)  = (branch[i].g*sin(branch_def[i])-branch[i].b*cos(branch_def[i]))*branch_tap[i]
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
  memory estimate:  16 bytes
  allocs estimate:  1
  --------------
  minimum time:     34.951 μs (0.00% GC)
  median time:      35.847 μs (0.00% GC)
  mean time:        36.166 μs (0.00% GC)
  maximum time:     138.493 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     1
```
#### Zygote

```julia
function obj_fun(bus_voltage, bus_angle, branch_tap, branch_def)
    sum( (bus[k].p_load +
          if !isempty(in_lines[k])
          sum( bus_voltage[k] * bus_voltage[branch[i].from] *
               (Gin(i) * cos(bus_angle[k] - bus_angle[branch[i].from]) +
                Bin(i) * sin(bus_angle[k] - bus_angle[branch[i].from])) for i in in_lines[k] ) else 0 end +
          if !isempty(out_lines[k])
          sum( bus_voltage[k] * bus_voltage[branch[i].to] *
               (Gout(i) * cos(bus_angle[k] - bus_angle[branch[i].to]) +
                Bout(i) * sin(bus_angle[k] - bus_angle[branch[i].to])) for i in out_lines[k] ) else 0 end
          + bus_voltage[k]^2*Gself(k) )^2
        for k in 1:nbus if bus[k].bustype == 2 || bus[k].bustype == 3)
end

@benchmark Zygote.gradient((bus_voltage, bus_angle, branch_tap, branch_def) -> obj_fun(bus_voltage, bus_angle, branch_tap, branch_def), bus_voltage, bus_angle, branch_tap, branch_def)
```

```
julia> @benchmark Zygote.gradient((bus_voltage, bus_angle, branch_tap, branch_def) -> obj_fun(bus_voltage, bus_angle, branch_tap, branch_def), bus_voltage, bus_angle, branch_tap, branch_def)
BenchmarkTools.Trial:
  memory estimate:  60.43 MiB
  allocs estimate:  245594
  --------------
  minimum time:     58.019 ms (25.22% GC)
  median time:      64.722 ms (32.33% GC)
  mean time:        70.026 ms (37.46% GC)
  maximum time:     147.928 ms (69.02% GC)
  --------------
  samples:          72
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
  memory estimate:  323.22 KiB
  allocs estimate:  20686
  --------------
  minimum time:     6.414 ms (0.00% GC)
  median time:      6.705 ms (0.00% GC)
  mean time:        6.844 ms (0.41% GC)
  maximum time:     10.161 ms (21.77% GC)
  --------------
  samples:          724
  evals/sample:     1
```

#### Zygote
```julia
function eval_constraint_jacobian_zygote(bus_voltage, bus_angle, bus_b_shunt, branch_tap, branch_def)
    for k in 1:nbus
        Zygote.gradient((bus_voltage, bus_angle, bus_b_shunt, branch_tap, branch_def) -> bus[k].p_gen - bus[k].p_load -
          if !isempty(in_lines[k])
             sum( bus_voltage[k] * bus_voltage[branch[i].from] *
                 (Gin(i) * cos(bus_angle[k] - bus_angle[branch[i].from]) +
                  Bin(i) * sin(bus_angle[k] - bus_angle[branch[i].from])) for i in in_lines[k] ) else 0 end -
          if !isempty(out_lines[k])
             sum( bus_voltage[k] * bus_voltage[branch[i].to] *
                 (Gout(i) * cos(bus_angle[k] - bus_angle[branch[i].to]) +
                  Bout(i) * sin(bus_angle[k] - bus_angle[branch[i].to])) for i in out_lines[k] ) else 0 end -
          bus_voltage[k]^2*Gself(k),
           bus_voltage, bus_angle, bus_b_shunt, branch_tap, branch_def)

        Zygote.gradient((bus_voltage, bus_angle, bus_b_shunt, branch_tap, branch_def) -> bus[k].q_gen - bus[k].q_load -
          if !isempty(in_lines[k])
            sum( bus_voltage[k] * bus_voltage[branch[i].from] *
                (Gin(i) * sin(bus_angle[k] - bus_angle[branch[i].from]) -
                 Bin(i) * cos(bus_angle[k] - bus_angle[branch[i].from])) for i in in_lines[k] ) else 0 end -
          if !isempty(out_lines[k])
            sum( bus_voltage[k] * bus_voltage[branch[i].to] *
                (Gout(i) * sin(bus_angle[k] - bus_angle[branch[i].to]) -
                 Bout(i) * cos(bus_angle[k] - bus_angle[branch[i].to])) for i in out_lines[k] ) else 0 end
             + bus_voltage[k]^2*Bself(k), bus_voltage, bus_angle, bus_b_shunt, branch_tap, branch_def)
    end
    
    for k in 1:nbus
        Zygote.gradient((bus_voltage, bus_angle, bus_b_shunt, branch_tap, branch_def) -> bus[k].q_load +
          if !isempty(in_lines[k])
            sum( bus_voltage[k] * bus_voltage[branch[i].from] *
                (Gin(i) * sin(bus_angle[k] - bus_angle[branch[i].from]) -
                 Bin(i) * cos(bus_angle[k] - bus_angle[branch[i].from])) for i in in_lines[k] ) else 0 end +
          if !isempty(out_lines[k])
            sum( bus_voltage[k] * bus_voltage[branch[i].to] *
                (Gout(i) * sin(bus_angle[k] - bus_angle[branch[i].to]) -
                 Bout(i) * cos(bus_angle[k] - bus_angle[branch[i].to])) for i in out_lines[k] ) else 0 end +
            - bus_voltage[k]^2*Bself(k) - bus[k].q_max, bus_voltage, bus_angle, bus_b_shunt, branch_tap, branch_def)

        Zygote.gradient((bus_voltage, bus_angle, bus_b_shunt, branch_tap, branch_def) -> bus[k].p_load +
          if !isempty(in_lines[k])
            sum( bus_voltage[k] * bus_voltage[branch[i].from] *
                (Gin(i) * cos(bus_angle[k] - bus_angle[branch[i].from]) +
                 Bin(i) * sin(bus_angle[k] - bus_angle[branch[i].from])) for i in in_lines[k] ) else 0 end +
          if !isempty(out_lines[k])
            sum( bus_voltage[k] * bus_voltage[branch[i].to] *
                (Gout(i) * cos(bus_angle[k] - bus_angle[branch[i].to]) +
                 Bout(i) * sin(bus_angle[k] - bus_angle[branch[i].to])) for i in out_lines[k] ) else 0 end  +
            + bus_voltage[k]^2*Gself(k) - p_gen_upper*bus[k].p_gen, bus_voltage, bus_angle, bus_b_shunt, branch_tap, branch_def)
    end
end

@benchmark eval_constraint_jacobian_zygote(bus_voltage, bus_angle, bus_b_shunt, branch_tap, branch_def)
```

```
julia> @benchmark eval_constraint_jacobian_zygote(bus_voltage, bus_angle, bus_b_shunt, branch_tap, branch_def)
BenchmarkTools.Trial:
  memory estimate:  1.46 GiB
  allocs estimate:  6210520
  --------------
  minimum time:     1.565 s (15.55% GC)
  median time:      1.568 s (15.65% GC)
  mean time:        1.569 s (15.65% GC)
  maximum time:     1.574 s (15.62% GC)
  --------------
  samples:          4
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
  memory estimate:  41.41 KiB
  allocs estimate:  2650
  --------------
  minimum time:     15.076 ms (0.00% GC)
  median time:      15.363 ms (0.00% GC)
  mean time:        15.655 ms (0.00% GC)
  maximum time:     18.839 ms (0.00% GC)
  --------------
  samples:          320
  evals/sample:     1
```
