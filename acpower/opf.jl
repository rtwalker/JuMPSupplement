using JuMP
using Ipopt
using Compat
using DelimitedFiles

mutable struct Bus
    bustype::Int
    name::String
    voltage0::Float64
    angle0::Float64
    p_gen::Float64
    q_gen::Float64
    q_min::Float64
    q_max::Float64
    p_load::Float64
    q_load::Float64
    g_shunt::Float64
    b_shunt0::Float64
    b_shunt_min::Float64
    b_shunt_max::Float64
    b_dispatch::Float64
    area::Float64
end

mutable struct Branch
    from::Int
    to::Int
    branchtype::Int
    r::Float64
    x::Float64
    c::Float64
    tap0::Float64
    tap_min0::Float64
    tap_max0::Float64
    def0::Float64
    def_min::Float64
    def_max::Float64
    g::Float64
    b::Float64
end

function opf(bus, branch,
    bus_voltage_min, bus_voltage_max,
    branch_tap_min, branch_tap_max,
    p_gen_upper, p_gen_lower)

    model = Model(with_optimizer(Ipopt.Optimizer, max_iter=3))

    nbus = length(bus)
    nbranch = length(branch)
    @variable(model, bus_voltage_min[bus[i].bustype] <= bus_voltage[i=1:nbus] <= bus_voltage_max[bus[i].bustype], start=1)
    @variable(model, bus[i].b_shunt_min <= bus_b_shunt[i=1:nbus] <= bus[i].b_shunt_max, start=bus[i].b_shunt0)
    @variable(model, bus_angle[1:nbus], start=0)

    @variable(model, branch_tap_min <= branch_tap[1:nbranch] <= branch_tap_max, start=1)
    @variable(model, branch[i].def_min <= branch_def[i=1:nbranch] <= branch[i].def_max, start = branch[i].def0)

    in_lines = [Int[] for i in 1:nbus] # indices of incoming branches
    out_lines = [Int[] for i in 1:nbus] # indices of outgoing branches
    @show nbranch
    @show nbus
    for i in 1:nbranch
        b = branch[i]
        push!(out_lines[b.from],i)
        push!(in_lines[b.to],i)
        @assert 1 <= b.to <= nbus
    end

    @NLexpression(model, Gself[k=1:nbus],
        bus[k].g_shunt +
        sum(branch[i].g*branch_tap[i]^2 for i in out_lines[k]) +
        sum( branch[i].g for i in in_lines[k]))

    @NLexpression(model, Gout[i=1:nbranch],
        (-branch[i].g*cos(branch_def[i])+branch[i].b*sin(branch_def[i]))*branch_tap[i])
    @NLexpression(model, Gin[i=1:nbranch],
        (-branch[i].g*cos(branch_def[i])-branch[i].b*sin(branch_def[i]))*branch_tap[i])

    @NLexpression(model, Bself[k=1:nbus],
        bus_b_shunt[k] +
        sum(branch[i].b*branch_tap[i]^2 + branch[i].c/2 for i in out_lines[k]) +
        sum( branch[i].b + branch[i].c/2 for i in in_lines[k]))

    @NLexpression(model, Bin[i=1:nbranch],
        (branch[i].g*sin(branch_def[i])-branch[i].b*cos(branch_def[i]))*branch_tap[i])
    @NLexpression(model, Bout[i=1:nbranch],
        (-branch[i].g*sin(branch_def[i])-branch[i].b*cos(branch_def[i]))*branch_tap[i])

    # Minimize active power

    @NLobjective(model, Min, sum( (bus[k].p_load +
        sum( bus_voltage[k] * bus_voltage[branch[i].from] *
            (Gin[i] * cos(bus_angle[k] - bus_angle[branch[i].from]) +
             Bin[i] * sin(bus_angle[k] - bus_angle[branch[i].from])) for i in in_lines[k] ) +
        sum( bus_voltage[k] * bus_voltage[branch[i].to] *
            (Gout[i] * cos(bus_angle[k] - bus_angle[branch[i].to]) +
             Bout[i] * sin(bus_angle[k] - bus_angle[branch[i].to])) for i in out_lines[k] )
        + bus_voltage[k]^2*Gself[k] )^2
        for k in 1:nbus if bus[k].bustype == 2 || bus[k].bustype == 3))

    for k in 1:nbus
        bus[k].bustype == 0 || continue
        # p_load
        @NLconstraint(model, bus[k].p_gen - bus[k].p_load -
            sum( bus_voltage[k] * bus_voltage[branch[i].from] *
                (Gin[i] * cos(bus_angle[k] - bus_angle[branch[i].from]) +
                 Bin[i] * sin(bus_angle[k] - bus_angle[branch[i].from])) for i in in_lines[k] ) -
            sum( bus_voltage[k] * bus_voltage[branch[i].to] *
                (Gout[i] * cos(bus_angle[k] - bus_angle[branch[i].to]) +
                 Bout[i] * sin(bus_angle[k] - bus_angle[branch[i].to])) for i in out_lines[k] )
            - bus_voltage[k]^2*Gself[k] == 0)
        # q_load
        @NLconstraint(model, bus[k].q_gen - bus[k].q_load -
            sum( bus_voltage[k] * bus_voltage[branch[i].from] *
                (Gin[i] * sin(bus_angle[k] - bus_angle[branch[i].from]) -
                 Bin[i] * cos(bus_angle[k] - bus_angle[branch[i].from])) for i in in_lines[k] ) -
            sum( bus_voltage[k] * bus_voltage[branch[i].to] *
                (Gout[i] * sin(bus_angle[k] - bus_angle[branch[i].to]) -
                 Bout[i] * cos(bus_angle[k] - bus_angle[branch[i].to])) for i in out_lines[k] )
             + bus_voltage[k]^2*Bself[k] == 0)
    end

    for k in 1:nbus
        (bus[k].bustype == 2 || bus[k].bustype == 3) || continue
        # q_inj
        @NLconstraint(model, bus[k].q_min <= bus[k].q_load +
            sum( bus_voltage[k] * bus_voltage[branch[i].from] *
                (Gin[i] * sin(bus_angle[k] - bus_angle[branch[i].from]) -
                 Bin[i] * cos(bus_angle[k] - bus_angle[branch[i].from])) for i in in_lines[k] ) +
            sum( bus_voltage[k] * bus_voltage[branch[i].to] *
                (Gout[i] * sin(bus_angle[k] - bus_angle[branch[i].to]) -
                 Bout[i] * cos(bus_angle[k] - bus_angle[branch[i].to])) for i in out_lines[k] ) +
            - bus_voltage[k]^2*Bself[k] <= bus[k].q_max)
        # p_inj
        @NLconstraint(model, 0 <= bus[k].p_load +
            sum( bus_voltage[k] * bus_voltage[branch[i].from] *
                (Gin[i] * cos(bus_angle[k] - bus_angle[branch[i].from]) +
                 Bin[i] * sin(bus_angle[k] - bus_angle[branch[i].from])) for i in in_lines[k] ) +
            sum( bus_voltage[k] * bus_voltage[branch[i].to] *
                (Gout[i] * cos(bus_angle[k] - bus_angle[branch[i].to]) +
                 Bout[i] * sin(bus_angle[k] - bus_angle[branch[i].to])) for i in out_lines[k] ) +
            + bus_voltage[k]^2*Gself[k] <= p_gen_upper*bus[k].p_gen)
    end

    for i in 1:nbus
        #---- FREEZE THE REFERENCE BUS ANGLE TO ZERO.
        if bus[i].bustype == 3
            set_lower_bound(bus_angle[i], 0.0)
            set_upper_bound(bus_angle[i], 0.0)
        end
        #---- FREEZE ANY DISPATCHABLE SHUNTS.
        if bus[i].b_dispatch == 0
            set_lower_bound(bus_b_shunt[i],bus[i].b_shunt0)
            set_upper_bound(bus_b_shunt[i],bus[i].b_shunt0)
        end
    end

    for i in 1:nbranch
        #---- FREEZE ANY BRANCH TAPS.
        if branch[i].branchtype == 0 || branch[i].branchtype == 3
            set_lower_bound(branch_tap[i],1)
            set_upper_bound(branch_tap[i],1)
        end
        #---- FREEZE CERTAIN PHASE SHIFTERS.
        if branch[i].branchtype != 4
            set_lower_bound(branch_def[i], start_value(branch_def[i]))
            set_upper_bound(branch_def[i], start_value(branch_def[i]))
        end
    end


    println("solving")
    optimize!(model)

end

function prepdata(bus, branch)

    buses = Array{Bus}(undef, size(bus, 1))
    refbus = 0
    shunt = 0
    busmap = Dict()
    for i in 1:size(bus,1)
        @assert !haskey(busmap, bus[i,1])
        busmap[bus[i,1]] = i
        buses[i] = Bus(bus[i,2:end]...)
        # rescale
        buses[i].p_gen /= 100
        buses[i].q_gen /= 100
        buses[i].q_min /= 100
        buses[i].q_max /= 100
        buses[i].p_load /= 100
        buses[i].q_load /= 100
        if buses[i].bustype == 3
            refbus += 1
        end
        if buses[i].b_dispatch == 0
            shunt += 1
        end
    end
    @show refbus
    @show shunt


    branches = Array{Branch}(undef, size(branch, 1))
    branchtap = 0
    phaseshift = 0
    for i in 1:size(branch,1)
        branches[i] = Branch(branch[i,2:end]...,0,0)
        @assert branch[i,1] == i
        branches[i].to = busmap[branches[i].to]
        branches[i].from = busmap[branches[i].from]
        branches[i].g = branches[i].r /(branches[i].r^2 + branches[i].x^2)
        branches[i].b = -branches[i].x /(branches[i].r^2 + branches[i].x^2)
        # rescale
        branches[i].def_min *= 3.14159/180
        branches[i].def_max *= 3.14159/180
        branches[i].def0 *= -3.14159/180
        if branches[i].branchtype == 0 || branches[i].branchtype == 3
            branchtap += 1
        end
        if branches[i].branchtype != 4
            phaseshift += 1
        end
    end
    @show branchtap
    @show phaseshift
    return branches, buses
end

numbuses = ARGS[1]
branches, buses = prepdata(readdlm("IEEE$numbuses.bus"),readdlm("IEEE$numbuses.branch"))


bus_voltage_min = @compat Dict(0 => 0.85, 1 => 0.85, 2 => 0.92, 3 => 0.99)
bus_voltage_max = @compat Dict(0 => 1.15, 1 => 1.15, 2 => 1.08, 3 => 1.01)
branch_tap_min = 0.85
branch_tap_max = 1.15

p_gen_upper = 1.10
p_gen_lower = 0.90
opf(buses, branches,
    bus_voltage_min, bus_voltage_max,
    branch_tap_min, branch_tap_max,
    p_gen_upper, p_gen_lower)
