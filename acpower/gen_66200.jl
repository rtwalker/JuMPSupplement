using DelimitedFiles
#bus = readdlm("IEEE014.bus")
bus = readdlm("IEEE662.bus", String)
busmap = Dict()
for i in 1:size(bus,1)
    @assert !haskey(busmap, bus[i,1])
    busmap[bus[i,1]] = string(i)
    bus[i,1] = string(i)
    bus[i,3] = string("\"", bus[i,3], "\"") # make sure names are quoted
end


#branch = readdlm("IEEE014.branch")
branch = readdlm("IEEE662.branch", String)
#branches = Array(Branch,size(branch,1))
for i in 1:size(branch,1)
    #branches[i] = Branch(branch[i,2:end]...,0,0)
    @assert branch[i,1] == string(i)
    #branches[i].to = busmap[branches[i].to]
    branch[i,2] = busmap[branch[i,2]]
    #branches[i].from = busmap[branches[i].from]
    branch[i,3] = busmap[branch[i,3]]
end

nbus = size(bus,1)
nbranch = size(branch,1)

NREP = 100

newbus = vcat([bus for i in 1:NREP]...)
newbranch = vcat([branch for i in 1:NREP]...)

@show size(bus)
@show size(newbus)

for i in 1:nbus
    for k in 1:NREP
        @assert parse(Int, newbus[i+(k-1)*nbus,1]) == i
        newbus[i+(k-1)*nbus,1] = string(i+(k-1)*nbus)
    end
end

for i in 1:nbranch
    for k in 1:NREP
        @assert 1 <= parse(Int, newbranch[i+(k-1)*nbranch,2]) <= nbus
        @assert 1 <= parse(Int, newbranch[i+(k-1)*nbranch,3]) <= nbus
        newbranch[i+(k-1)*nbranch,1] = string(i+(k-1)*nbranch)
        newbranch[i+(k-1)*nbranch,2] = string(parse(Int, newbranch[i+(k-1)*nbranch,2])+(k-1)*nbus)
        newbranch[i+(k-1)*nbranch,3] = string(parse(Int, newbranch[i+(k-1)*nbranch,3])+(k-1)*nbus)
    end
end

writedlm("IEEE66200.bus",newbus,' ')
writedlm("IEEE66200.branch",newbranch, ' ')

