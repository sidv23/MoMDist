using DrWatson
@quickactivate "wrips-code"

begin
    using Ripserer, PersistenceDiagrams, PersistenceDiagramsBase
    using Random, Distributions, Parameters, Pipe, ProgressMeter, Plots, StatsPlots, JLD2
    using Distances, LinearAlgebra, Statistics, LazySets, Roots, Hyperopt
    using DifferentialEquations
    include(srcdir("wRips.jl"))
    import Main.wRips
end


function parameterized_lorenz!(du,u,p,t)
    x, y, z = u
    σ, ρ, β, = p
    du[1] = dx = σ*(y-x)
    du[2] = dy = x*(ρ-z) - y
    du[3] = dz = x*y - β*z 
end


u0 = [1.0, 7.0, 0.0]
tspan = (0.0, 100.0)
p = [13.0, 24.0, 7/3]
prob = ODEProblem(parameterized_lorenz!, u0, tspan, p)
sol = solve(prob)


plot5 = plot(sol, vars = (1,2,3), xlabel="x",  ylabel="y", zlabel="z",label="x against y against z",title="ρ = 24")


begin 
    t = sol.t
    x = [x[1] for x in sol.u]
    y = [rand() < 0.2 ? u * randn() : u for u in x]
    plot(t, y)

    plot(
        scatter( y[1:(end-2)], y[2:(end-1)], y[3:end] ),
        scatter( x[1:(end-2)], x[2:(end-1)], x[3:end] )
    )

    Xmat = [y[1:(end-2)] y[2:(end-1)] y[3:end]]
    X = [tuple(x...) for x in eachrow(Xmat)]
    Xn = wRips._ArrayOfTuples_to_ArrayOfVectors(X)

    m = 0.1 * length(Xn)
    Q = 2 * m + 1
    dnq = wRips.momdist(Xn, floor(Int, Q))
    w_momdist = wRips.fit(Xn, dnq)

    D = wRips.wrips(Xn, w = w_momdist, p = 1, reps=true,alg=:involuted)
    @pipe D[2] |> plot(_,)

    plt1 = scatter(X)
    plot!(D[2][end], X, linewidth=3, label="Most persistent")
    plot!(D[2][end-1], X, linewidth=3, label="2nd most pertsistent")
end


function extract_vertices(A)
    list = Any[]
    for a in (@pipe A .|> Ripserer.vertices)
        push!(list, a...)
    end
    return list |> unique
end

extract_vertices(D[1][end-1].representative)