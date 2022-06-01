using DrWatson
@quickactivate "wrips-code"

function pltD(D; s=(300, 300), rotate=true)
    return plot(D, persistence=rotate, lim=(-0.01, 2), markershape=:o, markerstrokecolor=:black, markersize=5, markeralpha=1, size=s, title="")
end

begin
    using Ripserer, PersistenceDiagrams, PersistenceDiagramsBase
    using Random, Distributions, Parameters, Pipe, ProgressMeter, Plots, StatsPlots, JLD2
    using Distances, LinearAlgebra, Statistics, LazySets, Roots, Hyperopt
    using ProgressMeter
    import RobustTDA as wRips
end


function matern_circle(n, m; r=1.0, l=1.0, c=(0.0, 0.0))
    win = (-l, l, -l, l)
    noise = wRips.randMClust(m..., window=win, λ1=2, λ2=10, r=0.05)
    signal = @pipe (r .* wRips.randCircle(n, sigma=0.0)) 
    return @pipe [signal; noise] .|> _ .+ c
end

function rand_SO(d)
    A = rand(d, d)
    Q, R = qr(A)
    dt = det(Q)
    # while dt <= 0.0
    #     A = rand(d, d)
    #     Q, R = qr(A)
    #     dt = det(Q)
    # end
    return Q * Diagonal(sign.(diag(R)))
end


function Rot(X, d)
    Q = rand_SO(d)
    X = [tuple( (Q * [x..., zeros(d-2)...])...) for x in X]
    return X
end

begin
    d = 100
    m = 65
    Random.seed!(2022)
    X1 = @pipe matern_circle(500,m, r=1.5,l=0.5, c=(2.0,-2.0))
    X2 = @pipe matern_circle(500,m, r=1.5,l=0.5, c=(-2.0,2.0))
    X = [Rot(X1, d); Rot(X2, d)]
    Xn = X |> wRips._ArrayOfTuples_to_ArrayOfVectors
end;


begin
    Random.seed!(2022)
    Lep = wRips.lepski_params(a=0.05, b=1, mmin=20, mmax=200, pi=1.1, δ=0.05)
    m̂ = wRips.lepski(Xn=Xn, params=Lep)
end


begin
    # m̂ = m
    p = Inf
    # Q = 2 * m + 1
    Q = 2 * m̂ + 1
    dnq = wRips.momdist(Xn, floor(Int, Q))
    w_momdist = wRips.fit(Xn, dnq)
    D = wRips.wrips(Xn, w = w_momdist, p = p, dim_max=1)
    # plot(D[2])

    dnm = wRips.dtm(Xn, Q / length(Xn))
    w_dtm = wRips.fit(Xn, dnm)
    Dnm = wRips.wrips(Xn, w = w_dtm, p = p, dim_max=1)
    # plot(Dnm[2])

    # plot(plot(D[2], lim=(0.5, 2)), plot(Dnm[2], lim=(0.5, 2)))


    plot(
        pltD(D[2]),
        pltD(Dnm[2]),
        size=(800,400)
    )
end

plot(pltD(D[2]),pltD(Dnm[2]),size=(800, 400))
