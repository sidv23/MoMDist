using DrWatson
@quickactivate "wrips-code"

begin
    using Ripserer, PersistenceDiagrams, PersistenceDiagramsBase
    using Random, Distributions, Parameters, Pipe, ProgressMeter, Plots, StatsPlots, JLD2
    using Distances, LinearAlgebra, Statistics, LazySets, Roots, Hyperopt
    using ProgressMeter
    include(srcdir("wRips.jl"))
    import Main.wRips
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
    d = 5
    m = 50
    Random.seed!(2022)
    X1 = @pipe matern_circle(500,m, r=1.5,l=0.5, c=(2.0,-2.0))
    X2 = @pipe matern_circle(500,m, r=1.5,l=0.5, c=(-2.0,2.0))
    X = [Rot(X1, d); Rot(X2, d)]
    Xn = X |> wRips._ArrayOfTuples_to_ArrayOfVectors
end;

Lep = wRips.lepski_params(a=0.01, b=1, mmin=20, mmax=200, pi=1.1, δ=0.05)
m̂ = wRips.lepski(Xn=Xn, params=Lep)


begin
    p = 2
    Q = 2 * m + 1
    dnq = wRips.momdist(Xn, floor(Int, Q))
    w_momdist = wRips.fit(Xn, dnq)
    D = wRips.wrips(Xn, w = w_momdist, p = p, dim_max=1)
    # plot(D[2])

    dnm = wRips.dtm(Xn, 0.05)
    w_dtm = wRips.fit(Xn, dnm)
    Dnm = wRips.wrips(Xn, w = w_dtm, p = p, dim_max=1)
    # plot(Dnm[2])

    plot( plot(D[2]), plot(Dnm[2]) )
end