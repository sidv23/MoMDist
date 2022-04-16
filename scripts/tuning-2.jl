using DrWatson
@quickactivate


# Load Packages
begin
    using PersistenceDiagrams, PersistenceDiagramsBase, Ripserer
    using Distributions, Distances, JLD2, LinearAlgebra, Parameters, Pipe, Plots
    using LazySets, LambertW, ProgressMeter, Random, Statistics, StatsPlots

    import RobustTDA as rtda

    plot_par = rtda.plot_params(alpha=0.3)
    theme(:dao)
end


function randomRotation(; d=3)
    A = randn(d, d)
    M = (A + A') ./ √2
    _, U = eigen(M)
    return U
end

function interlockedCircles(n; args...)
    R = [1 0 0; 0 0 -1; 0 1 0]
    X1 = [[x...; 0] for x in rtda.randCircle(n)]
    X2 = [R * ([x...; 0] .+ [1.0, 0.0, 0.0]) for x in rtda.randCircle(n)]
    return [X1; X2]
end

scatter(X)

# Generate Data
begin
    Random.seed!(2022)
    m = 150
    n = 7 * m
    dim = 100
    R = randomRotation(d=dim)

    # signal1 = 1.5 .* rtda.randCircle(n, sigma=0.02)
    # signal2 = 1.5 .* rtda.randCircle(n, sigma=0.02)
    # signal1 = [x .+ (1.0, -1.0) for x in 1.5 .* rtda.randCircle(n, sigma=0.02)]
    # signal2 = [x .+ (-1.0, 1.0) for x in 1.5 .* rtda.randCircle(n, sigma=0.02)]
    # signal = [signal1; signal2]

    signal = interlockedCircles(n)
    signal = [Tuple(R * [x...; zeros(dim - 3)]) for x in signal]


    l = 1
    win = (-l, l, -l, l)
    # R = product_distribution(repeat([Uniform(-1, 1)], dim))
    R = product_distribution(repeat([Uniform(-1, 1)], dim))
    noise = [rand(R) for _ in 1:m]
    # noise = rtda.randMClust(m..., window=win, λ1=10, λ2=20, r=0.05)
    # noise = [Tuple(R * [x...; zeros(dim - 2)]) for x in noise]

    X = [signal; noise]
    Xn = [[x...] for x in X]
end


# Initialize Lepski Parameters
θ = rtda.lepski_params(
    a=0.1,
    b=1,
    mmin=100,
    mmax=500,
    pi=1.15,
    δ=0.02
)

# Calibration
M = rtda.lepski(Xn=Xn, params=θ)

# Refined Lepski
θ = rtda.lepski_params(
    a=0.2,
    b=1,
    mmin=round(Int, 0.8 * M),
    mmax=round(Int, 1.2 * M),
    pi=1.07,
    δ=0.01
)
M = rtda.lepski(Xn=Xn, params=θ)


begin
    Q = 2 * M + 1
    dnq = rtda.momdist(Xn, floor(Int, Q))
    dnm = rtda.dtm(Xn, Q / n)
    w_momdist = rtda.fit(Xn, dnq)
    w_dtm = rtda.fit(Xn, dnm)
    D1 = rtda.wrips(Xn, w=w_momdist, p=1)
    D2 = rtda.wrips(Xn, w=w_dtm, p=1)
end

savefig(
    plot(D1[2], markeralpha=1, title="MoM Dist", size=(300, 330)),
    plotsdir("highdim/interlocked-momdist.svg")
)

savefig(
    plot(D2[2], markeralpha=1, title="DTM", size=(300, 330)),
    plotsdir("highdim/interlocked-dtm.svg")
)

theme(:default)
plt = scatter(Tuple.(interlockedCircles(n)), ratio=1, label=nothing, size=(300,300), camera=(5,50))
savefig(plt, plotsdir("highdim/interlocked-scatter.pdf"))