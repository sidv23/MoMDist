using DrWatson
@quickactivate "momdist"

using Ripserer, PersistenceDiagrams, PersistenceDiagramsBase
using Random, Distributions, Parameters, Pipe, ProgressMeter, Plots, StatsPlots, JLD2
using Distances, LinearAlgebra, Statistics, LazySets, Roots, LambertW, LaTeXStrings

import RobustTDA as rtda


# Helper Functions

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



# Generate points

begin
    Random.seed!(2022)
    m = 150
    n = 7 * m
    dim = 100
    R = randomRotation(d=dim)

    signal = interlockedCircles(n)
    signal = [Tuple(R * [x...; zeros(dim - 3)]) for x in signal]

    l = 0.23
    R = product_distribution(repeat([Uniform(-l, l)], dim))
    noise = [rand(R) for _ in 1:m]

    X = [signal; noise] |> rtda._ArrayOfVectors_to_ArrayOfTuples
    Xn = [[x...] for x in X]
end

# Example plot
begin
    dims = rand(1:1:100, 3)
    @pipe [x[dims] for x in X] |> scatter(_, ratio=1, label="Coordinates for dims: $dims")
end


# Lepski's method (coarser)
begin
    # Initialize Lepski Parameters
    θ = rtda.lepski_params(
        a=0.1,
        b=1,
        mmin=100,
        mmax=500,
        pi=1.15,
        δ=0.01
    )

    M = rtda.lepski(Xn=Xn, params=θ)
end


# Lepski's method (finer)
begin
    # Change pi
    θ = rtda.lepski_params(
        a=0.2,
        b=1,
        mmin=round(Int, 0.8 * M),
        mmax=round(Int, 1.2 * M),
        pi=1.07,
        δ=0.01
    )

    M = rtda.lepski(Xn=Xn, params=θ)
end


# Compute MoM-Dist and DTM
begin
    Q = 2 * M + 1

    Random.seed!(2022)
    dnq = rtda.momdist(Xn, floor(Int, Q))
    dnm = rtda.dtm(Xn, Q / n)

    w_momdist = rtda.fit(Xn, dnq)
    w_dtm = rtda.fit(Xn, dnm)

    D1 = rtda.wrips(Xn, w=w_momdist, p=1)
    D2 = rtda.wrips(Xn, w=w_dtm, p=1)
end


# Results
plot(
    plot(D1[2], ylim=(-0.5, 1.5), persistence=true),
    plot(D2[2], ylim=(-0.5, 1.5), persistence=true)
)