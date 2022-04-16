using DrWatson
@quickactivate "projectdir()"


# Load Packages
begin
    using PersistenceDiagrams, PersistenceDiagramsBase, Ripserer
    using Distributions, Distances, JLD2, LinearAlgebra, Parameters, Pipe, Plots
    using LazySets, LambertW, ProgressMeter, Random, Statistics, StatsPlots

    import RobustTDA as rtda

    plot_par = rtda.plot_params(alpha=0.3)
    theme(:dao)
end



# Generate Data
begin
    Random.seed!(2022)
    m = 150
    n = 7 * m

    signal = 1.5 .* rtda.randCircle(n, sigma=0.02)
    l = 1
    win = (-l, l, -l, l)
    noise = rtda.randMClust(m..., window=win, λ1=10, λ2=20, r=0.05)

    X = [signal; noise]
    Xn_signal = rtda._ArrayOfTuples_to_ArrayOfVectors(signal)
    Xn = rtda._ArrayOfTuples_to_ArrayOfVectors(X)
    scatter(X, label=nothing, ratio=1)
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
    Q = M + 1
    dnq = rtda.momdist(Xn, floor(Int, Q))
    # dnq = rtda.dtm(Xn, 0.1)
    w_momdist = rtda.fit(Xn, dnq)
    D = rtda.wrips(Xn, w=w_momdist, p=1)
    scatter(X, marker_z=w_momdist, label=nothing, ratio=1)
end


plot(D)
rtda.filtration_plot(median(w_momdist) + 0.2; Xn=Xn, w=w_momdist, p=2, par=plot_par)