using DrWatson
@quickactivate "momdist"

########################
# Initialize
########################

begin
    using Plots, StatsPlots
    using Random, Distributions, Parameters, Pipe, ProgressMeter, JLD2
    using Distances, LinearAlgebra, Statistics, LazySets, LambertW, Hyperopt
    using Ripserer, PersistenceDiagrams, PersistenceDiagramsBase

    # Settings
    plot_par = rtda.plot_params(alpha=0.3)
    reps = 30
end

import RobustTDA as rtda


########################
# Lepski's Method
########################


function one_lepski_iter(mmin=50, mmax=150; params)

    # Generate Random Data
    begin
        n = 500
        m = rand([mmin:1:mmax]..., 1)[1]

        # Signal
        signal = 2.5 .* rtda.randCircle(n, sigma=0.05)

        # Noise
        l = 1.5
        win = (-l, l, -l, l)
        noise = rtda.randMClust(m, window=win, λ1=2, λ2=10, r=0.05)

        # Concatenate
        X = [signal; noise] |> rtda._ArrayOfVectors_to_ArrayOfTuples
        Xn = X |> rtda._ArrayOfTuples_to_ArrayOfVectors
    end

    m̂ = rtda.lepski(Xn=Xn, params=params)
    return (m̂ - m) / m
end


begin
    # Initialize the Hyperparameters

    θ = rtda.lepski_params(
        a=0.2,
        b=1,
        mmin=50,
        mmax=200,
        pi=1.1,
        δ=0.05
    )

    Lep = Any[]

    Random.seed!(2022)

    prog = Progress(reps)
    for i in 1:reps
        # println("\n i=$i")
        Lep = vcat(Lep, one_lepski_iter(params=θ))
        next!(prog)
    end

    # jldsave(datadir("Lep.jld2"); Lep)
end


plt = boxplot(Lep, label="Lepski", legend=:topright)







########################
# Heuristic Method: Illustration
########################

function heuristic(Xn; Q, p=1)
    D1 = @pipe rtda.momdist(Xn, floor(Int, Q)) |> rtda.fit(Xn, _) |> rtda.wrips(Xn, w=_, p=p)
    D2 = @pipe rtda.momdist(Xn, floor(Int, Q)) |> rtda.fit(Xn, _) |> rtda.wrips(Xn, w=_, p=p)
    return Bottleneck()(D1, D2)
end


# Generate some test data
begin
    n = 500
    m = rand([50:1:150]..., 1)[1]

    l = 1.5
    win = (-l, l, -l, l)
    noise = rtda.randMClust(m..., window=win, λ1=2, λ2=10, r=0.05)
    signal = 2.5 .* rtda.randCircle(n, sigma=0.05)
    X = [signal; noise]
    Xn_signal = rtda._ArrayOfTuples_to_ArrayOfVectors(signal)
    Xn = rtda._ArrayOfTuples_to_ArrayOfVectors(X)

    scatter(X, ratio=1, label=nothing)
end


# Heuristic Method overview
begin
    Ms = 50:50:250
    N = length(Ms)

    H = zeros(N, reps)

    prog = Progress(N * 5)
    for i in 1:N, j in 1:5
        H[i, j] = heuristic(Xn; Q=2 * Ms[i] + 1, p=1)
        next!(prog)
    end
end


plot(
    Ms, mean(H, dims=2),
    ribbon=std(H, dims=2)
)



########################
# Heuristic Method: Simulation
########################

function one_iter(mmin=50, step=5, mmax=150; method=heuristic, loops=10)

    # Generate Random Data
    begin
        n = 500
        m = rand([mmin:1:mmax]..., 1)[1]

        # Signal
        signal = 2.5 .* rtda.randCircle(n, sigma=0.05)

        # Noise
        l = 1.5
        win = (-l, l, -l, l)
        noise = rtda.randMClust(m, window=win, λ1=2, λ2=10, r=0.05)

        # Concatenate
        X = [signal; noise] |> rtda._ArrayOfVectors_to_ArrayOfTuples
        Xn = X |> rtda._ArrayOfTuples_to_ArrayOfVectors
    end

    Opt = @hyperopt for i = loops,
        sampler = RandomSampler(),
        Q = (2*mmin+1):step:(2*mmax+1),
        b = [true, false]

        method(Xn, Q=Q, p=1)
    end

    return (round(Int, 0.5 * (Opt.minimizer[1] - 1)) - m) / m
end



begin
    Heu = Any[]

    Random.seed!(2022)
    for i in 1:reps
        # println("\n i=$i")
        Heu = vcat(Heu, one_iter(loops=50))
    end

    # jldsave(datadir("Heu.jld2"); Heu)
end

boxplot(plt, Heu, label="Heuristic")

