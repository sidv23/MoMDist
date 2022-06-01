using DrWatson
@quickactivate "momdist"


begin
    using Ripserer, PersistenceDiagrams, PersistenceDiagramsBase
    using Random, Distributions, Parameters, Pipe, ProgressMeter, Plots, StatsPlots, JLD2
    using Distances, LinearAlgebra, Statistics, LazySets, LambertW

    import RobustTDA as rtda

    plot_par = plot_params(alpha=0.3)
    theme(:dao)
end


function M1(Xn; Q, p=1)
    D1 = @pipe rtda.momdist(Xn, floor(Int, Q)) |> rtda.fit(Xn, _) |> rtda.wrips(Xn, w=_, p=p)
    D2 = @pipe rtda.momdist(Xn, floor(Int, Q)) |> rtda.fit(Xn, _) |> rtda.wrips(Xn, w=_, p=p)
    return Bottleneck()(D1, D2)
end


begin
    # Random.seed!(2022)
    n = 500
    m = rand([50:1:150]..., 1)[1]

    l = 1.5
    win = (-l, l, -l, l)
    noise = rtda.randMClust(m..., window=win, λ1=2, λ2=10, r=0.05)
    signal = 2.5 .* rtda.randCircle(n, sigma=0.05)
    X = [signal; noise]
    Xn_signal = rtda._ArrayOfTuples_to_ArrayOfVectors(signal)
    Xn = rtda._ArrayOfTuples_to_ArrayOfVectors(X)
end

begin
    reps = 30
    Ms = 5:5:250

    N = length(Ms)

    method1 = zeros(N, reps)
    
    prog = Progress( N * reps )
    for i in 1:N, j in 1:reps
        method1[i, j] = M1(Xn; Q = 2 * Ms[i] + 1, p=1)
        next!(prog)
    end
end

plot(
    Ms, mean(method1, dims=2),
    ribbon=std(method1, dims=2)
)