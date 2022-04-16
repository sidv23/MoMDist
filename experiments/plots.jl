using DrWatson
@quickactivate "wrips-code"

begin
    using Ripserer, PersistenceDiagrams, PersistenceDiagramsBase
    # using InteractiveUtils, MLDataUtils
    using Random, Distributions, Parameters, Pipe, ProgressMeter, Plots, StatsPlots, JLD2
    using Distances, LinearAlgebra, Statistics, LazySets, Roots, Hyperopt

    include(srcdir("wRips.jl"))
    import Main.wRips
end

res = JLD2.load("./experiments/data/res.jld2")["res"]

M = JLD2.load("./experiments/data/M4.jld2")["M"][:, [2, 4]]
Lep = JLD2.load("./experiments/Lep.jld2")["Lep"]
theme(:dao)
plt = boxplot(Lep, label = "Lepski")
plt = boxplot(plt, M[:, 1], label = "Total Persistence")
plt = boxplot(plt, M[:, 2], label = "Bottleneck")
# plt = plot(plt, legend = :topright, size = (300, 300), ylabel = "", xticks = nothing)
plt = plot(plt, legend = nothing, size = (300, 300), ylabel = "", xticks = (1:3, ["Lepski", "Total Pers", "Bottleneck"]), label = nothing)
savefig(plt, "./experiments/plots/p1-3.pdf")

begin
    Random.seed!(2021)

    n = 500
    m = rand([50:1:150]..., 1)

    l = 1.5
    win = (-l, l, -l, l)
    noise = wRips.randMClust(m..., window = win, λ1 = 2, λ2 = 10, r = 0.05)
    # noise = wRips.randUnif(m..., a = -1, b = 1)
    signal = 2.5 .* wRips.randCircle(n, sigma = 0.05)


    X = [signal; noise]
    Xn_signal = wRips._ArrayOfTuples_to_ArrayOfVectors(signal)

    Xn = wRips._ArrayOfTuples_to_ArrayOfVectors(X)
    Q_seq = range(50, 300, step = 10)
    nQ = length(Q_seq)
    num_iter = 10
end

begin
    S = [[Dict() for _ = 1:num_iter] for _ in Q_seq]

    prog = Progress(convert(Int, length(Q_seq) * m[1]))

    for i in 1:length(Q_seq)
        for j = 1:num_iter
            dnq = wRips.momdist(Xn, Q_seq[i])
            w_momdist = wRips.fit(Xn, dnq)
            D_momdist = wRips.wrips(Xn, w = w_momdist, p = 1)
            S[i][j] = Dict("dgm" => D_momdist)
            next!(prog)
        end
    end

end


Q_seq = range(10, 250, step = 10)
method1 = D -> D |> persistence |> maximum
y = [([@pipe d["dgm"][2] .|> persistence |> filter(x -> x > mean(_), _) |> sum for d in D] |> mean) for D in Dgms]
Δy = [([@pipe d["dgm"][2] .|> persistence |> filter(x -> x > mean(_), _) |> sum for d in D] |> std) for D in Dgms]


theme(:dao)
plot(Q_seq, y, ribbon = Δy, label = "Total Persistence")
plot!(repeat([128], 25), LinRange(1, 1.6, 25), label = L"$2m+1$", legend = :topright, size = (300, 300))
savefig("./experiments/plots/tp.pdf")

plt

nQ = length(S)
num_iter = length(S[1])
res = zeros(0, 2)

prog = Progress(convert(Int, nQ * num_iter * (num_iter - 1) / 2))
for k in 1:nQ
    Σ = Any[]
    for i in 1:num_iter-1, j in i+1:num_iter
        push!(Σ, Bottleneck()(S[k][i]["dgm"], S[k][j]["dgm"]))
        next!(prog)
    end
    res = vcat(res, @pipe Σ |> [mean(_), std(_)]')
end
# JLD2.save("./experiments/data/res.jld2"; res)
