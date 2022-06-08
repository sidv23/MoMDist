using DrWatson
@quickactivate "momdist"

begin
    using Plots, StatsPlots
    using Random, Distributions, Parameters, Pipe, ProgressMeter, JLD2
    using Distances, LinearAlgebra, Statistics, LazySets, LambertW, Hyperopt
    using Ripserer, PersistenceDiagrams, PersistenceDiagramsBase
    import Ripserer: persistence
    using Markdown, InteractiveUtils, KernelFunctions, MLDataUtils, Roots, JLD2, StatsBase, NearestNeighbors, LaTeXStrings

    plot_par = rtda.plot_params(alpha=0.3)
end


import RobustTDA as rtda


# Helper Functions


begin
    function RKDist_fit(Xn, s)
        N = length(Xn)
        kernel = SqExponentialKernel() ∘ ScaleTransform(1 / s)
        w = rtda.rkde_W(Xn, k=kernel)
        Ω = zeros(N, N)
        for i in 1:N, j in 1:N
            Ω[i, j] = w[i] * w[j] * kernel(Xn[i], Xn[j])
        end
        [sqrt(sum(Ω) + kernel(0, 0) - (2 * sum(Ω[i, :]))) for i in 1:N]
    end

    function rkdistEval(Xn, s, x)
        N = length(Xn)
        kernel = SqExponentialKernel() ∘ ScaleTransform(1 / s)
        w = rtda.rkde_W(Xn, k=kernel)

        Ω = zeros(N, N)
        for i in 1:N, j in 1:N
            Ω[i, j] = w[i] * w[j] * kernel(Xn[i], Xn[j])
        end

        return sqrt(sum(Ω) + kernel(0, 0) - (2 * ((w .* [kernel(x, y) for y in Xn]) |> sum)))
    end


    function KDist_fit(Xn, s)
        N = length(Xn)
        kernel = SqExponentialKernel() ∘ ScaleTransform(1 / s)
        w = ones(N) ./ N
        Ω = zeros(N, N)
        for i in 1:N, j in 1:N
            Ω[i, j] = w[i] * w[j] * kernel(Xn[i], Xn[j])
        end
        [sqrt(sum(Ω) + kernel(0, 0) - (2 * sum(Ω[i, :]))) for i in 1:N]
    end

    function plot_results(R; var1, var2, var3, ylab=nothing)

        plt = plot(
            m_seq,
            [map(x -> x[var1], R[i]) |> mean for i = 1:length(m_seq)],
            ribbon=[map(x -> x[var1], R[i]) |> mad for i = 1:length(m_seq)],
            label="mom", lw=2
        )

        plt = plot(plt,
            m_seq,
            [map(x -> x[var2], R[i]) |> mean for i = 1:length(m_seq)],
            ribbon=[map(x -> x[var2], R[i]) |> mad for i = 1:length(m_seq)],
            label="dtm", lw=2,
            legend=:bottomright
        )

        plt = plot(plt,
            m_seq,
            [map(x -> x[var3], R[i]) |> mean for i = 1:length(m_seq)],
            ribbon=[map(x -> x[var3], R[i]) |> mad for i = 1:length(m_seq)],
            label="rkde", lw=2,
            legend=:bottomright
        )
        ylabel!(ylab)
        xlabel!(L"# of outliers ($m$)")
        return plt
    end

end








# One Iteration

function one_simulation(; n=500, m, p, Q=100, K=nothing, s=nothing, l=0.001)

    noise = rtda.randUnif(m, a=-l, b=l)
    signal = 2.0 * rtda.randCircle(n, sigma=0.1)
    X = [signal; noise]
    Xn_signal = rtda._ArrayOfTuples_to_ArrayOfVectors(signal)
    Xn = rtda._ArrayOfTuples_to_ArrayOfVectors(X)


    if isnothing(K)
        K = 50
    end


    # Choosing the bandwidth of the RKDE
    if isnothing(s)
        tree = BruteTree(reduce(hcat, Xn), leafsize=K)
        nns = Any[]
        for j ∈ 1:length(Xn)
            push!(nns,
                knn(tree, Xn[j], K)[2] |> maximum
            )
        end
        s = median(nns)
    end


    dnm = rtda.dtm(Xn, K / length(Xn))
    dnq = rtda.momdist(Xn, Q)
    w_momdist = rtda.fit(Xn, dnq)
    w_dtm = rtda.fit(Xn, dnm)
    w_rkde = RKDist_fit(Xn, s)

    dnm0 = rtda.dtm(Xn_signal, K / length(Xn_signal))
    dnq0 = rtda.momdist(Xn_signal, Q)
    w0_momdist = rtda.fit(Xn_signal, dnq0)
    w0_dtm = rtda.fit(Xn_signal, dnm0)
    w0_rkde = KDist_fit(signal, s)

    Dn = ripserer(Xn_signal)
    D = ripserer(Xn)

    D0_dtm = rtda.wrips(Xn_signal, w=w0_dtm, p=p)
    D0_momdist = rtda.wrips(Xn_signal, w=w0_momdist, p=p)
    D0_rkde = rtda.wrips(signal, w=w0_rkde, p=p)

    D_dtm = rtda.wrips(Xn, w=w_dtm, p=p)
    D_momdist = rtda.wrips(Xn, w=w_momdist, p=p)
    D_rkde = rtda.wrips(Xn, w=w_rkde, p=p)

    results = Dict()

    push!(results, "bottleneck0 dtm" => Bottleneck()(D_dtm, D0_dtm))
    push!(results, "bottleneck dtm" => Bottleneck()(D_dtm, Dn))
    push!(results, "birth dtm" => rtda.fit([[0, 0]], dnm))
    push!(results, "pers dtm" => D_dtm[2] .|> persistence |> maximum)
    push!(results, "rpers dtm" => (D_dtm[2] .|> persistence |> maximum) / (D0_dtm[2] .|> persistence |> maximum))

    push!(results, "bottleneck0 mom" => Bottleneck()(D_momdist, D0_momdist))
    push!(results, "bottleneck mom" => Bottleneck()(D_momdist, Dn))
    push!(results, "birth mom" => rtda.fit([[0, 0]], dnq))
    push!(results, "pers mom" => D_momdist[2] .|> persistence |> maximum)
    push!(results, "rpers mom" => (D_momdist[2] .|> persistence |> maximum) / (D0_momdist[2] .|> persistence |> maximum))


    push!(results, "bottleneck0 rkde" => Bottleneck()(D_rkde, D0_rkde))
    push!(results, "bottleneck rkde" => Bottleneck()(D_rkde, Dn))
    push!(results, "birth rkde" => rkdistEval(Xn, s, [0.0, 0.0]))
    push!(results, "pers rkde" => D_rkde[2] .|> persistence |> maximum)
    push!(results, "rpers rkde" => (D_rkde[2] .|> persistence |> maximum) / (D0_rkde[2] .|> persistence |> maximum))
end







# Simulate Data

m_seq = [0:5:95; 100:25:150] |> collect
N = 5
Influence = [[Dict() for _ = 1:N] for _ in m_seq]

for i in 1:length(m_seq), j in 1:N
    Influence[i][j] = one_simulation(m=m_seq[i], p=1, K=50)
    println([i, j])
end

# jldsave(datadir("Influence.jld2"); Influence)







# Results

# R = load(datadir("Influence.jld2"), "Influence");
R = Influence


plot_results(
    R,
    var1="birth mom",
    var2="birth dtm",
    var3="birth rkde",
    ylab="Birth Time"
)


plot_results(
    R,
    var1="rpers mom",
    var2="rpers dtm",
    var3="rpers rkde",
    ylab="Persistence"
)


plot_results(
    R,
    var1="bottleneck0 mom",
    var2="bottleneck0 dtm",
    var3="bottleneck0 rkde",
    ylab="Bottleneck Influence"
)

