using DrWatson
@quickactivate "momdist"

using Random, Distributions, Plots, Parameters, Distances, LinearAlgebra, Pipe, Statistics, RCall, Ripserer, LazySets, PersistenceDiagrams, PersistenceDiagramsBase, Markdown, InteractiveUtils, KernelFunctions, MLDataUtils, Roots, JLD2, StatsBase

import RobustTDA as rtda

plot_par = rtda.plot_params(alpha=0.3)
theme(:dao)

n = 500
m = 50
Q = 100
l = 2.5
ϵ = 0.5


function one_simulation(; n=500, m, p, Q=100, K=nothing, l=0.1)

    if isnothing(K)
        K = round(Int, Q)
    end

    noise = rtda.randUnif(m, a=-l, b=l)
    signal = 2.0 * rtda.randCircle(n, sigma=0.1)
    X = [signal; noise]
    Xn_signal = rtda._ArrayOfTuples_to_ArrayOfVectors(signal)
    Xn = rtda._ArrayOfTuples_to_ArrayOfVectors(X)

    dnm = rtda.dtm(Xn, K / length(Xn))
    dnq = rtda.momdist(Xn, Q)
    w_momdist = rtda.fit(Xn, dnq)
    w_dtm = rtda.fit(Xn, dnm)

    dnm0 = rtda.dtm(Xn_signal, K / length(Xn_signal))
    dnq0 = rtda.momdist(Xn_signal, Q)
    w0_momdist = rtda.fit(Xn_signal, dnq0)
    w0_dtm = rtda.fit(Xn_signal, dnm0)

    Dn = ripserer(Xn_signal)
    D = ripserer(Xn)

    D0_dtm = rtda.wrips(Xn_signal, w=w0_dtm, p=p)
    D0_momdist = rtda.wrips(Xn_signal, w=w0_momdist, p=p)

    D_dtm = rtda.wrips(Xn, w=w_dtm, p=p)
    D_momdist = rtda.wrips(Xn, w=w_momdist, p=p)

    results = Dict()

    push!(results, "bottleneck" => Bottleneck()(D, Dn))
    push!(results, "birth" => rtda.fit([[0, 0]], rtda.dist(Xn)))
    push!(results, "pers" => (Dn[2] .|> persistence |> maximum))
    push!(results, "rpers" => (D[2] .|> persistence |> maximum) / (Dn[2] .|> persistence |> maximum))


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
end


m_seq = [0:5:95; 100:25:200] |> collect
N = 10
R = [[Dict() for _ = 1:N] for _ in m_seq]

for i in 1:length(m_seq), j in 1:N
    R[i][j] = one_simulation(m=m_seq[i], p=1)
    println([i, j])
end

jldsave(datadir("R-alt2.jld2"); R)
# R = load(datadir("R-alt2.jld2"), "R");




# Bottleneck Distance

begin
    var1 = "bottleneck0 mom"
    var2 = "bottleneck0 dtm"

    plt1 = plot(
        m_seq,
        [map(x -> x[var1], R[i]) |> mean for i = 1:length(m_seq)],
        ribbon=[map(x -> x[var1], R[i]) |> mad for i = 1:length(m_seq)],
        label="mom", lw=2
    )

    plt1 = plot(plt1,
        m_seq,
        [map(x -> x[var2], R[i]) |> mean for i = 1:length(m_seq)],
        ribbon=[map(x -> x[var2], R[i]) |> mad for i = 1:length(m_seq)],
        label="dtm", lw=2,
        legend=:bottomright
    )

    var3 = "bottleneck"
    plt1 = plot(plt1,
        m_seq,
        [map(x -> x[var3], R[i]) |> mean for i = 1:length(m_seq)],
        ribbon=[map(x -> x[var3], R[i]) |> mad for i = 1:length(m_seq)],
        label="vanilla", lw=2,
        legend=:bottomright
    )
    ylabel!("Bottleneck Influence")
    xlabel!(L"# of outliers ($m$)")
    plot(plt1, size=(400, 300))
end

savefig(plotsdir("influence/influence-bottleneck.pdf"))





# Relative Bottleneck Distance

begin
    var1 = "bottleneck mom"
    var2 = "bottleneck dtm"

    plt1 = plot(
        m_seq,
        [map(x -> x[var1], R[i]) |> mean for i = 1:length(m_seq)],
        ribbon=[map(x -> x[var1], R[i]) |> mad for i = 1:length(m_seq)],
        label="mom", lw=2
    )

    plt1 = plot(plt1,
        m_seq,
        [map(x -> x[var2], R[i]) |> mean for i = 1:length(m_seq)],
        ribbon=[map(x -> x[var2], R[i]) |> mad for i = 1:length(m_seq)],
        label="dtm", lw=2,
        legend=:bottomright
    )

    var3 = "bottleneck"
    plt1 = plot(plt1,
        m_seq,
        [map(x -> x[var3], R[i]) |> mean for i = 1:length(m_seq)],
        ribbon=[map(x -> x[var3], R[i]) |> mad for i = 1:length(m_seq)],
        label="vanilla", lw=2,
        legend=:bottomright
    )
    ylabel!("Bottleneck Influence")
    xlabel!(L"# of outliers ($m$)")
    plot(plt1, size=(400, 300))
end

savefig(plotsdir("influence/influence-bottleneck0.pdf"))



begin

    # Total Persistence
    var1 = "rpers mom"
    var2 = "rpers dtm"

    plt1 = plot(
        m_seq,
        [map(x -> x[var1], R[i]) |> mean for i = 1:length(m_seq)],
        ribbon=[map(x -> x[var1], R[i]) |> mad for i = 1:length(m_seq)],
        label="mom", lw=2
    )

    plt1 = plot(plt1,
        m_seq,
        [map(x -> x[var2], R[i]) |> mean for i = 1:length(m_seq)],
        ribbon=[map(x -> x[var2], R[i]) |> mad for i = 1:length(m_seq)],
        label="dtm", lw=2
    )

    var3 = "rpers"
    plt1 = plot(plt1,
        m_seq,
        [map(x -> x[var3], R[i]) |> mean for i = 1:length(m_seq)],
        ribbon=[map(x -> x[var3], R[i]) |> mad for i = 1:length(m_seq)],
        label="vanilla", lw=2,
        legend=:bottomright
    )
    # title!(L"Influence : Total Persistence in $H_1$")
    ylabel!("max-persistence for H₁")
    xlabel!(L"# of outliers ($m$)")
    plot(plt1, size=(400, 300))
end

savefig(plotsdir("influence/influence-pers.pdf"))


# Birth Time

begin
    var1 = "birth mom"
    var2 = "birth dtm"

    plt1 = plot(
        m_seq,
        [map(x -> x[var1], R[i]) |> mean for i = 1:length(m_seq)],
        ribbon=[map(x -> x[var1], R[i]) |> mad for i = 1:length(m_seq)],
        label="mom", lw=3,markershape=:o
    )

    plt1 = plot(plt1,
        m_seq,
        [map(x -> x[var2], R[i]) |> mean for i = 1:length(m_seq)],
        ribbon=[map(x -> x[var2], R[i]) |> mad for i = 1:length(m_seq)],
        label="dtm", lw=3,markershape=:o
    )

    var3 = "birth"
    plt1 = plot(plt1,
        m_seq,
        [map(x -> x[var3], R[i]) |> mean for i = 1:length(m_seq)],
        ribbon=[map(x -> x[var3], R[i]) |> mad for i = 1:length(m_seq)],
        label="dist", lw=2,markershape=:o,
        legend=:topright
    )

    ylabel!("b({x₀}): birth time of x₀")
    xlabel!(L"# of outliers ($m$)")
    plot(plt1, size=(400, 300))
end

savefig(plotsdir("influence/influence-birth.pdf"))


