using DrWatson
@quickactivate "momdist"

using Random, Distributions, Plots, Parameters, Distances, LinearAlgebra, Pipe, Statistics, RCall, Ripserer, LazySets, PersistenceDiagrams, PersistenceDiagramsBase, Markdown, InteractiveUtils, KernelFunctions, MLDataUtils, Roots, JLD2

import RobustTDA as rtda

plot_par = rtda.plot_params(alpha=0.3)
theme(:dao)

n = 500
m = 50
Q = 100
l = 2.5
ϵ = 0.5


function one_simulation(; n=500, m, p, Q=100, K=10, l=0.1)

    noise = rtda.randUnif(m, a=-l, b=l)
    signal = 2.0 * rtda.randCircle(n, sigma=0.1)
    X = [signal; noise]
    Xn_signal = rtda._ArrayOfTuples_to_ArrayOfVectors(signal)
    Xn = rtda._ArrayOfTuples_to_ArrayOfVectors(X)

    dnm = rtda.dtm(Xn, K / length(Xn))
    dnq = rtda.momdist(Xn, Q)

    w_momdist = rtda.fit(Xn, dnq)
    
    w_dtm = rtda.fit(Xn, dnm)

    Dn = ripserer(Xn_signal)
    D = ripserer(Xn)
    D_dtm = rtda.wrips(Xn, w=w_dtm, p=p)
    D_momdist = rtda.wrips(Xn, w=w_momdist, p=p)

    results = Dict()

    push!(results, "bottleneck" => Bottleneck()(D, Dn))
    push!(results, "pers dtm" => D_dtm[2] .|> persistence |> maximum)
    push!(results, "birth dtm" => rtda.fit([[0, 0]], dnm))
    push!(results, "bottleneck dtm" => Bottleneck()(D_dtm, Dn))
    push!(results, "pers mom" => D_momdist[2] .|> persistence |> maximum)
    push!(results, "birth mom" => rtda.fit([[0, 0]], dnq))
    push!(results, "bottleneck mom" => Bottleneck()(D_momdist, Dn))
end


m_seq = 0:20:200 |> collect
N = 10
R = [[Dict() for _ = 1:N] for _ in m_seq]

for i in 1:length(m_seq), j in 1:N
    R[i][j] = one_simulation(m=m_seq[i], p=1, K=10)
    println([i, j])
end

# jldsave(datadir("R.jld2"); R)
# R = load(datadir("R.jld2"), "R")




# Bottleneck Distance

begin
    var1 = "bottleneck mom"
    var2 = "bottleneck dtm"
    var3 = "bottleneck"

    plt1 = plot(
        m_seq,
        [map(x -> x[var1], R[i]) |> mean for i = 1:length(m_seq)],
        ribbon=[map(x -> x[var1], R[i]) |> std for i = 1:length(m_seq)],
        label="mom", lw=2
    )

    plt1 = plot(plt1,
        m_seq,
        [map(x -> x[var2], R[i]) |> mean for i = 1:length(m_seq)],
        ribbon=[map(x -> x[var2], R[i]) |> std for i = 1:length(m_seq)],
        label="dtm", lw=2,
        legend=:bottomright
    )
    plt1 = plot(plt1,
        m_seq,
        [map(x -> x[var3], R[i]) |> mean for i = 1:length(m_seq)],
        ribbon=[map(x -> x[var3], R[i]) |> std for i = 1:length(m_seq)],
        label="vanilla", lw=2,
        legend=:bottomright
    )
    ylabel!("Bottleneck Influence")
    xlabel!(L"# of outliers ($m$)")
    plot(plt1, size=(400, 300))
end

savefig(plotsdir("influence/influence-bottleneck.pdf"))



begin

    # Total Persistence
    var1 = "pers mom"
    var2 = "pers dtm"

    plt1 = plot(
        m_seq,
        [map(x -> x[var1], R[i]) |> mean for i = 1:length(m_seq)],
        ribbon=[map(x -> x[var1], R[i]) |> std for i = 1:length(m_seq)],
        label="mom", lw=2
    )

    plt1 = plot(plt1,
        m_seq,
        [map(x -> x[var2], R[i]) |> mean for i = 1:length(m_seq)],
        ribbon=[map(x -> x[var2], R[i]) |> std for i = 1:length(m_seq)],
        label="dtm", lw=2
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
        ribbon=[map(x -> x[var1], R[i]) |> std for i = 1:length(m_seq)],
        label="mom", lw=3
    )

    plt1 = plot(plt1,
        m_seq,
        [map(x -> x[var2], R[i]) |> mean for i = 1:length(m_seq)],
        ribbon=[map(x -> x[var2], R[i]) |> std for i = 1:length(m_seq)],
        label="dtm", lw=3
    )

    ylabel!("b({x₀}): birth time of x₀")
    xlabel!(L"# of outliers ($m$)")
    plot(plt1, size=(400, 300))
end

savefig(plotsdir("influence/influence-birth.pdf"))


