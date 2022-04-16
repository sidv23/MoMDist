using DrWatson
@quickactivate "wrips-code"

begin
    using Ripserer, PersistenceDiagrams, PersistenceDiagramsBase
    # using InteractiveUtils, MLDataUtils
    using Random, Distributions, Parameters, Pipe, ProgressMeter, Plots, StatsPlots, JLD2
    using Distances, LinearAlgebra, Statistics, LazySets, Roots, Hyperopt, LambertW

    include(srcdir("wRips.jl"))
    import Main.wRips
end




begin
    # Random.seed!(2022)
    n = 500
    m = rand([50:1:150]..., 1)[1]

    l = 1.5
    win = (-l, l, -l, l)
    noise = wRips.randMClust(m..., window = win, λ1 = 2, λ2 = 10, r = 0.05)
    # noise = wRips.randUnif(m..., a = -1, b = 1)
    signal = 2.5 .* wRips.randCircle(n, sigma = 0.05)


    X = [signal; noise]
    Xn_signal = wRips._ArrayOfTuples_to_ArrayOfVectors(signal)

    Xn = wRips._ArrayOfTuples_to_ArrayOfVectors(X)
end


X |> scatter

Q = 2 * m + 1
dnq = wRips.momdist(Xn, floor(Int, Q))
w_momdist = wRips.fit(Xn, dnq)


plt = wRips.surfacePlot(-5:0.5:5, f = (x, y) -> wRips.fit([[x, y]], dnq), c = :plasma, alpha = 0.5, colorbar = false)
signal_z = [tuple([x..., 0]...) for x in signal]
noise_z = [tuple([x..., 0]...) for x in noise]
plt = scatter(plt, signal_z, label = nothing, c = :orange)
plt = scatter(plt, noise_z, label = nothing, c = :dodgerblue)
savefig(plot(plt, size = (400, 300)), "./experiments/plots/sublevel.pdf")



xseq = -3.5:0.05:3.5
G = [wRips.fit([[x, y]], dnq) for x in xseq, y in xseq]
D1 = G |> Cubical |> ripserer
plt1 = @pipe D1 |> plot(_, label = [L"H_0" L"H_1"])
savefig(plot(plt1, size = (400, 400)), "./experiments/plots/sublevel_dgm.pdf")

D2 = wRips.wrips(Xn, w = w_momdist, p = 1)
plt2 = @pipe D2 |> plot(_, label = [L"H_0" L"H_1"])
savefig(plot(plt2, size = (400, 400)), "./experiments/plots/wrips_dgm.pdf")

cols = colormap("Blues", 5)

plt3 = plot(wRips.Balls(Xn, wRips.rfx.(2.5, w_momdist, 1)), c = cols[5], linealpha = 0, ratio = 1, fillalpha = 0.1)
plt3 = plot!(wRips.Balls(Xn, wRips.rfx.(2, w_momdist, 1)), c = cols[4], linealpha = 0, fillalpha = 0.1)
plt3 = plot!(wRips.Balls(Xn, wRips.rfx.(1.5, w_momdist, 1)), c = cols[3], linealpha = 0, fillalpha = 0.1)
plt3 = plot!(wRips.Balls(Xn, wRips.rfx.(1.25, w_momdist, 1)), c = cols[2], linealpha = 0, fillalpha = 0.1)
plt3 = scatter!(X, marker_z = w_momdist, label = nothing, c = :plasma, markeralpha = 1, markersize = 3, colorbar = true)
plt3 = plot(plt3, size = (400, 300))

savefig(plt3, "./experiments/plots/filtrations.pdf")


