using DrWatson
@quickactivate "momdist"

begin
    using Ripserer, PersistenceDiagrams, PersistenceDiagramsBase
    using Random, Distributions, Parameters, Pipe, ProgressMeter, Plots, StatsPlots, JLD2
    using Distances, LinearAlgebra, Statistics, LazySets, Roots, LambertW

    import RobustTDA as rtda
end




begin
    Random.seed!(2022)
    n = 500
    m = rand([50:1:150]..., 1)[1]

    l = 1.5
    win = (-l, l, -l, l)
    noise = rtda.randMClust(m..., window=win, λ1=2, λ2=10, r=0.05)
    # noise = rtda.randUnif(m..., a = -1, b = 1)
    signal = 2.5 .* rtda.randCircle(n, sigma=0.05)


    X = [signal; noise]
    Xn_signal = rtda._ArrayOfTuples_to_ArrayOfVectors(signal)

    Xn = rtda._ArrayOfTuples_to_ArrayOfVectors(X)
end


X |> scatter

Random.seed!(2022)
Q = 2 * m + 1
dnq = rtda.momdist(Xn, floor(Int, Q))
w_momdist = rtda.fit(Xn, dnq)



theme(:dao)
plt = rtda.surfacePlot(-5:0.5:5, f=(x, y) -> rtda.fit([[x, y]], dnq), c=:plasma, alpha=0.5, colorbar=false)
signal_z = [tuple([x..., 0]...) for x in signal]
noise_z = [tuple([x..., 0]...) for x in noise]
plt = scatter(plt, signal_z, label=nothing, c=:orange)
plt = scatter(plt, noise_z, label=nothing, c=:dodgerblue)
savefig(plot(plt, size=(400, 300)), plotsdir("sublevel/sublevel.pdf"))


cols = colormap("Blues", 5)

plt3 = plot(rtda.Balls(Xn, rtda.rfx.(2.25, w_momdist, 1)), c=cols[5], linealpha=0, ratio=1, fillalpha=0.1)
plt3 = plot!(rtda.Balls(Xn, rtda.rfx.(2, w_momdist, 1)), c=cols[4], linealpha=0, fillalpha=0.1)
plt3 = plot!(rtda.Balls(Xn, rtda.rfx.(1.75, w_momdist, 1)), c=cols[3], linealpha=0, fillalpha=0.1)
plt3 = plot!(rtda.Balls(Xn, rtda.rfx.(1.5, w_momdist, 1)), c=cols[2], linealpha=0, fillalpha=0.1)
plt3 = scatter!(X, marker_z=w_momdist, label=nothing, c=:plasma, markeralpha=1, markersize=3, colorbar=true)
plt3 = plot(plt3, size=(400, 300))
savefig(plt3, plotsdir("sublevel/filtrations.pdf"))




theme(:default)

xseq = -3.5:0.05:3.5
G = [rtda.fit([[x, y]], dnq) for x in xseq, y in xseq]
D1 = G |> Cubical |> ripserer
plt1 = @pipe D1 |> plot(_, label=[L"H_0" L"H_1"], markersize=6, markeralpha=1)
title!("Sublevel Dgm")
savefig(plot(plt1, size=(400, 400)), plotsdir("sublevel/sublevel_dgm.pdf"))

D2 = rtda.wrips(Xn, w=w_momdist, p=1)
plt2 = @pipe D2 |> plot(_, label=[L"H_0" L"H_1"], markersize=6, markeralpha=1)
title!("Weighted-Offset Dgm")
savefig(plot(plt2, size=(400, 400)), plotsdir("sublevel/wrips_dgm.pdf"))




