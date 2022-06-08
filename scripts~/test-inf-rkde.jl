function rkdistEval(Xn, s, x)
    N = length(Xn)
    kernel = SqExponentialKernel() ∘ ScaleTransform(1 / s)
    w = rtda.rkde_W(Xn, k=kernel)

    Fit = zeros(size(G))
    n, m = size(G)

    Ω = zeros(N, N)
    for i in 1:N, j in 1:N
        Ω[i, j] = w[i] * w[j] * kernel(Xn[i], Xn[j])
    end

    return sqrt(sum(Ω) + kernel(0, 0) - (2 * ((w .* [kernel(x, y) for y in Xn]) |> sum)))
end


function rkdistFit(G, s, Xn)
    N = length(Xn)
    kernel = SqExponentialKernel() ∘ ScaleTransform(1 / s)
    w = rtda.rkde_W(Xn, k=kernel)

    Fit = zeros(size(G))
    n, m = size(G)

    Ω = zeros(N, N)
    for i in 1:N, j in 1:N
        Ω[i, j] = w[i] * w[j] * kernel(Xn[i], Xn[j])
    end

    for i in 1:n, j in 1:m
        Fit[i, j] = sqrt(sum(Ω) + kernel(0, 0) - (2 * ((w .* [kernel(G[i, j], x) for x in Xn]) |> sum)))
    end
    return Fit
end


function kdistFit(G, s, Xn)
    N = length(Xn)
    kernel = SqExponentialKernel() ∘ ScaleTransform(1 / s)
    w = ones(N) ./ N

    Fit = zeros(size(G))
    n, m = size(G)

    Ω = zeros(N, N)
    for i in 1:N, j in 1:N
        Ω[i, j] = w[i] * w[j] * kernel(Xn[i], Xn[j])
    end

    for i in 1:n, j in 1:m
        Fit[i, j] = sqrt(sum(Ω) + kernel(0, 0) - (2 * ((w .* [kernel(G[i, j], x) for x in Xn]) |> sum)))
    end
    return Fit
end

function rkdeFit(G, s, Xn)
    N = length(Xn)
    kernel = SqExponentialKernel() ∘ ScaleTransform(1 / s)
    w = rtda.rkde_W(Xn, k=kernel)

    Fit = zeros(size(G))
    n, m = size(G)

    for i in 1:n, j in 1:m
        Fit[i, j] = (w .* [kernel(G[i, j], x) for x in Xn]) |> sum
    end
    return Fit
end


function makeGrid(xseq, yseq=nothing)
    if isnothing(yseq)
        yseq = xseq
    end
    G = [[[x,y] for x in xseq] for y in yseq]
    return hcat(G...)
end


begin
    n = 500
    m = 100

    noise = rtda.randUnif(m, a=-l, b=l)
    signal = 2.0 * rtda.randCircle(n, sigma=0.1)
    X = [signal; noise]
    Xn_signal = rtda._ArrayOfTuples_to_ArrayOfVectors(signal)
    Xn = rtda._ArrayOfTuples_to_ArrayOfVectors(X)

    K = round(Int, Q)

    tree = BruteTree(reduce(hcat, Xn), leafsize=K)
    nns = Any[]
    for j ∈ 1:length(Xn)
        push!(nns,
            knn(tree, Xn[j], K)[2] |> maximum
        )
    end
    s = median(nns)


    dnm = rtda.dtm(Xn, K / length(Xn))
    dnq = rtda.momdist(Xn, Q)
    w_momdist = rtda.fit(Xn, dnq)
    w_dtm = rtda.fit(Xn, dnm)
    w_rkde = RKDE_fit(Xn, s)

    dnm0 = rtda.dtm(Xn_signal, K / length(Xn_signal))
    dnq0 = rtda.momdist(Xn_signal, Q)
    w0_momdist = rtda.fit(Xn_signal, dnq0)
    w0_dtm = rtda.fit(Xn_signal, dnm0)
    w0_rkde = KDE_fit(signal, s)

    Dn = ripserer(Xn_signal)
    D = ripserer(Xn)

    D0_dtm = rtda.wrips(Xn_signal, w=w0_dtm, p=p)
    D0_momdist = rtda.wrips(Xn_signal, w=w0_momdist, p=p)
    D0_rkde = rtda.wrips(signal, w=w0_rkde, p=p)

    D_dtm = rtda.wrips(Xn, w=w_dtm, p=p)
    D_momdist = rtda.wrips(Xn, w=w_momdist, p=p)
    D_rkde = rtda.wrips(Xn, w=w_rkde, p=p)

    [
        Bottleneck()(D_momdist, D0_momdist),
        Bottleneck()(D_dtm, D0_dtm),
        Bottleneck()(D_rkde, D0_rkde)
    ]

    # [
    #     Bottleneck()(D_momdist[2], Dn[2]),
    #     Bottleneck()(D_dtm[2], Dn[2]),
    #     Bottleneck()(D_rkde[2], Dn[2])
    # ]

    # [
    #     D_momdist[2] .|> persistence |> maximum,
    #     D_dtm[2] .|> persistence |> maximum,
    #     D_rkde[2] .|> persistence |> maximum
    # ]
    
    # xseq = -5:0.1:5
    # yseq = -5:0.1:5

    # G = makeGrid(xseq)

    # RK = rkdistFit(G, s, Xn[1:end])
    # heatmap(RK, c=:PuBu)

    # @pipe RK |> Cubical |> ripserer |> plot


end