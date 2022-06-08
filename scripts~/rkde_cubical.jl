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

xseq = -5:0.1:5
yseq = -5:0.1:5

G = makeGrid(xseq)

RK = rkdistFit(G, 2s, Xn[1:end])
heatmap(RK, c=:PuBu)

RK |> Cubical |> ripserer |> plot

