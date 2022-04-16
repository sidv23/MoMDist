using DrWatson
@quickactivate projectdir()


begin
    using Ripserer, PersistenceDiagrams, PersistenceDiagramsBase
    # using InteractiveUtils, MLDataUtils
    using Random, Distributions, Parameters, Pipe, ProgressMeter, Plots, StatsPlots, JLD2
    using Distances, LinearAlgebra, Statistics, LazySets, LambertW

    import RobustTDA as rtda
end

plot_par = plot_params(alpha = 0.3)
theme(:dao)

function M2(Xn; Q, p = 1, dim = 2)
    dnq = rtda.momdist(Xn, floor(Int, Q))
    w_momdist = rtda.fit(Xn, dnq)
    D = rtda.wrips(Xn, w = w_momdist, p = p)
    pers = (dim == 1 ? D[dim][1:end-1] : D[dim]) .|> persistence
    return filter( x -> x >= mean(x), pers ) |> mean
end

function M1(Xn; Q, p = 1, dim = 2)
    D1 = @pipe rtda.momdist(Xn, floor(Int, Q)) |> rtda.fit(Xn, _) |> rtda.wrips(Xn, w = _, p = p)
    D2 = @pipe rtda.momdist(Xn, floor(Int, Q)) |> rtda.fit(Xn, _) |> rtda.wrips(Xn, w = _, p = p)
    return Bottleneck()(D1, D2)
end

# function M3(Xn; Q, p = 1, dim = 2)
#     dnq = rtda.momdist(Xn, floor(Int, Q))
#     w_momdist = rtda.fit(Xn, dnq)
#     D = rtda.wrips(Xn, w = w_momdist, p = p)
#     return (dim == 1 ? D[dim][1:end-1] : D[dim]) .|> persistence |> std
# end

# function M4(Xn; Q, p = 1, dim = 2)
#     dnq = rtda.momdist(Xn, floor(Int, Q))
#     w_momdist = rtda.fit(Xn, dnq)
#     D = rtda.wrips(Xn, w = w_momdist, p = p)
#     pers = (dim == 1 ? D[dim][1:end-1] : D[dim]) .|> persistence
#     return filter(x -> x ≥ median(pers), pers) |> sum
# end


function one_iter(; M, epochs = 10)

    begin
        # Random.seed!(2022)
        n = 500
        m = rand([50:1:150]..., 1)[1]

        l = 1.5
        win = (-l, l, -l, l)
        noise = rtda.randMClust(m..., window = win, λ1 = 2, λ2 = 10, r = 0.05)
        # noise = rtda.randUnif(m..., a = -1, b = 1)
        signal = 2.5 .* rtda.randCircle(n, sigma = 0.05)


        X = [signal; noise]
        Xn_signal = rtda._ArrayOfTuples_to_ArrayOfVectors(signal)

        Xn = rtda._ArrayOfTuples_to_ArrayOfVectors(X)
    end

    begin
        Q_seq = range(10, 300, step = 10)
        nQ = length(Q_seq)
    end

    H = @hyperopt for i = epochs,
        sampler = RandomSampler(),
        Q = 10:1:300,
        b = [true, false]
        # println(i, "\t $Q", "\t     ")
        M(Xn, Q = Q, p = 1, dim = 2)
    end

    return (round(Int, 0.5 * (H.minimizer[1]-1)) - m) / m
end



function lepski(; Xn, mmin, mmax, pi = 1.1, a, b, p = 1, δ = 0.5)
    n = length(Xn)
    δmin = exp(-8 * (1 + b) * (2 * mmin + 1))

    h = (n, m, δ) ->
        2 * ((2 * m + 1) / (a * n) * lambertw((n * δmin) / (2 * m + 1)))^(1 / b) +
        ((1) / (a * (n - m)) * lambertw((n - m) / ((δ - δmin)^4)))^(1 / b)

    M = [round(Int, mmin * pi^j) for j in 1:1:floor(Int, log(pi, mmax / mmin))] |> unique
    Q = [2 * m + 1 for m in M]
    J = length(M)
    D = @pipe Q .|> rtda.momdist(Xn, _) .|> rtda.fit(Xn, _) .|> rtda.wrips(Xn, w = _, p = 1)

    jhat = J

    prog = Progress(convert(Int, J * (J - 1) / 2))
    generate_showvalues(j) = () -> [(:m, M[j])]

    for j in 1:(J-1)
        flag = false
        # Dj = @pipe Q[j] |> rtda.momdist(Xn, _) |> rtda.fit(Xn, _) |> rtda.wrips(Xn, w = _, p = 1)
        for i in (j+1):J
            # Di = @pipe Q[i] |> rtda.momdist(Xn, _) |> rtda.fit(Xn, _) |> rtda.wrips(Xn, w = _, p = 1)
            # flag = Bottleneck()(Di, Dj) > 2 * h(n, M[i])
            flag = Bottleneck()(D[i], D[j]) ≤ 2 * h(n, M[i], δ)
            next!(prog; showvalues = generate_showvalues(j))
        end

        if flag
            jhat = j
            break
        end
    end

    return M[jhat]
end

function one_lepski_iter(; mmin, mmax, pi = 1.1, a, b, p = 1, δ = 1e-9)
    begin
        n = 500
        m = rand([mmin:1:mmax]..., 1)[1]

        l = 1.5
        win = (-l, l, -l, l)
        noise = rtda.randMClust(m, window = win, λ1 = 2, λ2 = 10, r = 0.05)
        # noise = rtda.randUnif(m..., a = -1, b = 1)
        signal = 2.5 .* rtda.randCircle(n, sigma = 0.05)

        X = [signal; noise]
        Xn = rtda._ArrayOfTuples_to_ArrayOfVectors(X)
    end

    # println(m)

    m̂ = lepski(Xn = Xn, a = a, b = b, mmin = mmin, mmax = mmax, pi = pi, p = 1, δ = δ)

    return (m̂ - m) / m
end


begin
    a = 0.25
    b = 1
    mmin = 50
    mmax = 200
    pi = 1.1
    δ = 0.05
end

Lep = Any[]

Random.seed!(2022)
for i in 1:3
    println("\n i=$i")
    Lep = vcat(Lep, one_lepski_iter(a = a, b = b, mmin = mmin, mmax = mmax, pi = pi, p = 1, δ = δ))
end
jldsave("./experiments/Lep.jld2"; Lep)
plt = boxplot(Lep, label = "Lepski", legend = :bottomright)
savefig(plt, "./experiments/plots/lepski.pdf")


method1 = Any[]
Random.seed!(2022)
for i in 1:30
    println("\n i=$i")
    method1 = vcat(method1, one_iter(M = M1, epochs = 150))
end
jldsave("./experiments/method1.jld2"; method1)
plt = boxplot(plt, method1, label = "Method1")
savefig(plt, "./experiments/plots/Method1.pdf")


method2 = Any[]
Random.seed!(2022)
for i in 1:20
    println("\n i=$i")
    method2 = vcat(method2, one_iter(M = M1, epochs = 300))
end
jldsave("./experiments/method2.jld2"; method2)
plt = boxplot(plt, method2, label = "Method2")
savefig(plt, "./experiments/plots/Method2.pdf")


# method3 = Any[]
# Random.seed!(2022)
# for i in 1:30
#     println("\n i=$i")
#     method3 = vcat(method3, one_iter(M = M3, epochs = 100))
# end
# jldsave("./experiments/method3.jld2"; method2)
# plt = boxplot(plt, method3, label = "Method2")
# savefig(plt, "./experiments/plots/Method2.pdf")












# M = zeros(0, 4)

# Random.seed!(2022)
# for i in 1:50
#     println("\n i=$i")
#     M = vcat(M, one_iter(250)')
# end

# jldsave("./experiments/M4.jld2"; M)
# savefig(boxplot(M, label = ["M11" "M1" "M2","M3"], legend = :topright), "./experiments/plots/m4.pdf")
# M



# # Plots

# method1 = D -> D[2] .|> persistence |> sum


# @pipe signal |> scatter(_, ratio = 1, label = "Signal", legend = :bottomright)
# @pipe noise |> scatter!(_, ratio = 1, label = "Outliers", legend = :bottomright)
# savefig("./plots/experiments/calibration/scatter.pdf")

# y = [[x["dgm"] |> method1 for x in S_row] |> mean for S_row in S]
# Δy = [[x["dgm"] |> method1 for x in S_row] |> std for S_row in S]
# plot(Q_seq, y, ribbon = Δy, label = "Total Persistence", legend = :bottomright)
# plot!(repeat(2 .* m .+ 1, 100), LinRange(1, 2, 100), label = "2m+1")
# savefig("./plots/experiments/calibration/total-persistence.pdf")