using DrWatson
@quickactivate "wrips-code"

begin
    using Ripserer, PersistenceDiagrams, PersistenceDiagramsBase
    using Random, Distributions, Parameters, Pipe, ProgressMeter, Plots, StatsPlots, JLD2
    using Distances, LinearAlgebra, Statistics, LazySets, Roots, Hyperopt
    using LambertW

    include(srcdir("wRips.jl"))
    import Main.wRips
end

plot_par = wRips.plot_params(alpha = 0.3)
theme(:dao)


function lepski(; Xn, mmin, mmax, pi = 1.1, a, b, p = 1, δ = 0.5)
    n = length(Xn)
    δmin = exp(-8 * (1 + b) * (2 * mmin + 1))

    h = (n, m, δ) ->
        2 * ((2 * m + 1) / (a * n) * lambertw((n * δmin) / (2 * m + 1)))^(1 / b) +
        ((1) / (a * (n - m)) * lambertw((n - m) / ((δ - δmin)^4)))^(1 / b)

    M = [round(Int, mmin * pi^j) for j in 1:1:floor(Int, log(pi, mmax / mmin))] |> unique
    Q = [2 * m + 1 for m in M]
    J = length(M)
    D = @pipe Q .|> wRips.momdist(Xn, _) .|> wRips.fit(Xn, _) .|> wRips.wrips(Xn, w = _, p = 1)

    jhat = J

    prog = Progress(convert(Int, J * (J - 1) / 2))
    generate_showvalues(j) = () -> [(:m, M[j])]

    for j in 1:(J-1)
        flag = false
        # Dj = @pipe Q[j] |> wRips.momdist(Xn, _) |> wRips.fit(Xn, _) |> wRips.wrips(Xn, w = _, p = 1)
        for i in (j+1):J
            # Di = @pipe Q[i] |> wRips.momdist(Xn, _) |> wRips.fit(Xn, _) |> wRips.wrips(Xn, w = _, p = 1)
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
        noise = wRips.randMClust(m, window = win, λ1 = 2, λ2 = 10, r = 0.05)
        # noise = wRips.randUnif(m..., a = -1, b = 1)
        signal = 2.5 .* wRips.randCircle(n, sigma = 0.05)

        X = [signal; noise]
        Xn = wRips._ArrayOfTuples_to_ArrayOfVectors(X)
    end

    # println(m)

    m̂ = lepski(Xn = Xn, a = a, b = b, mmin = mmin, mmax = mmax, pi = pi, p = 1, δ = δ)

    return (m̂ - m) / m
end

begin
    a = 0.5
    b = 1
    mmin = 50
    mmax = 200
    pi = 1.1
    δ = 0.05
end

Lep = Any[]

Random.seed!(2022)
for i in 1:50
    println("\n i=$i")
    Lep = vcat(Lep, one_lepski_iter(a = a, b = b, mmin = mmin, mmax = mmax, pi = pi, p = 1, δ = δ))
end
jldsave("./experiments/Lep.jld2"; Lep)


method1 = Any[]
Random.seed!(2022)
for i in 1:10
    println("\n i=$i")
    method1 = vcat(method1, one_iter(M = M1, epochs = 250))
end
jldsave("./experiments/method1.jld2"; method1)


method2 = Any[]
Random.seed!(2022)
for i in 1:10
    println("\n i=$i")
    method2 = vcat(method2, one_iter(M = M1, epochs = 250))
end
jldsave("./experiments/method2.jld2"; method2)



method3 = Any[]
Random.seed!(2022)
for i in 1:10
    println("\n i=$i")
    method3 = vcat(method3, one_iter(M = M1, epochs = 250))
end
jldsave("./experiments/method3.jld2"; method3)

