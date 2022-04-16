
f(x) = [M1(Xn, Q = q, p = 1, dim = 2) for q in 2 .* x .+ 1] |> mean

hohb = @hyperopt for resources = 50, sampler = Hyperband(R = 50, η = 3, inner = RandomSampler()),
    algorithm = [SimulatedAnnealing()],
    Q = 50.:150.,
    c = 50.:150.

    if state !== nothing
        algorithm, x0 = state
    else
        x0 = [Q, c]
    end
    println(resources, " algorithm: ", typeof(algorithm).name.name)
    res = Optim.optimize(x -> f(x[1]), x0, algorithm, Optim.Options(time_limit = resources + 1, show_trace = false))
    Optim.minimum(res), (algorithm, Optim.minimizer(res))
end