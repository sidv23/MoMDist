using DrWatson
@quickactivate projectdir()

# Load Packages
begin
    using PersistenceDiagrams, PersistenceDiagramsBase, Ripserer
    using Distributions, Distances, JLD2, LinearAlgebra, Parameters, Pipe, Plots
    using LazySets, LambertW, ProgressMeter, Random, Statistics, StatsPlots

    plot_par = plot_params(alpha=0.3)
    theme(:dao)

    import RobustTDA as rtda
end


function randomRotation(;d=3)
    A = randn(d,d)
    M = (A + A') ./ √2
    _, U = eigen(M)
    return U
end
