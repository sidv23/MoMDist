### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ f2816824-5921-11ec-1367-093aabfe0f10
begin
	import Pkg
	pkgs = ["PlutoUI"]
	# map( x->Pkg.activate(x), pkgs
	using PlutoUI
	using DrWatson
end

# ╔═╡ af7a6e36-c6d3-434c-bb43-165d850bbde2
using Random, Distributions, Plots, Parameters, Distances, LinearAlgebra, Pipe, Statistics, RCall, Ripserer, LazySets, PersistenceDiagrams, Markdown, InteractiveUtils, KernelFunctions, MLDataUtils, Roots, NearestNeighbors

# ╔═╡ 74fca17c-a04b-4d91-b02c-f566056ea710
wRips = include("/storage/home/suv87/work/julia/wrips-code/src/wRips.jl")

# ╔═╡ 3fac284c-3128-40cb-a81b-64d7ca80f7a1
@quickactivate "wrips-code"

# ╔═╡ 1c8d665e-25ae-471b-b10f-13ecf4e9dd56
md"---------"

# ╔═╡ 3469f605-f29e-4ee4-bf6e-ca3988aabeea
begin
p_list = ["1", "1.5", "2", "2.5", "5", "100", "1000"];
type_list = ["Unweighted", "KDist", "MoMKDist"];

md"
n = $(@bind n PlutoUI.Slider(10:10:1000; default=200, show_value=true)) |
m = $(@bind m PlutoUI.Slider(1:1:n; default=convert(Int, 1), show_value=true)) |
Q = $(@bind Q PlutoUI.Slider(2*m:1:n+m; default=2*m, show_value=true))

----

type = $(@bind filt PlutoUI.Select(type_list)) 
| ----------------- |
p = $(@bind pee PlutoUI.Select(p_list))

----  

K = $(@bind K PlutoUI.Slider(range(1, 500, length=500); default=2, show_value=true))

--- ---

t = $(@bind t PlutoUI.Slider(range( 1e-2, 2.5, length=1000); default=0.01, show_value=true))

"
end

# ╔═╡ df057c19-704d-4612-b08d-6252b13494df
begin
	l = 0.5
    Random.seed!(1)
    # noise = wRips.randUnif(m, -l, l)
    win = (-l, l, -l, l)
    noise = wRips.randMClust(m, window = win, λ1 = 2, λ2 = 10, r = 0.05)
    signal = 2 .* wRips.randCircle(n, σ = 0.05)	
    X = [signal; noise]
    Xn = wRips._ArrayOfTuples_to_ArrayOfVectors(X)
end

# ╔═╡ cbf0d00d-f60c-48e1-ac02-2b7f9be2e6fb
begin
	Random.seed!(1)
	plot_par = wRips.plot_params(alpha = 0.3)

	σ = wRips.bandwidth_select(Xn, K)
	M = K / length(Xn)	
    
	# Kernel Stuff
	k = 1.0 * RationalQuadraticKernel() ∘ ScaleTransform(1 / σ)
	DnH = wRips.KDist(Xn, k)
    DnQ = wRips.momKDist(Xn, k, Q = Q)


	#DTM Stuff
	dnm = wRips.dtm(Xn, M)
	dnmq = wRips.momdtm(Xn, M, Q)
	dnq = wRips.momdist(Xn, Q)

	# Weights
	wn = @pipe wRips.fit(Xn, DnH) .|> wRips.convert_radius(_, k)
    wnq = @pipe wRips.fit(Xn, DnQ) .|> wRips.convert_radius(_, k)

	vnm  = wRips.fit(Xn, dnm)
	vnmq = wRips.fit(Xn, dnmq)
	vnq  = wRips.fit(Xn, dnq)
	
end

# ╔═╡ 47973395-1243-443e-9c04-c7ffa832f45c
md"(σ = $σ, M = $M)"

# ╔═╡ 8255ccc6-559f-4d0f-8c02-84a3ce243a3a
p = parse(Float64, pee)

# ╔═╡ 9f55c841-0de7-4b55-b1e6-0aacb3b89a6e
begin
    lim = (-4., 4.)
    clims = extrema([0; wn; wnq])
	
	# Vanilla
    plt1 = wRips.filtration_plot(t; Xn = Xn, w = repeat([0], n + m), p = p, par = plot_par)
    plt1 = plot(plt1, title = "Vanilla", xlim = lim, ylim = lim)

	# KDIST
    plt2 = wRips.filtration_plot(t; Xn = Xn, w = wn, p = p, par = plot_par) #, clim = clims)
    plt2 = plot(plt2, title = "KDist", xlim = lim, ylim = lim)

	# MOM KDIST
    plt3 = wRips.filtration_plot(t; Xn = Xn, w = wnq, p = p, par = plot_par)
    plt3 = plot(plt3, title = "MoM KDist", xlim = lim, ylim = lim)
	
	# DTM
    plt4 = wRips.filtration_plot(t; Xn = Xn, w = vnm, p = p, par = plot_par)
    plt4 = plot(plt4, title = "DTM", xlim = lim, ylim = lim)

	# MOMDTM
    plt5 = wRips.filtration_plot(t; Xn = Xn, w = vnmq, p = p, par = plot_par) #, clim = clims)
    plt5 = plot(plt5, title = "MOM DTM", xlim = lim, ylim = lim)

	# MOMDIST
    plt6 = wRips.filtration_plot(t; Xn = Xn, w = vnq, p = p, par = plot_par)
    plt6 = plot(plt6, title = "MoM Dist", xlim = lim, ylim = lim)


    # layout = @layout [grid(1, 3) a{0.01w}]
    # plot(plt1, plt2, plt3, h2, size = (900, 300), layout = layout)

    layout = @layout (2, 3)
	plot(plt1, plt2, plt3, plt4, plt5, plt6, size = (900, 450), layout = layout)
end

# ╔═╡ dd798235-abe9-4333-bfdc-cbb61339f533
begin
    plotly()
    xseq = range(-3, 3, length = 100)
    filter = (x, y) -> wRips.fit([[x, y]], dnq)
    surf = wRips.surfacePlot(xseq, f=filter)
    gr()
end

# ╔═╡ Cell order:
# ╠═f2816824-5921-11ec-1367-093aabfe0f10
# ╠═3fac284c-3128-40cb-a81b-64d7ca80f7a1
# ╠═af7a6e36-c6d3-434c-bb43-165d850bbde2
# ╠═74fca17c-a04b-4d91-b02c-f566056ea710
# ╠═cbf0d00d-f60c-48e1-ac02-2b7f9be2e6fb
# ╟─1c8d665e-25ae-471b-b10f-13ecf4e9dd56
# ╟─3469f605-f29e-4ee4-bf6e-ca3988aabeea
# ╟─47973395-1243-443e-9c04-c7ffa832f45c
# ╟─9f55c841-0de7-4b55-b1e6-0aacb3b89a6e
# ╠═df057c19-704d-4612-b08d-6252b13494df
# ╠═8255ccc6-559f-4d0f-8c02-84a3ce243a3a
# ╠═dd798235-abe9-4333-bfdc-cbb61339f533
