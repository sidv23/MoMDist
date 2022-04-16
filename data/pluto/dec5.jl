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

# ╔═╡ 61462f1a-565c-11ec-0b26-658b3b176ef2
begin
	import Pkg
	pkgs = ["PlutoUI", "Plots"]
	map( x->Pkg.activate(x), pkgs)
	using PlutoUI, Plots, Parameters, 
			Ripserer, Distances, LinearAlgebra, RCall, 
			LazySets, Pipe, Parameters, Random,
			Statistics
	using Markdown, InteractiveUtils
	using DrWatson
end

# ╔═╡ 94d2afd2-046a-49f7-950c-8ae828b3cb67
@quickactivate "wrips-code"

# ╔═╡ e60e5ef7-bacd-4f62-b571-b62c7eeeb697
using KernelFunctions, MLDataUtils, Roots, Distributions

# ╔═╡ cf3e963b-d13d-43b9-a572-793e9be1ba1c
# begin
# 	b = l
# 	if filt == "Unweighted"
# 		w = repeat([0], n+m)
# 	elseif filt == "KDist"
# 		w = fit(Xn, DnH)
# 	elseif filt == "MoMKDist"
# 		w = fit(Xn, DnQ)
# 	end
# 	wR = rfx.(t, w, p)
# 	wB = Balls(Xn, convert_radius.(wR, k))
# 	plt = plot_balls(wB, par=plot_params(alpha=0.3))
# 	scatter(plt, X, label=nothing, xlim=(-b,b), ylim=(-b,b), ratio=1, marker_z = w)
# end

# ╔═╡ 60fd8b8b-eabf-45f0-9467-16fded51e0b4
md"""
-------
-------
-------
"""

# ╔═╡ 5da9115d-70a6-4693-8130-6e4f088ace63
md"### Parameters"

# ╔═╡ 650773d5-8936-452a-bcfa-f87f240af7d5
md"### Functions"

# ╔═╡ d28ab5b1-96f9-41cd-a7f9-99efdbf098f0
function Base.:*(a::Real, b::Tuple{Vararg{<:Real}})
	return a .* b
end

# ╔═╡ 33c23c67-83c2-43f4-a2a5-672ab21bc999
begin
p_list = ["1", "1.5", "2", "2.5", "5", "100", "1000"];
type_list = ["Unweighted", "KDist", "MoMKDist"];

md"
n = $(@bind n PlutoUI.Slider(10:10:1000; default=200, show_value=true)) |
m = $(@bind m PlutoUI.Slider(1:1:n; default=convert(Int, n/5), show_value=true)) |
Q = $(@bind Q PlutoUI.Slider(2*m:1:n+m; default=2*m, show_value=true))

----

type = $(@bind filt PlutoUI.Select(type_list)) 
| ----------------- |
p = $(@bind pee PlutoUI.Select(p_list))

----  

σ = $(@bind σ PlutoUI.Slider(range(1e-3, 2, length=1000); default=1/sqrt(n), show_value=true))

--- ---

t = $(@bind t PlutoUI.Slider(range( 1e-2, 2*√2, length=1000); default=0.01, show_value=true))

"
end

# ╔═╡ 7ba9f8c7-d4b7-4d5b-8a71-fa2075574fa9
begin

	@with_kw mutable struct plot_params
		col::String = "orange"
		alpha::Float16 = 0.01
		lwd::Float16 = 0
		lalpha::Float16 = 0.0
		lcol::String = col
		msize::Float16 = 1
	end

	@with_kw mutable struct RKHS
		k::Kernel
		kinv::Function
	end

	@with_kw mutable struct wRips_params
		f::Any
		p::Number = 1
		ρ::Any
	end

	@with_kw mutable struct DistanceLikeFunction
	    k::Kernel
	    σ::Vector{T} where T <: Real
	    X::AbstractVector{<: AbstractVector{<: Union{Tuple{Vararg{<: Real}},Vector{<: Real}}}}
	    Kxx::AbstractVecOrMat
	    type::String
	    Q::Int64
	end

end

# ╔═╡ bbc38b48-2e41-44be-b3e5-52e09cec8777
md"Radius functions"

# ╔═╡ 12e74b16-f099-4b2a-b1f2-65b307712152
begin
	function rfx(t, w, p)
	    return t .> w ? ( p != Inf ? ( t^p .- w.^p ).^( 1 / p ) : t ) : 0
	end
	
	function convert_radius(rad, k)
		
	    # r0 = find_zero(t -> dH(k)(0, t) - rad, 0., atol=eps(1.))
	    # r0 = find_zero(t -> dH(k)(0, t) - rad, 0., rtol=1e-10)
		t0 = (2 - ( (rad)^2 )) / 2
		scale = 1 / (k.transform.s...)

		if t0 <= 0
			return √2
		else
			if typeof(k.kernel.kernel) <: SqExponentialKernel
				return sqrt(-2 * log(t0)) * scale
				
			elseif typeof(k.kernel.kernel) <: ExponentialKernel
				return -log(t0) * scale
				
			elseif typeof(k.kernel.kernel) <: GammaExponentialKernel
				return (-log(t0))^(1/k.kernel.kernel.:γ...) * scale
				
			elseif typeof(k.kernel.kernel) <: RationalKernel
				return ((k.kernel.kernel.:α...) * ( (t0^(-1/k.kernel.kernel.:α...)) - 1)) * scale
				
			elseif typeof(k.kernel.kernel) <: RationalQuadraticKernel
				return sqrt((2 * k.kernel.kernel.:α...) * ((t0^(-1/k.kernel.kernel.:α...)) - 1)) * scale
			
			elseif typeof(k.kernel.kernel) <: GammaRationalKernel
				return ((k.kernel.kernel.:α...) * ( (t0^(-1/k.kernel.kernel.:α...)) - 1)) ^ (1/k.kernel.kernel.:γ...) * scale
				
			else
				throw(DomainError(k, "Kernel Type Unsupported"))
			end
		end
	end
end

# ╔═╡ 55a15785-767f-4c83-94f6-7d1ad77832b3
md"Established Functions"

# ╔═╡ 27151fb5-fd1d-4fc5-8d7a-372e2d4a2091
begin

	#############################################################
	############ Helper Functions
	
	_Matrix_to_ArrayOfTuples = M -> Tuple.(eachcol(M)...)
	_ArrayOfTuples_to_Matrix = A -> hcat(collect.(A)...)'
	_ArrayOfVectors_to_ArrayOfTuples = A -> Tuple.(A)
	_ArrayOfTuples_to_ArrayOfVectors = A -> [[a...] for a in A]

	
	#############################################################
	############ Kernel Distance

	function dH(k::Kernel)
	    # return function(x,y) sqrt(k(x,x) + k(y,y) - 2k(x,y)) end
	    return function(x,y) sqrt(2 - 2 * k(x,y)) end
	end

	function MMD(X::AbstractVector, Y::AbstractVector; k::Kernel)
	    m = length(X)
	    n = length(Y)
	    Kxx = kernelmatrix(k, X) |> sum
	    Kyy = kernelmatrix(k, Y) |> sum
	    Kxy = kernelmatrix(k, X, Y) |> sum
	    return sqrt( (Kxx / (m^2)) + (Kyy / (n^2)) - 2 * (Kxy / (m*n)) )
	end

	function KDist(
	    data::AbstractVector{T}, 
	    k::Kernel
	    ) where {T <: Union{Tuple{Vararg{<:Real}},Vector{<:Real}}}
	    Kxx = kernelmatrix(k, data) |> Statistics.mean
	    return DistanceLikeFunction(
	        k = k,
	        σ = :transform ∈ fieldnames(typeof(k)) ? k.transform.s : [1.0],
	        X = [data],
	        Kxx = [Kxx],
	        type = "vanilla",
	        Q = 1
	    )
	end

	function momKDist(
	    data::AbstractVector{T}, 
	    k::Kernel;
	    Q::Integer=0
	    ) where {T <: Union{Tuple{Vararg{<:Real}},Vector{<:Real}}}
	    
	    if Q < 1
	        println("Invalid value of Q supplied. Defaulting to Q = n_obs / 5 = $Q")
	        Q = ceil(Int16, length(data) / 5)
	    end
	
	    Xq = [ fold[2] for fold in kfolds(shuffleobs(data), Q) ]
	    
	    Kxx = [(kernelmatrix(k, xq) |> Statistics.mean) for xq in Xq]
	    
	    return DistanceLikeFunction(
	        k = k,
	        σ = :transform ∈ fieldnames(typeof(k)) ? k.transform.s : [1.0],
	        X = Xq,
	        Kxx = Kxx,
	        type = "mom",
	        Q = Q
	    )
	end
	
	function eval(
	    x::Union{T,AbstractVector{T}}, 
	    D::DistanceLikeFunction
	    ) where {T <: Union{Tuple{Vararg{<:Real}},Vector{<:Real}}}
	    
	    @unpack k, X, Kxx, Q = D
	
	    if typeof(x) <: T
	        x = [x]
	    end
	
	    fit = []
	    for j ∈ 1:length(x)
	        push!(fit, 
	            [ sqrt(k(0, 0) + Kxx[i] - 2 * ( kernelmatrix(k, X[i], [x[j]]) |> Statistics.mean )) for i in 1:Q] 
	        )
	    end
	    return reduce(vcat, median.(fit))
	end

	
	function fit(
	    x::Union{T,AbstractVector{T}}, 
	    D::DistanceLikeFunction
	    ) where {T <: Union{Tuple{Vararg{<:Real}},Vector{<:Real}}}
	    
	    @unpack k, X, Kxx, Q = D
	
	    if typeof(x) <: T
	        x = [x]
	    end
	
	    fit = []
	    for j ∈ 1:length(x)
	        push!(fit, 
	            [ sqrt(k(0, 0) + Kxx[i] - 2 * ( kernelmatrix(k, X[i], [x[j]]) |> Statistics.mean )) for i in 1:Q] 
	        )
	    end
	    return reduce(vcat, median.(fit))
	end
		
	
	#############################################################
	############ Balls
	
	function Balls(X, R)
		if length(R) == 1
			return [Ball2([x...], float(R)) for x in X]
		else
			return [Ball2([x...], float(r)) for (x, r) in zip(X, R)]
		end
	end
	
	#############################################################
	############ Plot Balls
	
	function plot_balls(B; X = nothing, p = nothing, par::plot_params)
		if isnothing(X) & isnothing(p)
			p = plot(B, 
					c = par.col, fillalpha = par.alpha, linewidth = par.lwd, 
					linealpha =par.lalpha, linecolor=par.col, ratio = 1)
		
		elseif isnothing(X)
			p = plot(B, 
					c = par.col, fillalpha = par.alpha, linewidth = par.lwd, 
					linealpha =par.lalpha, linecolor=par.col, ratio = 1)
		
		elseif isnothing(p)
			p = scatter(X, ratio=1, label=nothing)
			p = plot(p, B, 
					c = par.col, fillalpha = par.alpha, linewidth = par.lwd, 
					linealpha =par.lalpha, linecolor=par.col, ratio = 1)
		
		else
			p = scatter(p, X, ratio=1, label=nothing)
			p = plot(B, 
					c = par.col, fillalpha = par.alpha, linewidth = par.lwd, 
					linealpha =par.lalpha, linecolor=par.col, ratio = 1)
		end

		return p
	end
			
end

# ╔═╡ 8dbc1bce-82f0-4e11-9295-3e2a536730db
begin
	# k = 1 * GammaExponentialKernel() ∘ ScaleTransform(1/σ)
	# k = 1 * RationalQuadraticKernel() ∘ ScaleTransform(1/σ)
	k = 1 * ExponentialKernel() ∘ ScaleTransform(1/σ)
	# k = GaussianKernel() ∘ ScaleTransform(1/σ)
	ρ = dH(k)
	p = parse(Float64, pee);
end

# ╔═╡ 729dfdad-fc82-4132-bd5d-7f2af52af639
md"Shapes"

# ╔═╡ bfb3c82a-0d8d-496f-9a59-80c5275bfd7c
begin
	function randLemniscate(n; σ = 0)
	    
	    signal = R"tdaunif::sample_lemniscate_gerono($n)"
	    noise = R"matrix(rnorm(2 * $n, 0, $σ), ncol=2)"
	    
	    X = tuple.(eachcol(rcopy(signal + noise))...)
	    # X = [eachrow( rcopy( signal + noise ) )...]
	    
	    return X
	end
	
	function randCircle(n; σ = 0)
	
	    signal = R"TDA::circleUnif($n)"
	    noise = R"matrix(rnorm(2 * $n, 0, $σ), ncol=2)"
	
	    X = tuple.(eachcol(rcopy(signal + noise))...)
	    # X = [eachrow( rcopy( signal + noise ) )...]
	
	    return X
	end
	
	
	function randUnif(n; a = 0, b = 1, d = 2)
	    return [ tuple(rand(Uniform(a, b), d)...) for _ ∈ 1:1:n ]
	    # return [ rand(Uniform(a, b), d) for _ ∈ 1:1:n ]
	end
end

# ╔═╡ 12ff32bb-d7bf-4201-aa68-3ee92cf2c123
begin
	l = 2
	ϵ = 0.2
	win=(-l,l,-l,l)
	win = ((1-ϵ) * l, (1+ϵ) * l, (1-ϵ) * l, (1+ϵ) * l)
	signal = randCircle(n, σ = 0.05)
	# noise  = randUnif(m, a=l*(1-ϵ), b=l*(1+ϵ), d=2)
	noise  = randUnif(m, a=-l, b=l, d=2)
	# noise = randMClust(window=win, λ1=2, λ2=10, r=0.05)
	noise = rand(noise, m)
	X = [signal; noise]
	Xn = _ArrayOfTuples_to_ArrayOfVectors(X)
end

# ╔═╡ f8c5eec1-b5d9-4654-9558-68e920eb0fea
begin
		DnH = KDist(Xn, k)
		DnQ = momKDist(Xn, k, Q=Q)
end

# ╔═╡ 20619647-c7ab-47eb-8dad-cdf71271e11b
begin
	b = maximum([2, l])
	if filt == "Unweighted"
		w = repeat([0], n+m)
	elseif filt == "KDist"
		w = fit(Xn, DnH)
	elseif filt == "MoMKDist"
		w = fit(Xn, DnQ)
	end
	w_convert = convert_radius.(w, k)
	wR = rfx.(t, w_convert, p)
	wB = Balls(Xn, wR)
	plt = plot_balls(wB, par=plot_params(alpha=0.3))
	scatter(plt, X, label=nothing, xlim=(-b,b), ylim=(-b,b), ratio=1, marker_z = w_convert)
end

# ╔═╡ 9322f0a6-089b-4ac3-8b08-edbcc0a1c069
md"Matérn Cluster Process"

# ╔═╡ 75cee682-f58c-4dc7-80b2-d1af6b550d2a
function randMClust(; window=(-1,1,-1,1), λ1=5, λ2=5, r=0.1)
    # Simulation window parameters
        
    xMin, xMax, yMin, yMax = window

    lambdaParent = λ1;
    lambdaDaughter = λ2;
    radiusCluster = r;

    # xMin = -.5;
    # xMax = .5;
    # yMin = -.5;
    # yMax = .5;

    # # Parameters for the parent and daughter point processes
    # lambdaParent = 10;# density of parent Poisson point process
    # lambdaDaughter = 10;# mean number of points in each cluster
    # radiusCluster = 0.01;# radius of cluster disk (for daughter points)

    # Extended simulation windows parameters
    rExt = radiusCluster; # extension parameter -- use cluster radius
    xMinExt = xMin - rExt;
    xMaxExt = xMax + rExt;
    yMinExt = yMin - rExt;
    yMaxExt = yMax + rExt;
    # rectangle dimensions
    xDeltaExt = xMaxExt - xMinExt;
    yDeltaExt = yMaxExt - yMinExt;
    areaTotalExt = xDeltaExt * yDeltaExt; # area of extended rectangle

    # Simulate Poisson point process
    numbPointsParent = rand(Poisson(areaTotalExt * lambdaParent)); # Poisson number of points

    # x and y coordinates of Poisson points for the parent
    xxParent = xMinExt .+ xDeltaExt * rand(numbPointsParent);
    yyParent = yMinExt .+ yDeltaExt * rand(numbPointsParent);

    # Simulate Poisson point process for the daughters (ie final poiint process)
    numbPointsDaughter = rand(Poisson(lambdaDaughter), numbPointsParent);
    numbPoints = sum(numbPointsDaughter); # total number of points

# Generate the (relative) locations in polar coordinates by
    # simulating independent variables.
    theta = 2 * pi * rand(numbPoints); # angular coordinates
    rho = radiusCluster * sqrt.(rand(numbPoints)); # radial coordinates

    # Convert polar to Cartesian coordinates
    xx0 = rho .* cos.(theta);
    yy0 = rho .* sin.(theta);

    # replicate parent points (ie centres of disks/clusters)
    xx = vcat(fill.(xxParent, numbPointsDaughter)...);
    yy = vcat(fill.(yyParent, numbPointsDaughter)...);

    # Shift centre of disk to (xx0,yy0)
    xx = xx .+ xx0;
    yy = yy .+ yy0;

    # thin points if outside the simulation window
    booleInside = ((xx .>= xMin) .& (xx .<= xMax) .& (yy .>= yMin) .& (yy .<= yMax));
    # retain points inside simulation window
    xx = xx[booleInside];
    yy = yy[booleInside];

    return tuple.(eachcol([xx yy])...)
end

# ╔═╡ Cell order:
# ╠═61462f1a-565c-11ec-0b26-658b3b176ef2
# ╠═e60e5ef7-bacd-4f62-b571-b62c7eeeb697
# ╠═94d2afd2-046a-49f7-950c-8ae828b3cb67
# ╠═f8c5eec1-b5d9-4654-9558-68e920eb0fea
# ╟─8dbc1bce-82f0-4e11-9295-3e2a536730db
# ╟─33c23c67-83c2-43f4-a2a5-672ab21bc999
# ╟─20619647-c7ab-47eb-8dad-cdf71271e11b
# ╠═12ff32bb-d7bf-4201-aa68-3ee92cf2c123
# ╠═cf3e963b-d13d-43b9-a572-793e9be1ba1c
# ╟─60fd8b8b-eabf-45f0-9467-16fded51e0b4
# ╠═5da9115d-70a6-4693-8130-6e4f088ace63
# ╠═7ba9f8c7-d4b7-4d5b-8a71-fa2075574fa9
# ╟─650773d5-8936-452a-bcfa-f87f240af7d5
# ╠═d28ab5b1-96f9-41cd-a7f9-99efdbf098f0
# ╟─bbc38b48-2e41-44be-b3e5-52e09cec8777
# ╠═12e74b16-f099-4b2a-b1f2-65b307712152
# ╟─55a15785-767f-4c83-94f6-7d1ad77832b3
# ╠═27151fb5-fd1d-4fc5-8d7a-372e2d4a2091
# ╟─729dfdad-fc82-4132-bd5d-7f2af52af639
# ╠═bfb3c82a-0d8d-496f-9a59-80c5275bfd7c
# ╠═9322f0a6-089b-4ac3-8b08-edbcc0a1c069
# ╠═75cee682-f58c-4dc7-80b2-d1af6b550d2a
