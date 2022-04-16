### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 551ee4e4-09a1-11ec-3ee1-290dc932aeb5
using DrWatson, Plots, Parameters, ProgressBars, Pipe, PersistenceDiagramsBase, Ripserer, Distances, KernelFunctions, MLDataUtils, LinearAlgebra, RCall, LazySets, Random, Distributions, MappedArrays,  LaTeXStrings, PlutoUI, Statistics, Roots

# ╔═╡ d67c4d46-554b-4fa7-8e96-51f754cdfdb6
md"""
###  Efficient and Outlier Robust Topological Inference
####  30, Aug, 2021
"""

# ╔═╡ e3aa21d4-4eff-4185-a352-d0db84a36bd2
md"""
-------
"""

# ╔═╡ 164f6f93-0fc2-4d32-ac46-16fe6eaf103b
md"""
## 1. Preliminary Functions
"""

# ╔═╡ d0240346-5032-4dea-972d-5136df5e28fd
begin
		function ingredients(path::String)
		# this is from the Julia source code (evalfile in base/loading.jl)
		# but with the modification that it returns the module instead of the last object
		name = Symbol(basename(path))
		m = Module(name)
		Core.eval(m,
			Expr(:toplevel, 
				 :(eval(x) = $(Expr(:core, :eval))($name, x)),
				 :(include(x) = $(Expr(:top, :include))($name, x)),
				 :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
				 :(include($path))))
		m
	end
end

# ╔═╡ a4812be5-0c65-4bb8-8ec8-3d242b5700fb
tdafun = ingredients("../src/tda-functions.jl")

# ╔═╡ 5da3a5af-ba59-4c33-b45a-bb027bed8bd9
kernelfun = ingredients("../src/kernel-functions.jl")

# ╔═╡ c132a0bc-cc80-46d9-bd66-ece730869eaf
mfun = ingredients("../src/mom-functions.jl")

# ╔═╡ 3e658e75-814f-4274-9a2e-01d13e592cc5
helperfun = ingredients("../src/helper-functions.jl")

# ╔═╡ 37d5004d-0a49-4776-8acb-cd9381fea6ee
ballfun = ingredients("../src/ball-functions.jl")

# ╔═╡ 74f073dd-ba00-4ab0-89b0-c7e0e77fbe27
mc = ingredients("../src/matern-cluster.jl")

# ╔═╡ cce9f845-b1df-451c-89a9-624a59ccf2db
@with_kw mutable struct plot_params
    col::String = "orange"
    alpha::Float16 = 0.01
    lwd::Float16 = 0
    lalpha::Float16 = 0.0
    lcol::String = col
	msize::Float16 = 1
end

# ╔═╡ dd2ecb76-122b-4f87-8df2-9e753bbe993b
P = plot_params(alpha = 1, lwd = 1, lalpha = 1)

# ╔═╡ da58526e-93b8-4ffc-b91a-a7e643a8fe68
gr()

# ╔═╡ 1c876bb8-4364-470a-9c82-09901e624bc9
md"""
-------
"""

# ╔═╡ 9e84aa65-d2ae-4829-8f0a-0277dc7b29b6
md"""
## 2. Data
"""

# ╔═╡ 04674cae-5f27-4a39-8ac6-3233da5358d9
md"""
#### 2A. Additive Noise
"""

# ╔═╡ cca9117e-390b-4917-8470-99380fcdd727
md"""
Noise σ = $(@bind noise_σ Slider(0:0.05:0.5; default=0.1, show_value=true))
"""

# ╔═╡ 81f47846-0e22-44b4-a7bc-9005bbe33ab4
begin
	x = tdafun.randCircle(500, σ = noise_σ);
	y = tdafun.randLemniscate(500, σ = noise_σ);
	plt11 = helperfun.vscatter(x, ratio = 1, title = "Circle", label="")
	plt21 = helperfun.vscatter(y, ratio = 1, title = "Lemniscate", label="")
	plot(plt11, plt21)
end

# ╔═╡ ef02bcff-7872-4b05-8267-0dc303150ee1
md"""
#### 2B. Mixture Noise
"""

# ╔═╡ 2beeb6f9-f49f-451b-a608-503865374190
md"""
a = $(@bind noise_a Slider(-5:0.5:0; default=-2, show_value=true))
b = $(@bind noise_b Slider(0.:0.5:5; default=2, show_value=true))
"""

# ╔═╡ 69e1eaf3-992a-4e62-941f-d5bde6ca2abc
begin
	plt12 = helperfun.vscatter([x; tdafun.randUnif(300, a=noise_a, b=noise_b)], ratio = 1, title = "Circle", label="")
	plt22 = helperfun.vscatter([y; tdafun.randUnif(400, a=noise_a, b=noise_b)], ratio = 1, title = "Lemniscate", label="")
	plot(plt12, plt22)
end

# ╔═╡ fc3e0cd3-3284-47d0-b6f9-4329fe345eee
md"""
-------
"""

# ╔═╡ e7cf7877-7a4c-4158-a9b6-8c0685d5b766
md"""
## 3: Unweighted Rips filtration $|| \cdot ||$ vs. $d_{\mathcal{H}}(\cdot, \cdot)$
"""

# ╔═╡ 8b96195b-4e1c-4fea-937a-5de98318c27c
md"""
* This is the standard Rips filtration for the circle and the lemniscate
"""

# ╔═╡ 9bc58a68-a8e3-4a0b-ac4c-173d745954f2
begin
	Random.seed!(2021); 
	x_noise = [tdafun.randCircle(200, σ=0.1); tdafun.randUnif(50, a=-2, b=2)];
	y_noise = [tdafun.randLemniscate(200, σ=0.1); tdafun.randUnif(50, a=-1.5, b=1.5)];
end;

# ╔═╡ e429b51f-705d-4a96-ac60-ffaca191f935
md"""
r = $(@bind r1 Slider(0.01:0.01:0.55; default=0.1, show_value=true))
"""

# ╔═╡ 2fff8666-5626-41b0-b155-3dd5b7f58d02
begin    
	plt13 = plot(tdafun.Balls(x_noise, r = r1), c = P.col, fillalpha = P.alpha, linewidth = P.lwd, linealpha = P.lalpha, linecolor = P.col, xlims = (-2, 2), ylims = (-2, 2), ratio = 1)
	plt13 = scatter!(plt13, Tuple.(eachcol(x_noise)...), markersize = 2, label = "")
	
	plt23 = plot(tdafun.Balls(y_noise, r = r1), c = P.col, fillalpha = P.alpha, linewidth = P.lwd, linealpha = P.lalpha, linecolor = P.col, xlims = (-2, 2), ylims = (-2, 2), ratio = 1)
	plt23 = scatter!(plt23, Tuple.(eachcol(y_noise)...), markersize = 2, label = "")
	
	plot(plt13, plt23)
end

# ╔═╡ 1753e4ce-f8d0-4e0c-87fa-75cd7950895b
md"""
-------
"""

# ╔═╡ ed077d47-471e-4fe3-b505-ff8bc8d0bbcc
md"""
##### Define the kernel $\mathcal{K}$ and the metric $d_{\mathcal H}$
"""

# ╔═╡ 13ff936a-257d-4ed3-93b8-c0c84bf274ed
md"""
* $\mathcal{K}(x, y) = \exp\left( \frac{||x - y||^2}{\sigma^2} \right)$
"""

# ╔═╡ 79d63b7e-4d28-49b3-b662-ce080a5b4b01
k = GaussianKernel() ∘ ScaleTransform(0.8)

# ╔═╡ 7f628a8c-0b7d-4a32-99cd-c37b1facc242
md"""
* $d_{\mathcal H}(x, y) = \mathcal{K}(x,x,) + \mathcal{K}(y,y) - 2\mathcal{K}(x,y)$
"""

# ╔═╡ d582e84b-8c17-4a4d-923a-f39559bef6a9
d = kernelfun.dH(k)

# ╔═╡ 915f12e5-6b05-486c-ac02-103632941ce2
md"""
Comparison of filtrations induced by the two metrics
"""

# ╔═╡ 1dd935b5-b28e-4f70-ba42-b588a739438a
md"""
r = $(@bind r2 Slider(0.01:0.01:0.55; default=0.1, show_value=true))
"""

# ╔═╡ 17295b3b-7532-4403-9454-6b305d9dfd38
begin    
	plt14 = plot(tdafun.Balls(x_noise, r = r2), c = P.col, fillalpha = P.alpha, linewidth = P.lwd, linealpha = P.lalpha, linecolor = P.col, xlims = (-2, 2), ylims = (-2, 2), ratio = 1)
	plt14 = scatter!(plt14, Tuple.(eachcol(x_noise)...), markersize = 2, label = "")
		
	W0 = zeros(length(x_noise))
	
	plt24 = plot(tdafun.weightedBalls(x_noise, t = r2, W=W0, p = 2, metric=d), c = P.col, fillalpha = P.alpha, linewidth = P.lwd, linealpha = P.lalpha, linecolor = P.col, xlims = (-2, 2), ylims = (-2, 2), ratio = 1)
	plt24 = scatter!(plt24, Tuple.(eachcol(x_noise)...), markersize = 2, label = "")
	
	plot(plt14, plt24)
end

# ╔═╡ 3b059004-beb4-4da5-b812-8a796cabb946
md"""
##### This is their associated persistence diagrams.
"""

# ╔═╡ 8841f0fe-52ea-44cf-a1ff-71624b133cdc
begin
	g = x -> zeros(size(x, 1))
	D11 = @pipe x_noise |> pairwise(Euclidean(), _) |> ripserer
	plt15 = plot(D11, title=L"\textrm{Dgm \ for \ }: ||\cdot||")
	D21 = @pipe x_noise |> pairwise(d, _) |> ripserer
	plt25 = plot(D21, title=L"\textrm{Dgm \ for \ }: d_{\mathcal{H}}")
	plot(plt15, plt25)
end

# ╔═╡ 5d9c07ac-4d20-43ae-9f71-2bda2750aea8
md"""
##### And this is their persistence diagrams on a log-scale.
"""

# ╔═╡ 79bfe0ab-2682-4fa8-a508-c3fc9ee3a2fe
begin
	D12 = @pipe D11 |> tdafun._offset_births(_,exp(-20)) |> log
	plt16 = plot(D12, title= L"\textrm{log-Dgm \ for \ }: || \cdot ||", xlab="log-birth", ylab="log-death", xlim=(-21,3), ylim=(-20,3))
	
	D22 = @pipe D21 |> tdafun._offset_births(_, exp(-20)) |> log
	plt26 = plot(D22, title= L"\textrm{log-Dgm \ for \ }: d_{\mathcal{H}}", xlab="log-birth", ylab="log-death", xlim=(-21,3), ylim=(-20,3))
	
	plot(plt16, plt26)
end

# ╔═╡ 07ffc1c1-8ec1-4344-b8c5-f0da8ccccb54
md"""
-------
"""

# ╔═╡ 45ec2b46-95dd-4a84-b088-3232b35b0764
md"""
## 4: Weighted Rips filtrations
"""

# ╔═╡ f01c10a3-f97d-4d99-b2f8-d86efe4f1402
md"""
The following is an example for weighted Rips filtrations computed using an **oracle** function, i.e., when the points $\mathbb{X}_n$ are sampled from a circle $S^1$ with noise, the weight-function $f:\mathbb{R}^2 \rightarrow \mathbb{R}^2$ is given by

* $f(x) = \inf_{y \in S^1} \lvert| x - y \rvert|$
"""

# ╔═╡ dbfa3ab7-34e8-439f-ba58-5973a2327413
begin
	f = x -> abs.(norm.(x) .- 1)
	x_alt = [tdafun.randCircle(200, σ = 0.1); tdafun.randUnif(100, a = -3, b = 3)]
end

# ╔═╡ 55a5bf16-09c6-434e-8460-5b59b5ce5f2f
θ = tdafun.wRips_params(p = Inf, f = f, ρ = Distances.Euclidean(1e-6))

# ╔═╡ a00a1d28-2020-4637-8080-e79b0d3a13bc
begin
	D_when_p1 = tdafun.WeightedRips(x_alt, f = θ.f, ρ = θ.ρ, p = 1.0)
	# D_when_p2 = WeightedRips(x_alt, f = θ.f, ρ = θ.ρ, p = 2)
	D_when_p3 = tdafun.WeightedRips(x_alt, f = θ.f, ρ = θ.ρ, p = Inf)
end;

# ╔═╡ ccc6ad73-9a5c-4caa-a4d5-096ba39f655e
md"""
t = $(@bind t0 Slider(0.005:0.005:1.4;default=0.1, show_value=true))
"""

# ╔═╡ 3d5f295b-c00b-452c-8ff8-59da96b4d448
begin	
	filt_when_p1 = plot(tdafun.weightedBalls(x_alt, t = t0, f = f, p = 1), c = P.col, fillalpha = P.alpha, linewidth = P.lwd, linealpha = P.lalpha, linecolor = P.col, xlims = (-2, 2), ylims = (-2, 2), ratio = 1)
    filt_when_p1 = scatter!(filt_when_p1, Tuple.(x_alt), markersize = 2, label = "", ratio = 1, title=L"p=1");
	
	filt_when_p2 = plot(tdafun.weightedBalls(x_alt, t = t0, f = f, p = 2), c = P.col, fillalpha = P.alpha, linewidth = P.lwd, linealpha = P.lalpha, linecolor = P.col, xlims = (-2, 2), ylims = (-2, 2), ratio = 1)
    filt_when_p2 = scatter!(filt_when_p2, Tuple.(x_alt), markersize = 2, label = "", ratio = 1, title=L"p=2");
	
	filt_when_p3 = plot(tdafun.weightedBalls(x_alt, t = t0, f = f, p = Inf), c = P.col, fillalpha = P.alpha, linewidth = P.lwd, linealpha = P.lalpha, linecolor = P.col, xlims = (-2, 2), ylims = (-2, 2), ratio = 1);
    filt_when_p3 = scatter!(filt_when_p3, Tuple.(x_alt), markersize = 2, label = "", ratio = 1, title=L"p=\infty");
end;

# ╔═╡ 7fb728c2-abe0-4125-9725-b623a6b0f5c5
	plot(
		plot(filt_when_p1, filt_when_p2, filt_when_p3, layout=(1,3)),
		plot(plot(D_when_p1, title=L"p=1"), plot(D_when_p3, title=L"p=\infty"), layout=(1,2)),
		layout=(2,1)
		)

# ╔═╡ 0ebd6dd7-0577-4583-bb3a-c48aa7cd2d7e
md"""
-------
"""

# ╔═╡ 90cc2493-ddce-481b-bfa0-31ef46dcb3af
md"""
## 5: KDist $\mathsf{D}_{n, \mathcal{H}}$ and MoM KDist  $\mathsf{D}_{n, Q}$
"""

# ╔═╡ f9c521dd-c244-4be5-9470-5bbef264e782
md"""
Here, in the absence of an oracle weight function $f$, we examine a data-drived process for estimating the weight-function. We can first examine the two distance functions $\mathsf{D}_{n, \mathcal{H}}$ and $\mathsf{D}_{n, Q}$ themselves. Their sublevel persistence diagrams are not plotted as it takes up too much memory. But their surface plots provide a rough idea of their properties. 
"""

# ╔═╡ 357b00fd-d236-498c-b26a-56c66d1ec377
begin
	Random.seed!(42)
	l=0.75
	n_in = 1000
	signal = 1.5 .* tdafun.randCircle(400, σ=0.03);
	noise = mc.randMClust(;window=(-l,l,-l,l), λ1=2, λ2=30, r=0.05)
	n_out = length(noise)
	Y = [signal; noise]
	# Y = [
		# 2 .* tdafun.randCircle(400, σ=0.07);
		# tdafun.randUnif(200, a=-l, b=l);
		# map(x -> x .+ (-l,-l), tdafun.randUnif(25, a=-0.5-rand(), b=0.5+rand()));
		# map(x -> x .+ (-l,l), tdafun.randUnif(25, a=-0.5-rand(), b=0.5+rand()));
		# map(x -> x .+ (l,-l), tdafun.randUnif(25, a=-0.5-rand(), b=0.5+rand()));
		# map(x -> x .+ (l,l), tdafun.randUnif(25, a=-0.5-rand(), b=0.5+rand()));
		# map(x -> x .+ (0,0), tdafun.randUnif(50, a=-0.5-rand(), b=0.5+rand()));
	# ]
	Xn = helperfun._ArrayOfTuples_to_ArrayOfVectors(Y)
	scatter(Y, ratio=1)
end

# ╔═╡ 46ddaff1-c390-47ce-a030-cac2cda73cec
noise

# ╔═╡ 8f7a3f7e-aa54-453e-a08f-7d92c5b509b0
n_out

# ╔═╡ d7833440-2fac-4948-8c9f-3fce3185aa86
md"""
Kernel Bandwidth σ = $(@bind s Slider(0.05:0.01:1.1; default=0.2, show_value=true))  ----------
Blocks Q = $(@bind Q Slider(1:1:(n_in + n_out) / 2; default= 2 * n_out + 2, show_value=true))
"""

# ╔═╡ 42c7bb91-75a3-44eb-9f28-26f8277a9f5c
kern = (1/s^2) * KernelFunctions.RationalQuadraticKernel() ∘ ScaleTransform(1/s)

# ╔═╡ 219495fa-9c32-4a57-a549-d935ad2458aa
DnH = kernelfun.KDist(Xn, kern)

# ╔═╡ f9e87cd3-47d0-46b0-9673-d9f22f613e88
DnQ = kernelfun.momKDist(Xn, kern, Q=Q)

# ╔═╡ 7d3340b3-bd5e-465a-9f4e-9bb3e983b539
plt = let
	plotly()
	surf1 = plot(-5:0.1:5, -5:0.2:5, (u,v)->kernelfun.eval([u,v], DnH), st=:surface, title="DnH")
	surf2 = plot(-5:0.1:5, -5:0.2:5, (u,v)->kernelfun.eval([u,v], DnQ), st=:surface, title="DnQ")
	# surf3 = plot(-5:0.2:5, -5:0.2:5, (u,v)->mfun.eval3([u,v], DnQ3), st=:surface, title="DnQ3")
	plot(surf1, surf2)
end

# ╔═╡ dafd8044-e44c-452e-9da9-9bbfcabed47c
begin
	plt != nothing ? gr() : gr()
end

# ╔═╡ a93e7c84-5dd1-4a32-9f76-31f958ac4b48
md"""
-------
"""

# ╔═╡ 423f7e1c-9e43-42bf-bd19-624f100e893b
md"""
## 4: Weighted Rips filtrations
"""

# ╔═╡ 49a1c8e0-706f-4910-a6f6-f3c7daf40a4e
md"""
The following is an example for weighted Rips filtrations computed using an **oracle** function, i.e., when the points $\mathbb{X}_n$ are sampled from a circle $S^1$ with noise, the weight-function $f:\mathbb{R}^2 \rightarrow \mathbb{R}^2$ is given by

* $f(x) = \inf_{y \in S^1} \lvert| x - y \rvert|$
"""

# ╔═╡ 1a6a324f-6e5e-406b-a2c0-3c4a27919de7
begin
	w1 = kernelfun.eval(Xn, DnH)
	w2 = kernelfun.eval(Xn, DnQ)
	w2 = ((w2 .- mean(w2)) ./ std(w2)) .* std(w1) .+ mean(w1)
	wplt1 = scatter(Y, marker_z=w1, ratio=1, title="DnH", label="")
	wplt2 = scatter(Y, marker_z=w2, ratio=1, title="DnQ", label="")
	plot(wplt1, wplt2)
end

# ╔═╡ 0046db0f-63f3-4e2c-b7c4-a427e0d94bdd
begin

# md"""
#t1 = $(@bind T1 Slider( helperfun.partition(w1, n=100, kludge=0), default=0, show_value=true))   ||------------------------------||     t2 = $(@bind T2 Slider( # helperfun.partition(w2, n=100, kludge=0.2), default=0, show_value=true))
# """

md"""
t1 = $(@bind T1 Slider( helperfun.partition(w1, n=100, kludge=0.1), default=0.01, show_value=true))
"""
	
end

# ╔═╡ 102d4372-b723-4802-9ced-bec0e12e4c2c
T2=T1

# ╔═╡ 72085db6-d480-4204-908d-e529f666dd5e
begin
	wscatter1 = plot(tdafun.weightedBalls(Xn, t = T1, f=nothing, W=w1, p = 1, metric=Euclidean()), c = P.col, fillalpha = P.alpha, linewidth = P.lwd, linealpha = P.lalpha, linecolor = P.col, xlims = (-l, l), ylims = (-l, l), ratio = 1)
    
    scatter!(Y, markersize = 2, label = "", ratio = 1)
	
		wscatter2 = plot(tdafun.weightedBalls(Xn, t = T2, f=nothing, W=w2, p = 1, metric=Euclidean()), c = P.col, fillalpha = P.alpha, linewidth = P.lwd, linealpha = P.lalpha, linecolor = P.col, xlims = (-l, l), ylims = (-l, l), ratio = 1)
    
    scatter!(Y, markersize = 2, label = "", ratio = 1)
	
	plot(wscatter1, wscatter2)

# 		wscatter3 = plot(tdafun.weightedBalls(Xn, t = T1, W=w1, p = 1, metric=kernelfun.dH(k)), c = P.col, fillalpha = P.alpha, linewidth = P.lwd, linealpha = P.lalpha, linecolor = P.col, xlims = (-4, 4), ylims = (-4, 4), ratio = 1)
    
#     scatter!(Y, markersize = 2, label = "", ratio = 1)
	
# 		wscatter4 = plot(tdafun.weightedBalls(Xn, t = T2, W=w2, p = 1, metric=kernelfun.dH(k)), c = P.col, fillalpha = P.alpha, linewidth = P.lwd, linealpha = P.lalpha, linecolor = P.col, xlims = (-4, 4), ylims = (-4, 4), ratio = 1)
    
#     scatter!(Y, markersize = 2, label = "", ratio = 1)
	
# 	plot(wscatter1, wscatter2, wscatter3, wscatter4)
	
end

# ╔═╡ b9d69d61-ccc4-4d5d-b34f-e97b3e729bc7
pairwise(Euclidean(), Xn)

# ╔═╡ 4686bb49-ed62-4cb6-b627-aec97f8c094e
begin
	D1 = tdafun.WeightedRips(Xn, W=w1, p=1, type="points")
	D2 = tdafun.WeightedRips(Xn, W=w2, p=1, type="points")
end;

# ╔═╡ b619b0ff-8266-4ac9-8e77-c02b8715fce5
D2 |> plot

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
DrWatson = "634d3b9d-ee7a-5ddf-bec9-22491ea816e1"
KernelFunctions = "ec8451be-7e33-11e9-00cf-bbf324bd1392"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LazySets = "b4f0291d-fe17-52bc-9479-3d1a343d9043"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MLDataUtils = "cc2ba9b6-d476-5e6d-8eaf-a92d5412d41d"
MappedArrays = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"
PersistenceDiagramsBase = "b1ad91c1-539c-4ace-90bd-ea06abc420fa"
Pipe = "b98c9c47-44ae-5843-9183-064241ee97a0"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressBars = "49802e3a-d2f1-5c88-81d8-b72133a6f568"
RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Ripserer = "aa79e827-bd0b-42a8-9f10-2b302677a641"
Roots = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
Distances = "~0.10.3"
Distributions = "~0.25.11"
DrWatson = "~2.2.1"
KernelFunctions = "~0.10.16"
LaTeXStrings = "~1.2.1"
LazySets = "~1.49.1"
MLDataUtils = "~0.5.4"
MappedArrays = "~0.4.1"
Parameters = "~0.12.2"
PersistenceDiagramsBase = "~0.1.1"
Pipe = "~1.3.0"
Plots = "~1.20.1"
PlutoUI = "~0.7.9"
ProgressBars = "~1.4.0"
RCall = "~0.13.12"
Ripserer = "~0.16.8"
Roots = "~1.2.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "f87e559f87a45bece9c9ed97458d3afe98b1ebb9"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.1.0"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Statistics", "UUIDs"]
git-tree-sha1 = "aa3aba5ed8f882ed01b71e09ca2ba0f77f44a99e"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.1.3"

[[BinDeps]]
deps = ["Libdl", "Pkg", "SHA", "URIParser", "Unicode"]
git-tree-sha1 = "1289b57e8cf019aede076edab0587eb9644175bd"
uuid = "9e28174c-4ba2-5203-b857-d8d62c4213ee"
version = "1.0.2"

[[BinaryProvider]]
deps = ["Libdl", "Logging", "SHA"]
git-tree-sha1 = "ecdec412a9abc8db54c0efc5548c64dfce072058"
uuid = "b99e7846-7c00-51b0-8f62-c81ae34c0232"
version = "0.5.10"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c3598e525718abcc440f69cc6d5f60dda0a1b61e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.6+5"

[[CMake]]
deps = ["BinDeps"]
git-tree-sha1 = "50a8b41d2c562fccd9ab841085fc7d1e2706da82"
uuid = "631607c0-34d2-5d66-819e-eb0f9aa2061a"
version = "1.2.0"

[[CRlibm]]
deps = ["Libdl"]
git-tree-sha1 = "9d1c22cff9c04207f336b8e64840d0bd40d86e0e"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "0.8.0"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "e2f47f6d8337369411569fd45ae5753ca10394c6"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.0+6"

[[Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[CategoricalArrays]]
deps = ["DataAPI", "Future", "JSON", "Missings", "Printf", "RecipesBase", "Statistics", "StructTypes", "Unicode"]
git-tree-sha1 = "1562002780515d2573a4fb0c3715e4e57481075e"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.0"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "bdc0937269321858ab2a4f288486cb258b9a0af7"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.3.0"

[[CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "2e62a725210ce3c3c2e1a3080190e7ca491f18d7"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.7.2"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "9995eb3977fbf67b86d0a0a0508e83017ded03f2"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.14.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[CommonSolve]]
git-tree-sha1 = "68a0743f578349ada8bc911a5cbd5a2ef6ed6d1f"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.0"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "727e463cfebd0c7b999bbf3e9e7e16f254b94193"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.34.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[Conda]]
deps = ["JSON", "VersionParsing"]
git-tree-sha1 = "299304989a5e6473d985212c28928899c74e9421"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.5.2"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d785f42445b63fc86caa08bb9a9351008be9b765"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.2.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "3ed8fa7178a10d1cd0f1ca524f249ba6937490c0"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.3.0"

[[Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "abe4ad222b26af3337262b8afb28fab8d215e9f8"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.3"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3889f646423ce91dd1055a76317e9a1d3a23fff1"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.11"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[DrWatson]]
deps = ["Dates", "FileIO", "LibGit2", "MacroTools", "Pkg", "Random", "Requires", "UnPack"]
git-tree-sha1 = "65fa2e9c6e187198fad815e0b14db4527143f8ea"
uuid = "634d3b9d-ee7a-5ddf-bec9-22491ea816e1"
version = "2.2.1"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "92d8f9f208637e8d2d28c664051a00569c01493d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.1.5+1"

[[ErrorfreeArithmetic]]
git-tree-sha1 = "d6863c556f1142a061532e79f611aa46be201686"
uuid = "90fa49ef-747e-5e6f-a989-263ba693cf1a"
version = "0.5.2"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[ExprTools]]
git-tree-sha1 = "b7e3d17636b348f005f11040025ae8c6f645fe92"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.6"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "LibVPX_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "3cc57ad0a213808473eafef4845a74766242e05f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.3.1+4"

[[FastRounding]]
deps = ["ErrorfreeArithmetic", "Test"]
git-tree-sha1 = "224175e213fd4fe112db3eea05d66b308dc2bf6b"
uuid = "fa42c844-2597-5d31-933b-ebd51ab2693f"
version = "0.2.0"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "937c29268e405b6808d958a9ac41bfe1a31b08e7"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.11.0"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "7c365bdef6380b29cfc5caaf99688cd7489f9b87"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.2"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "35895cf184ceaab11fd778b4590144034a167a2f"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.1+14"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "NaNMath", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "b5e930ac60b613ef3406da6d4f42c35d8dc51419"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.19"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "cbd58c9deb1d304f5a245a0b7eb841a2560cfec6"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.1+5"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Functors]]
deps = ["MacroTools"]
git-tree-sha1 = "4cd9e70bf8fce05114598b663ad79dfe9ae432b3"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.2.3"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "dba1e8614e98949abfa60480b13653813d8f0157"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+0"

[[GLPK]]
deps = ["BinaryProvider", "GLPK_jll", "Libdl", "MathOptInterface", "SparseArrays"]
git-tree-sha1 = "86573ecb852e303b209212046af44871f1c0e49c"
uuid = "60bf3e95-4087-53dc-ae20-288a0d20c6a6"
version = "0.13.0"

[[GLPKMathProgInterface]]
deps = ["GLPK", "LinearAlgebra", "MathProgBase", "SparseArrays"]
git-tree-sha1 = "dcca815a687d8f398c8fc701c3796a36ef6b73a5"
uuid = "3c7084bd-78ad-589a-b5bb-dbd673274bea"
version = "0.5.0"

[[GLPK_jll]]
deps = ["GMP_jll", "Libdl", "Pkg"]
git-tree-sha1 = "ccc855de74292e478d4278e3a6fdd8212f75e81e"
uuid = "e8aa6df9-e6ca-548a-97ff-1f85fc5b8b98"
version = "4.64.0+0"

[[GMP_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "781609d7-10c4-51f6-84f2-b8444358ff6d"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "182da592436e287758ded5be6e32c406de3a2e47"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.58.1"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "d59e8320c2747553788e4fc42231489cc602fa50"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.58.1+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "44e3b40da000eab4ccb1aecdc4801c040026aeb5"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.13"

[[Hungarian]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "371a7df7a6cce5909d6c576f234a2da2e3fa0c98"
uuid = "e91730f6-4275-51fb-a7a0-7064cfbd3b39"
version = "0.6.0"

[[Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[IntervalArithmetic]]
deps = ["CRlibm", "FastRounding", "LinearAlgebra", "Markdown", "Random", "RecipesBase", "RoundingEmulator", "SetRounding", "StaticArrays"]
git-tree-sha1 = "45d133a69e4944b3ee1886575fb62791fb2ae43b"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.19.0"

[[InvertedIndices]]
deps = ["Test"]
git-tree-sha1 = "15732c475062348b0165684ffe28e85ea8396afc"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.0.0"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JSONSchema]]
deps = ["HTTP", "JSON", "ZipFile"]
git-tree-sha1 = "b84ab8139afde82c7c65ba2b792fe12e01dd7307"
uuid = "7d188eb4-7ad8-530c-ae41-71a32a6d4692"
version = "0.3.3"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[JuMP]]
deps = ["Calculus", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MathOptInterface", "MutableArithmetics", "NaNMath", "Printf", "Random", "SparseArrays", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "4f0a771949bbe24bf70c89e8032c107ebe03f6ba"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "0.21.9"

[[KernelFunctions]]
deps = ["ChainRulesCore", "Compat", "CompositionsBase", "Distances", "FillArrays", "Functors", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Random", "Requires", "SpecialFunctions", "StatsBase", "TensorCore", "Test", "ZygoteRules"]
git-tree-sha1 = "6f46fb7fa868699dfbb6ae7973ba2825d3558ade"
uuid = "ec8451be-7e33-11e9-00cf-bbf324bd1392"
version = "0.10.16"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

[[LazySets]]
deps = ["Distributed", "ExprTools", "GLPK", "GLPKMathProgInterface", "InteractiveUtils", "IntervalArithmetic", "JuMP", "LinearAlgebra", "MathProgBase", "Random", "RecipesBase", "Reexport", "Requires", "SharedArrays", "SparseArrays"]
git-tree-sha1 = "8562d0b08bc20d72f16155f2f6acf93a5d187d50"
uuid = "b4f0291d-fe17-52bc-9479-3d1a343d9043"
version = "1.49.1"

[[LearnBase]]
deps = ["LinearAlgebra", "StatsBase"]
git-tree-sha1 = "47e6f4623c1db88570c7a7fa66c6528b92ba4725"
uuid = "7f8f8fb0-2700-5f03-b4bd-41f8cfc144b6"
version = "0.3.0"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[LibVPX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "12ee7e23fa4d18361e7c2cde8f8337d4c3101bc7"
uuid = "dd192d2f-8180-539f-9fb4-cc70b1dcf69a"
version = "1.10.0+0"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LightGraphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "432428df5f360964040ed60418dd5601ecd240b6"
uuid = "093fc24a-ae57-5d10-9952-331d41423f4d"
version = "1.3.5"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "3d682c07e6dd250ed082f883dc88aee7996bf2cc"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.0"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MLDataPattern]]
deps = ["LearnBase", "MLLabelUtils", "Random", "SparseArrays", "StatsBase"]
git-tree-sha1 = "e99514e96e8b8129bb333c69e063a56ab6402b5b"
uuid = "9920b226-0b2a-5f5f-9153-9aa70a013f8b"
version = "0.5.4"

[[MLDataUtils]]
deps = ["DataFrames", "DelimitedFiles", "LearnBase", "MLDataPattern", "MLLabelUtils", "Statistics", "StatsBase"]
git-tree-sha1 = "ee54803aea12b9c8ee972e78ece11ac6023715e6"
uuid = "cc2ba9b6-d476-5e6d-8eaf-a92d5412d41d"
version = "0.5.4"

[[MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "0c2bcd5c5b99988bb88552a4408beb3da1f1fd4d"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.2.0"

[[MLLabelUtils]]
deps = ["LearnBase", "MappedArrays", "StatsBase"]
git-tree-sha1 = "3211c1fdd1efaefa692c8cf60e021fb007b76a08"
uuid = "66a33bbf-0c2b-5fc8-a008-9da813334f0a"
version = "0.5.6"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "0fb723cd8c45858c22169b2e42269e53271a6df7"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.7"

[[MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "JSON", "JSONSchema", "LinearAlgebra", "MutableArithmetics", "OrderedCollections", "SparseArrays", "Test", "Unicode"]
git-tree-sha1 = "575644e3c05b258250bb599e57cf73bbf1062901"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "0.9.22"

[[MathProgBase]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9abbe463a1e9fc507f12a69e7f29346c2cdc472c"
uuid = "fdba3010-5040-5b88-9595-932c9decdf73"
version = "0.7.8"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[MiniQhull]]
deps = ["CMake", "Libdl", "Qhull_jll"]
git-tree-sha1 = "e21881c219cc9de70ee990fabfd1f031aa4f414a"
uuid = "978d7f02-9e05-4691-894f-ae31a51d76ca"
version = "0.3.0"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "4ea90bd5d3985ae1f9a908bd4500ae88921c5ce7"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.0"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "3927848ccebcc165952dc0d9ac9aa274a87bfe01"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "0.2.20"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "2276ac65f1e236e0a6ea70baff3f62ad4c625345"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.2"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "477bf42b4d1496b454c10cce46645bb5b8a0cf2c"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.2"

[[PersistenceDiagrams]]
deps = ["Compat", "Hungarian", "MLJModelInterface", "PersistenceDiagramsBase", "RecipesBase", "Statistics", "Tables"]
git-tree-sha1 = "4605264f15a3cf2800774e8ec0ec18d6e87cae1e"
uuid = "90b4794c-894b-4756-a0f8-5efeb5ddf7ae"
version = "0.9.3"

[[PersistenceDiagramsBase]]
deps = ["Compat", "Tables"]
git-tree-sha1 = "ec6eecbfae1c740621b5d903a69ec10e30f3f4bc"
uuid = "b1ad91c1-539c-4ace-90bd-ea06abc420fa"
version = "0.1.1"

[[Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "501c20a63a34ac1d015d5304da0e645f42d91c9f"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.11"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "8365fa7758e2e8e4443ce866d6106d8ecbb4474e"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.20.1"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "44e225d5837e2a2345e69a1d1e01ac2443ff9fcb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.9"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "cde4ce9d6f33219465b55162811d8de8139c0414"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.2.1"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "0d1245a357cc61c8cd61934c07447aa569ff22e6"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.1.0"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[ProgressBars]]
deps = ["Printf"]
git-tree-sha1 = "938525cc66a4058f6ed75b84acd13a00fbecea11"
uuid = "49802e3a-d2f1-5c88-81d8-b72133a6f568"
version = "1.4.0"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[Qhull_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4de074cc5e5f20d221694291dcd79d1b6ec1a104"
uuid = "784f63db-0788-585a-bace-daefebcd302b"
version = "2020.2.0+0"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "12fbe86da16df6679be7521dfb39fbc861e1dc7b"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.1"

[[RCall]]
deps = ["CategoricalArrays", "Conda", "DataFrames", "DataStructures", "Dates", "Libdl", "Missings", "REPL", "Random", "Requires", "StatsModels", "WinReg"]
git-tree-sha1 = "80a056277142a340e646beea0e213f9aecb99caa"
uuid = "6f49c342-dc21-5d91-9882-a32aef131414"
version = "0.13.12"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "2a7a2469ed5d94a98dea0e85c46fa653d76be0cd"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.3.4"

[[Reexport]]
git-tree-sha1 = "5f6c21241f0f655da3952fd60aa18477cf96c220"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.1.0"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Ripserer]]
deps = ["Compat", "DataStructures", "Distances", "Future", "IterTools", "LightGraphs", "LinearAlgebra", "MLJModelInterface", "MiniQhull", "PersistenceDiagrams", "ProgressMeter", "RecipesBase", "SparseArrays", "StaticArrays", "TupleTools"]
git-tree-sha1 = "dc5afb66e97ee3db5d73bcd01f621e3103605cb2"
uuid = "aa79e827-bd0b-42a8-9f10-2b302677a641"
version = "0.16.8"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[Roots]]
deps = ["CommonSolve", "Printf"]
git-tree-sha1 = "06ba8114bf7fc4fd1688e2e4d2259d2000535985"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "1.2.0"

[[RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[ScientificTypesBase]]
git-tree-sha1 = "8a476e63390bfc987aa3cca02d90ea1dbf8b457e"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "2.1.0"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SetRounding]]
git-tree-sha1 = "d7a25e439d07a17b7cdf97eecee504c50fedf5f6"
uuid = "3cc68bcd-71a2-5612-b932-767ffbe40ab0"
version = "0.2.1"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[ShiftedArrays]]
git-tree-sha1 = "22395afdcf37d6709a5a0766cc4a5ca52cb85ea0"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "1.0.0"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "a322a9493e49c5f3a10b50df3aedaf1cdb3244b7"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3240808c6d463ac46f1c1cd7638375cd22abbccb"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.12"

[[StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "730732cae4d3135e2f2182bd47f8d8b795ea4439"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "2.1.0"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "fed1ec1e65749c4d96fc20dd13bea72b55457e62"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.9"

[[StatsFuns]]
deps = ["IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "20d1bb720b9b27636280f751746ba4abb465f19d"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.9"

[[StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "a209a68f72601f8aa0d3a7c4e50ba3f67e32e6f8"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.24"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "000e168f5cc9aded17b6999a560b7c11dda69095"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.0"

[[StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "e36adc471280e8b346ea24c5c87ba0571204be7a"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.7.2"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "d0c690d37c73aeb5ca063056283fde5585a41710"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[TupleTools]]
git-tree-sha1 = "3c712976c47707ff893cf6ba4354aa14db1d8938"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.3.0"

[[URIParser]]
deps = ["Unicode"]
git-tree-sha1 = "53a9f49546b8d2dd2e688d216421d050c9a31d0d"
uuid = "30578b45-9adc-5946-b283-645ec420af67"
version = "0.4.1"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[VersionParsing]]
git-tree-sha1 = "80229be1f670524750d905f8fc8148e5a8c4537f"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.2.0"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[WinReg]]
deps = ["Test"]
git-tree-sha1 = "808380e0a0483e134081cc54150be4177959b5f4"
uuid = "1b915085-20d7-51cf-bf83-8f477d6f5128"
version = "0.3.1"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "c3a5637e27e914a7a445b8d0ad063d701931e9f7"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.9.3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "9e7a1e8ca60b742e508a315c17eef5211e7fbfd7"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.1"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "acc685bcf777b2202a904cdcb49ad34c2fa1880c"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.14.0+4"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7a5780a0d9c6864184b3a2eeeb833a0c871f00ab"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "0.1.6+4"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d713c1ce4deac133e3334ee12f4adff07f81778f"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2020.7.14+2"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "487da2f8f2f0c8ee0e83f39d13037d6bbf0a45ab"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.0.0+3"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─d67c4d46-554b-4fa7-8e96-51f754cdfdb6
# ╟─e3aa21d4-4eff-4185-a352-d0db84a36bd2
# ╟─164f6f93-0fc2-4d32-ac46-16fe6eaf103b
# ╠═551ee4e4-09a1-11ec-3ee1-290dc932aeb5
# ╟─d0240346-5032-4dea-972d-5136df5e28fd
# ╠═a4812be5-0c65-4bb8-8ec8-3d242b5700fb
# ╠═5da3a5af-ba59-4c33-b45a-bb027bed8bd9
# ╠═c132a0bc-cc80-46d9-bd66-ece730869eaf
# ╠═3e658e75-814f-4274-9a2e-01d13e592cc5
# ╠═37d5004d-0a49-4776-8acb-cd9381fea6ee
# ╠═74f073dd-ba00-4ab0-89b0-c7e0e77fbe27
# ╠═cce9f845-b1df-451c-89a9-624a59ccf2db
# ╠═dd2ecb76-122b-4f87-8df2-9e753bbe993b
# ╠═da58526e-93b8-4ffc-b91a-a7e643a8fe68
# ╟─1c876bb8-4364-470a-9c82-09901e624bc9
# ╟─9e84aa65-d2ae-4829-8f0a-0277dc7b29b6
# ╟─04674cae-5f27-4a39-8ac6-3233da5358d9
# ╟─cca9117e-390b-4917-8470-99380fcdd727
# ╠═81f47846-0e22-44b4-a7bc-9005bbe33ab4
# ╟─ef02bcff-7872-4b05-8267-0dc303150ee1
# ╟─2beeb6f9-f49f-451b-a608-503865374190
# ╟─69e1eaf3-992a-4e62-941f-d5bde6ca2abc
# ╟─fc3e0cd3-3284-47d0-b6f9-4329fe345eee
# ╟─e7cf7877-7a4c-4158-a9b6-8c0685d5b766
# ╟─8b96195b-4e1c-4fea-937a-5de98318c27c
# ╠═9bc58a68-a8e3-4a0b-ac4c-173d745954f2
# ╟─e429b51f-705d-4a96-ac60-ffaca191f935
# ╠═2fff8666-5626-41b0-b155-3dd5b7f58d02
# ╟─1753e4ce-f8d0-4e0c-87fa-75cd7950895b
# ╟─ed077d47-471e-4fe3-b505-ff8bc8d0bbcc
# ╟─13ff936a-257d-4ed3-93b8-c0c84bf274ed
# ╠═79d63b7e-4d28-49b3-b662-ce080a5b4b01
# ╟─7f628a8c-0b7d-4a32-99cd-c37b1facc242
# ╠═d582e84b-8c17-4a4d-923a-f39559bef6a9
# ╟─915f12e5-6b05-486c-ac02-103632941ce2
# ╟─1dd935b5-b28e-4f70-ba42-b588a739438a
# ╟─17295b3b-7532-4403-9454-6b305d9dfd38
# ╟─3b059004-beb4-4da5-b812-8a796cabb946
# ╠═8841f0fe-52ea-44cf-a1ff-71624b133cdc
# ╟─5d9c07ac-4d20-43ae-9f71-2bda2750aea8
# ╠═79bfe0ab-2682-4fa8-a508-c3fc9ee3a2fe
# ╟─07ffc1c1-8ec1-4344-b8c5-f0da8ccccb54
# ╟─45ec2b46-95dd-4a84-b088-3232b35b0764
# ╟─f01c10a3-f97d-4d99-b2f8-d86efe4f1402
# ╠═dbfa3ab7-34e8-439f-ba58-5973a2327413
# ╠═55a5bf16-09c6-434e-8460-5b59b5ce5f2f
# ╠═3d5f295b-c00b-452c-8ff8-59da96b4d448
# ╠═a00a1d28-2020-4637-8080-e79b0d3a13bc
# ╟─ccc6ad73-9a5c-4caa-a4d5-096ba39f655e
# ╠═7fb728c2-abe0-4125-9725-b623a6b0f5c5
# ╟─0ebd6dd7-0577-4583-bb3a-c48aa7cd2d7e
# ╟─90cc2493-ddce-481b-bfa0-31ef46dcb3af
# ╠═f9c521dd-c244-4be5-9470-5bbef264e782
# ╠═357b00fd-d236-498c-b26a-56c66d1ec377
# ╠═46ddaff1-c390-47ce-a030-cac2cda73cec
# ╠═8f7a3f7e-aa54-453e-a08f-7d92c5b509b0
# ╠═42c7bb91-75a3-44eb-9f28-26f8277a9f5c
# ╠═219495fa-9c32-4a57-a549-d935ad2458aa
# ╠═f9e87cd3-47d0-46b0-9673-d9f22f613e88
# ╠═d7833440-2fac-4948-8c9f-3fce3185aa86
# ╠═7d3340b3-bd5e-465a-9f4e-9bb3e983b539
# ╠═dafd8044-e44c-452e-9da9-9bbfcabed47c
# ╟─a93e7c84-5dd1-4a32-9f76-31f958ac4b48
# ╟─423f7e1c-9e43-42bf-bd19-624f100e893b
# ╠═49a1c8e0-706f-4910-a6f6-f3c7daf40a4e
# ╠═1a6a324f-6e5e-406b-a2c0-3c4a27919de7
# ╠═0046db0f-63f3-4e2c-b7c4-a427e0d94bdd
# ╠═102d4372-b723-4802-9ced-bec0e12e4c2c
# ╠═72085db6-d480-4204-908d-e529f666dd5e
# ╠═b9d69d61-ccc4-4d5d-b34f-e97b3e729bc7
# ╠═4686bb49-ed62-4cb6-b627-aec97f8c094e
# ╠═b619b0ff-8266-4ac9-8e77-c02b8715fce5
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
