using DrWatson
@quickactivate "momdist"

begin
    using Random, Plots, Pipe, Images, FileIO
    using Ripserer, PersistenceDiagrams, PersistenceDiagramsBase

    import RobustTDA as rtda

    theme(:dao)
end

function conv(X)
    if typeof(X) <: Matrix{<:Number}
        return Gray.(X)
    elseif typeof(X) <: Matrix{<:Gray}
        return map(x -> Base.convert(Float16, x), gray.(X))
    else
        @error "Not a valid datatype"
    end
end

img = load(datadir("6.png"))

img_matrix_8 = @pipe img |> conv
for u in 8:13, v in 19:20
    img_matrix_8[u, v] = 0.85
end
img_matrix_8[14, 20] = 0.85
new_img = img_matrix_8 |> conv

# FileIO.save("./experiments/data/8.png", new_img)

offset = 0.05

begin
    Random.seed!(2022)
    seq = 1:28
    X = Tuple{Float64,Float64}[]
    for i in seq, j in seq
        if img_matrix_8[i, j] > 0
            X = [X
                @pipe rtda.randUnif(round(Int, img_matrix_8[i, j] * 10); a=offset, b=1 - offset) .|> _ .+ (j, 28 - i)
            ]
        end
    end
    Xn = X |> rtda._ArrayOfTuples_to_ArrayOfVectors
    X = Xn |> rtda._ArrayOfVectors_to_ArrayOfTuples
    plt3 = @pipe X |> scatter(_, ratio=1, label=nothing)
    plt3 = plot(plt3, size=(300, 300), axis=false, ticks=false, legend=false)
end


begin
    img_matrix_6 = img |> conv
    Random.seed!(2022)
    seq = 1:28
    signal = Tuple{Float64,Float64}[]
    for i in seq, j in seq
        if img_matrix_6[i, j] > 0
            signal = [signal
                @pipe rtda.randUnif(round(Int, img_matrix_6[i, j] * 10); a=offset, b=1 - offset) .|> _ .+ (j, 28 - i)
            ]
        end
    end
    signaln = signal |> rtda._ArrayOfTuples_to_ArrayOfVectors
    signal = signaln |> rtda._ArrayOfVectors_to_ArrayOfTuples
    plt6 = @pipe signal |> scatter(_, ratio=1, label=nothing)
end

function plt(X; xlim=(0, 3))
    return plot(X, title="", persistence=true, ylim=(-0.1, 3), xlim=xlim, ratio=1, size=(300, 300), markersize=7, legendfontsize=10, markeralpha=0.9)
end

begin
    Random.seed!(2022)
    m = length(Xn) - length(signaln)
    Q = 2 * m + 1
    Random.seed!(2022)
    dnq = rtda.momdist(Xn, floor(Int, Q))
    w_momdist = rtda.fit(Xn, dnq)
end


begin
    D1 = @pipe signaln |> ripserer(_, dim_max=1)
    D2 = @pipe Xn |> ripserer(_, dim_max=1)
    D3 = @pipe Xn |> rtda.wrips(_, w=w_momdist, dim_max=1, p=2)
end;

begin
    theme(:default)
    plot(
        plt(D1[2]),
        plt(D2[2]),
        plt(D3[2], xlim=(1, 4)),
        layout=(1, 3), size=(750, 300))
end

begin
    savefig(plt(D1[2]), plotsdir("mnist/dgm6.pdf"))
    savefig(plt(D2[2]), plotsdir("mnist/dgm8.pdf"))
    savefig(plt(D3[2], xlim=(1, 4)), plotsdir("mnist/rdgm8.pdf"))
end


begin
    fitted_img = [rtda.fit([[y, x]], dnq) for x in 28:-1:1, y in 1:28]
    mmax = maximum(fitted_img)
    denoised_img = [(mmax - rtda.fit([[y, x]], dnq)) / mmax for x in 1:1:28, y in 1:28]
    denoised_img_alt = [(mmax - rtda.fit([[y, x]], dnq)) / mmax for x in 1:0.1:28, y in 1:0.1:28]
end
heatplot = heatmap(denoised_img_alt, axes=false, ticks=false, label=false, ratio=1, xlim=(20, 280), c=cgrad(:inferno, 0.92:0.01:1, rev=:false))

heatplot = heatmap(denoised_img, axes=false, ticks=false, label=false, ratio=1, xlim=(2, 28), c=cgrad(:inferno, 0.92:0.01:1))

savefig(plot(heatplot, size=(280, 280)), plotsdir("mnist/heatmap2.pdf"))

begin
    D1 = Cubical(@pipe img .|> -1 * _) |> ripserer
    D2 = Cubical(@pipe new_img .|> 1 * _) |> ripserer
    D3 = Cubical(@pipe denoised_img .|> -1 * _) |> ripserer
end
