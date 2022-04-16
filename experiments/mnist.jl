using DrWatson
@quickactivate "wrips-code"

begin
    using Random, Plots, Pipe, Images, FileIO
    using Ripserer, PersistenceDiagrams, PersistenceDiagramsBase
    include(srcdir("wRips.jl"))
    import Main.wRips
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

img = load("./experiments/data/6.png")

img_matrix = @pipe img |> conv
for u in 8:13, v in 19:20
    img_matrix[u, v] = 0.85
end
img_matrix[14, 20] = 0.85
new_img = img_matrix |> conv

# FileIO.save("./experiments/data/8.png", new_img)

begin
    Random.seed!(2022)
    seq = 1:28
    X = Tuple{Float64,Float64}[]
    for i in seq, j in seq
        if img_matrix[i, j] > 0
            X = [X
                @pipe wRips.randUnif(round(Int, img_matrix[i, j] * 10); a = 0.01, b = 0.99) .|> _ .+ (j, 28 - i)
            ]
        end
    end
    Xn = X |> wRips._ArrayOfTuples_to_ArrayOfVectors
    X = Xn |> wRips._ArrayOfVectors_to_ArrayOfTuples
    plt3 = @pipe X |> scatter(_, ratio = 1, label = nothing)
end

savefig(plt3, "./experiments/plots/8-scatter.pdf")

begin
    m = 110
    Q = 2 * m + 1
    Random.seed!(2022)
    dnq = wRips.momdist(Xn, floor(Int, Q))
    w_momdist = wRips.fit(Xn, dnq)
    @pipe X |> scatter(_, marker_z = w_momdist, ratio = 1, label = nothing)
end

begin
    xseq = 1:1:27
    cls = palette(:plasma, 1000)
    plt4 = plot(xseq, xseq, (x, y) -> 4 / wRips.fit([[x, y]], dnq)^2, st = :heatmap, c = cls, axis = ([], false))
end

# plt5 = plt4

# savefig(plot(plt4, size = (500, 400)), "./experiments/plots/denoised-small.pdf")
# savefig(plot(plt4, size = (500, 400)), "./experiments/plots/denoised.pdf")

denoised_img = [1 / wRips.fit([[y, x]], dnq) for x in 28:-1:1, y in 1:28]
denoised_img = [x / (denoised_img |> maximum) for x in denoised_img] |> conv


D1 = Cubical(@pipe img |> conv .|> -1 * _) |> ripserer
D2 = Cubical(@pipe new_img |> conv .|> -1 * _) |> ripserer
D3 = Cubical(@pipe denoised_img |> conv .|> -1 * _) |> ripserer

plot(
    plot(D1, title = "", lim = (-1.1, 0.2)),
    plot(D2[2], title = "", lim = (-1.1, 0.2)),
    plot(D3[2], title = "", lim = (-1.1, 0.2)),
    layout = (1, 3), size = (950, 300)
)





begin
    img_matrix_6 = img |> conv
    Random.seed!(2022)
    seq = 1:28
    signal = Tuple{Float64,Float64}[]
    for i in seq, j in seq
        if img_matrix_6[i, j] > 0
            signal = [signal
                @pipe wRips.randUnif(round(Int, img_matrix_6[i, j] * 10); a = 0.01, b = 0.99) .|> _ .+ (j, 28 - i)
            ]
        end
    end
    signaln = signal |> wRips._ArrayOfTuples_to_ArrayOfVectors
    signal = signaln |> wRips._ArrayOfVectors_to_ArrayOfTuples
    plt6 = @pipe signal |> scatter(_, ratio = 1, label = nothing)
end

D_6 = wRips.wrips(signaln, w = nothing, dim_max = 1, sparse = true)
D_8 = wRips.wrips(Xn, w = nothing, dim_max = 1, sparse = true)
D = wRips.wrips(Xn, w = w_momdist, dim_max = 1, sparse = true, p = 2)

savefig(scatter(signal, size = (500, 400), label = nothing, ratio = 1), "./experiments/plots/scatter-6.pdf")
savefig(scatter(X, size = (500, 400), label = nothing, ratio = 1), "./experiments/plots/scatter-8.pdf")
savefig(scatter(X, size = (500, 400), label = nothing, ratio = 1, marker_z = w_momdist), "./experiments/plots/scatter-8-heat.pdf")

@pipe plot(D_6[2], title = "", lim = (-0.2, 4), label = L"H_1", size=(400,400)) |> savefig(_, "./experiments/plots/dgm-6.pdf")
@pipe plot(D_8[2], title = "", lim = (-0.2, 4), label = L"H_1", size=(400,400)) |> savefig(_, "./experiments/plots/dgm-8.pdf")
@pipe plot(D[2], title = "", lim = (-0.2, 4), label = L"H_1", size=(400,400)) |> savefig(_, "./experiments/plots/dgm-denoised.pdf")