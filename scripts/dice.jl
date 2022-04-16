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


function image_to_pts(img; N = 10)
    img = img |> conv
    n, m = size(img)
    X = Any[]
    for i in 1:n, j in 1:m
        if img[i, j] > 0
            X = [X
                @pipe wRips.randUnif(round(Int, img[i, j] * N); a = 0.01, b = 0.99) .|> _ .+ (j, n - i)
            ]
        end
    end
    return X |> wRips._ArrayOfTuples_to_ArrayOfVectors
end


img4 = load("./experiments/data/dice4.png") .|> Gray
img5 = load("./experiments/data/dice5.png") .|> Gray


Xn4 = image_to_pts(img4, N = 5)
Xn5 = image_to_pts(img5, N = 5)

X4 = Xn4 |> wRips._ArrayOfVectors_to_ArrayOfTuples
X5 = Xn5 |> wRips._ArrayOfVectors_to_ArrayOfTuples

@pipe X4 |> scatter(_, ratio = 1, label = nothing)
@pipe X5 |> scatter(_, ratio = 1, label = nothing)

begin
    m = 400
    Q = 2 * m + 1
    Random.seed!(2022)
    dnq = wRips.momdist(Xn5, floor(Int, Q))
    w_momdist = wRips.fit(Xn5, dnq)
    @pipe X5 |> scatter(_, marker_z = w_momdist, ratio = 1, label = nothing)
end

begin
    xseq = 1:1:50
    cls = palette(:plasma, 5)
    heatplot = plot(xseq, xseq, (x, y) -> 4 / wRips.fit([[x, y]], dnq)^2, st = :heatmap, c = cls, axis = ([], false))
end
