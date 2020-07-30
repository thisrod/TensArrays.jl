module TensArrays

export TensArray

"""
    TensArray{T,N} <: AbstractArray{T,N}
    TensArray(x, cs, rs)

An Array with co- and contravariant dimensions.

TensArray wraps an AbstractArray with metadata that allows it to
be interpreted as a tensor.  Refer to the manual of Tensars.jl for
details.
"""
struct TensArray{T,N} <: AbstractArray{T,N}
    parent::AbstractArray{T,N}
    cs::Dims
    rs::Dims
    
    function TensArray(x::AbstractArray{T,N}, cs, rs) where T where N
        if match_shape(size(x), cs, rs)
            new{T,N}(Array(x), cs, rs)
        else
            "Array dimensions do not match tensor dimensions" |>
                DimensionMismatch |> throw
        end
    end
end

function match_shape(as, cs, rs)
    cs = isempty(cs) ? (1,) : cs
    rs = isempty(rs) ? (1,) : rs
    as = cumprod(collect(as))
    as[end] == prod(cs)*prod(rs) &&
        prod(cs) in as &&
        issubseq(as, cumprod([cs..., rs...]))
end

function issubseq(xs, ys)
    if isempty(xs)
        true
    elseif isempty(ys)
        false
    elseif first(xs) == first(ys)
        issubseq(xs[2:end], ys[2:end])
    else
        issubseq(xs[2:end], ys)
    end
end

parent(x::TensArray) = x.parent

Base.size(x::TensArray) = size(parent(x))
Base.getindex(x::TensArray, j...) = x.parent[j...]
Base.setindex!(x::TensArray, v, j) = (x.parent[j] = v)

Base.similar(x::TensArray, T, dims) =
    TensArray(similar(parent(x), T, dims), x.cs, x.rs)

# matmul

function Base.:*(x::TensArray, y::TensArray)
    a = parent(x)*parent(y)
    x.rs == y.cs ? TensArray(a, x.cs, y.rs) : a
end

# TODO reshaping

# Broadcasting, copied from the ArrayAndChar magic pixie dust.  For
# now, the tensor shape is discarded except for scalars and exactly
# matching matrices.

Base.BroadcastStyle(::Type{<:TensArray}) = Broadcast.ArrayStyle{TensArray}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{TensArray}}, ::Type{ElType}) where ElType
    # Find the TensArray
    A = find_aac(bc)
    # Create the output
    pA = similar(Array{ElType}, axes(bc))
    (A == false) ? pA : TensArray(pA, A.cs, A.rs)
end

find_aac(bc::Base.Broadcast.Broadcasted) = find_aac(bc.args)
find_aac(::Tuple{}) = true
find_aac(args::Tuple) = find_aac(find_aac(args[1]), Base.tail(args))
find_aac(x) = isempty(size(x))
find_aac(a::TensArray) = a
find_aac(a, rest) = merge_aac(a, find_aac(rest))

merge_aac(a::Bool, b::TensArray) = a && b
merge_aac(b::TensArray, a::Bool) = a && b
merge_aac(a::Bool, b::Bool) = a && b
merge_aac(a::TensArray, b::TensArray) =
    (a.cs, a.rs) == (b.cs, b.rs) && a

end # module
