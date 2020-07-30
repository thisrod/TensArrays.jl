module TensArrays

export TensArray

struct TensArray{T} <: AbstractMatrix{T}
    parent::AbstractMatrix{T}
    cs::Dims
    rs::Dims
    
    function TensArray(x::AbstractMatrix{T}, cs, rs) where T
        if size(x) == (prod(cs), prod(rs))
            new{T}(Array(x), cs, rs)
        else
            throw(DimensionMismatch())
        end
    end
end

parent(x::TensArray) = x.parent

Base.size(x::TensArray) = size(parent(x))
Base.getindex(x::TensArray, j...) = x.parent[j...]
Base.setindex!(x::TensArray, v, j) = (x.parent[j] = v)

Base.similar(x::TensArray, T, dims) =
    TensArray(similar(parent(x), T, dims), x.cs, x.rs)

# TODO show with rowsize and colsize

# matmul

function Base.:*(x::TensArray, y::TensArray)
    a = parent(x)*parent(y)
    x.rs == y.cs ? TensArray(a, x.cs, y.rs) : a
end

# Broadcasting, copied from the ArrayAndChar magic pixie dust
# For now, the tensor shape is discarded except for scalars and exactly matching matrices.

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
