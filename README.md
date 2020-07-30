# TensArrays.jl

Kronecker product matrices that remember their tensor shape.

This a minimal package, to be included in other packages so that they can interoperate with `Tensars`.  For example Jacobian shapes.

Logic: if an operation is consistent with the tensor shapes, it returns a TensArray.  Otherwise, it lowers everything to Matrix.

Think about TensArray*vector and adjoint*TensArray.  In the unilinear case, the lowered vectors and matrices construct the same Tensars as their TensArrays do, so don't think too hard.

If Tensar stored a matrix, it could have the same in-memory layout as TensArray.  This would make conversion trivial.  In particular, Tensar can use the operations defined for TensArray, so there is no need to redefine broadcasting.  Except that Tensars should broadcast differently: an (x⊗y) tensar should broadcast along the y axis, but the TensArray is a column vector that should only broadcast over the whole plane.