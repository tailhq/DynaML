!!! summary

    The `algebra` package has utility functions for commonly used operations on block matrices and vectors. We give
    the user a glimpse here.

!!! warning

    The routines in this section assume that the block sizes of the input matrix are homogenous i.e. the number of row blocks
    is equal to number of column blocks.

## Blocked Operations

Following operations are the blocked implementations of standard algorithms used for matrices.

### $LU$ Decomposition

[$LU$ decomposition](https://en.wikipedia.org/wiki/LU_decomposition) consists of decomposing a square matrix into
lower and upper triangular factors.

```scala
//Initialize a square matrix
val sq_mat: PartitionedMatrix = _

val (lower, upper) = bLU(sq_mat)
```

### Cholesky Decomposition

[Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition) consists of decomposing a
symmetric positive semi-definite matrix uniquely into lower and upper triangular factors.

```scala
//Initialize a psd matrix
val psd_mat: PartitionedPSDMatrix = _

val (lower, upper) = bcholesky(psd_mat)
```


### Trace

Trace of a square matrix is the sum of the diagonal elements.

```scala
//Initialize a square matrix
val sq_mat: PartitionedMatrix = _

val tr = btrace(sq_mat)
```


### Determinant

The [determinant](https://en.wikipedia.org/wiki/Determinant) of a square matrix represents the scaling factor of the
transformation described by the matrix


```scala
//Initialize a square matrix
val sq_mat: PartitionedMatrix = _

val de = bdet(sq_mat)
```


### Diagonal

Obtain diagonal elements of a square block matrix in the form of a block vector.

```scala
//Initialize a square matrix
val sq_mat: PartitionedMatrix = _

val dia: PartitionedVector = bdiagonal(sq_mat)
```


## Quadratic Forms

Quadratic forms are often encountered in algebra, they involve products on inverse positive semi-definite matrices with
vectors. The two common quadratic forms are.

 - _Self Quadratic Forms_:  $\mathbf{x}^\intercal \Omega^{-1} \mathbf{x}$


 - _Cross Quadratic Form_:  $\mathbf{y}^\intercal \Omega^{-1} \mathbf{x}$


```scala

val (x,y): (DenseVector[Double], DenseVector[Double]) = (_,_)

val omega: DenseMatrix[Double] = _

//Use breeze function
val lower = cholesky(omega)

val x_omega_x = quadraticForm(lower, x)

val y_omega_x = crossQuadraticForm(y, lower, x)

//Blocked Version of the same.

val (xb,yb): (PartitionedVector, PartitionedVector) = (_,_)

val omegab: PartitionedPSDMatrix = _

//Use DynaML algebra function
val lowerb = bcholesky(omegab)

val x_omega_x_b = blockedQuadraticForm(lowerb, xb)

val y_omega_x_b = blockedCrossQuadraticForm(yb, lowerb, xb)


```
