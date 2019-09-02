!!! summary
    Here we show how to use the block matrix API


The [`algebra`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/#io.github.mandar2812.dynaml.algebra.package) package
contains a number of block matrix implementations.

Class | Represents | Notes
------------ | ------------- | -------------
 `#!scala PartitionedMatrix` | A general block matrix | User facing i.e. can be instantiated directly
 `#!scala LowerTriPartitionedMatrix` | Lower triangular block matrix | Result of `algebra` API calls
 `#!scala UpperTriPartitionedMatrix` | Upper triangular block matrix | Result of `algebra` API calls
 `#!scala PartitionedPSDMatrix` | Symmetric positive semi-definite matrix | Result of applying kernel function on data.


## Creation

Block can be created in two major ways.

### From Input Blocks

```scala

val index_set = (0L until 10L).toStream
//Create the data blocks
val data_blocks: Stream[((Long, Long), DenseVector[Double])] =
  utils.combine(index_set).map(
    indices => (
      (indices.head, indices.last), DenseMatrix.ones[Double](500, 500)
    )
  )

//Instantiate the partitioned matrix
//must provide dimensions
val part_matrix = PartitionedMatrix(
  data_blocks, numrows = 5000L, numcols = 5000L)
```

### From Tabulating Functions

```scala

val tabFunc: (Long, Long) => Double =
  (indexR: Long, indexC: Long) => {
    math.sin(2d*math.Pi*indexR/5000d)*math.cos(2d*math.Pi*indexC/5000d)
  }

//Instantiate the partitioned matrix
val part_matrix = PartitionedMatrix(
  nRows = 5000L, nCols = 5000L,
  numElementsPerRBlock = 1000,
  numElementsPerCBlock = 1000,
  tabFunc)
```

### From Outer Product

A `#!scala PartitionedMatrix` can also be constructed from the product of a `#!scala PartitionedDualVector` and
`#!scala PartitionedVector`.

```scala
val random_var = RandomVariable(new Beta(1.5, 2.5))

val rand_vec1 = PartitionedVector.rand(2000L, 500, random_var)
val rand_vec2 = PartitionedVector.rand(2000L, 500, random_var)

val p_mat = rand_vec1*rand_vec2.t
```

### Matrix Concatenation

You can vertically join matrices, as long as the number of rows and row blocks match.

```scala

val mat1: PartitionedMatrix = _
val mat2: PartitionedMatrix = _

val mat3 = PartitionedMatrix.vertcat(mat1, mat2)
```

!!! note "Positive Semi-Definite Matrices"
    The class `#!scala PartitionedPSDMatrix` can be instantiated in two ways.

      - From outer product.

        ```scala
        val random_var = RandomVariable(new Beta(1.5, 2.5))

        val rand_vec = PartitionedVector.rand(2000L, 500, random_var)
        val psd_mat = PartitionedPSDMatrix.fromOuterProduct(rand_vec)
        ```

      - From kernel evaluation

        ```scala
        //Obtain data
        val data: Seq[DenseVector[Double]] = _
        //Create kernel instance
        val kernel: LocalScalarKernel[DenseVector[Double]] = _

        val psd_gram_mat = kernel.buildBlockedKernelMatrix(data, data.length)
        ```


## Algebraic Operations

Partitioned vectors and dual vectors have a number of algebraic operations available in the API.

```scala
val beta_var = RandomVariable(Beta(1.5, 2.5))
val gamma_var = RandomVariable(Gamma(1.5, 2.5))

val p_vec_beta = PartitionedVector.rand(5000L, 1000, beta_var)
val p_vec_gamma = PartitionedVector.rand(5000L, 1000, gamma_var)

val dvec_beta = p_vec_beta.t
val dvec_gamma = p_vec_gamma.t

val mat1 = p_vec_gamma*dvec_gamma
val mat2 = p_vec_beta*dvec_beta

//Addition
val add_mat = mat1 + mat2

//Subtraction
val sub_mat = mat2 - mat1

//Element wise multiplication
val mult_mat = mat1 :* mat2

//Matrix matrix product

val prod_mat = mat1*mat2

//matrix vector Product
val prod = mat1*p_vec_beta
val prod_dual = dvec_gamma*mat2

//Scaler multiplication
val sc_mat = mat1*1.5
```

## Misc. Operations

### Map Partitions

Map each index, partition pair by a Scala function.

```scala
val vec: PartitionedMatrix = _

val other_vec = vec.map(
   (pair: ((Long, Long), DenseMatrix[Double])) => (pair._1, pair._2*1.5)
)
```

### Slice

Obtain subset of elements, the new matrix is repartitioned and re-indexed accordingly.

```scala
val vec: PartitionedVector = PartitionedVector.ones(5000L, 1000)

val mat = vec*vec.t

val other_mat = vec(999L until 2000L, 0L until 999L)
```

### Upper and Lower Triangular Sections


```scala
val vec: PartitionedVector = PartitionedVector.ones(5000L, 1000)

val mat = vec*vec.t

val lower_tri: LowerTriPartitionedMatrix = mat.L
val upper_tri: UpperTriPartitionedMatrix = mat.U

```

### Convert to Breeze Matrix

```scala
val mat: PartitionedMatrix = _

//Do not use on large vectors as
//it might lead to overflow of memory.
val breeze_mat = mat.toBreezeMatrix
```
