!!! summary
    Here we show how to use blocked vectors and blocked dual vectors.


Blocked vectors and dual vectors in the [`algebra`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/#io.github.mandar2812.dynaml.algebra.package) package are wrappers around `#!scala Stream[(Long, DenseVector[Double])]`
each partition consists of an ordered index and the partition content which is in the form of a breeze vector.

The relevant API endpoints are `#!scala PartitionedVector` and `#!scala PartitionedDualVector`. In order to access these
objects, you must do `#!scala import io.github.mandar2812.dynaml.algebra._` (already loaded by default in the DynaML shell).

## Creation

Block vectors can be created in a number of ways. `#!scala PartitionedVector` and `#!scala PartitionedDualVector`
are column row vectors respectively and treated as transposes of each other.

### From Input Blocks

```scala

//Create the data blocks
val data_blocks: Stream[(Long, DenseVector[Double])] =
  (0L until 10L).toStream.map(index => (index, DenseVector.ones[Double](500)))

//Instantiate the partitioned vector

val part_vector = PartitionedVector(data_blocks)

//Optionally you may also provide the total length
//of the partitioned vector

val part_vector = PartitionedVector(data_blocks, num_rows: Long = 5000L)


//Created Block Dual Vector

val part_dvector = PartitionedDualVector(data_blocks)

//Optionally you may also provide the total length
//of the partitioned dual vector

val part_dvector = PartitionedDualVector(data_blocks, num_rows: Long = 5000L)



```

### From Tabulating Functions

```scala

val tabFunc: (Long) => Double =
  (index: Long) => math.sin(2d*math.Pi*index/5000d)

//Instantiate the partitioned vector
val part_vector = PartitionedVector(
  length = 5000L, numElementsPerBlock = 500,
  tabFunc)

//Instantiate the partitioned dual vector
val part_dvector = PartitionedDualVector(
  length = 5000L, numElementsPerBlock = 500,  tabFunc)
```



### From a Stream

```scala

//Create the data stream
val data: Stream[Double] = Stream.fill[Double](5000)(1.0)

//Instantiate the partitioned vector
val part_vector = PartitionedVector(data, length = 5000L, num_elements_per_block = 500)
```


### From a Breeze Vector

```scala

//Create the data blocks
val data_vector = DenseVector.ones[Double](5000)

//Instantiate the partitioned vector
val part_vector = PartitionedVector(data_vector, num_elements_per_block = 500)
```

Apart from the above methods of creation there are a number of convenience functions available.

### Vector with Filled Values

#### Vector of zeros

```scala
val ones_vec = PartitionedVector.zeros(5000L, 500)
```

#### Vector of Ones

```scala
val ones_vec = PartitionedVector.ones(5000L, 500)
```

#### Vector of Random Values

```scala
val random_var = RandomVariable(new Beta(1.5, 2.5))

val rand_vec = PartitionedVector.rand(5000L, 500, random_var)
```

### Vector Concatenation

```scala

val random_var = RandomVariable(new Beta(1.5, 2.5))

val rand_vec1 = PartitionedVector.rand(2000L, 500, random_var)
val rand_vec2 = PartitionedVector.rand(2000L, 500, random_var)

//Vector of length 4000, having 8 blocks of 500 elements each
val vec = PartitionedVector.vertcat(rand_vec1, rand_vec2)
```

!!! tip
    A `#!scala PartitionedDualVector` can be created via the transpose operation
    on a `#!scala PartitionedVector` instance and vice versa.

    ```scala
    val random_var = RandomVariable(new Beta(1.5, 2.5))

    val p_vec = PartitionedVector.rand(5000L, 500, random_var)

    val p_dvec = p_vec.t
    ```


## Algebraic Operations

Partitioned vectors and dual vectors have a number of algebraic operations available in the API.

```scala

val beta_var = RandomVariable(Beta(1.5, 2.5))
val gamma_var = RandomVariable(Gamma(1.5, 2.5))

val p_vec_beta = PartitionedVector.rand(5000L, 500, beta_var)
val p_vec_gamma = PartitionedVector.rand(5000L, 500, gamma_var)

val dvec_beta = p_vec_beta.t
val dvec_gamma = p_vec_gamma.t

//Addition
val add_vec = p_vec_beta + p_vec_gamma
val add_dvec = dvec_beta + dvec_gamma

//Subtraction
val sub_vec = p_vec_beta - p_vec_gamma
val sub_dvec = dvec_beta - dvec_gamma

//Element wise multiplication
val mult_vec = p_vec_beta :* p_vec_gamma

//Element wise division
val div_vec = p_vec_beta :/ p_vec_gamma

//Inner Product
val prod = dvec_gamma*p_vec_beta

//Scaler multiplication
val sc_vec = add_vec*1.5
val sc_dvec = add_dvec*2.5
```
