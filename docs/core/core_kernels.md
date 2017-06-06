---
title: Kernels
---


![kernel](/images/kernel.png)


_Positive definite_ functions or _positive type_ functions occupy an important place in various areas of mathematics, from the construction of covariances of random variables to quantifying distance measures in _Hilbert spaces_. Symmetric positive type functions defined on the cartesian product of a set with itself $K: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$ are also known as _kernel_ functions in machine learning. They are applied extensively in problems such as.

1. Model non-linear behavior in SVM models: [_SVM_](https://en.wikipedia.org/wiki/Support_vector_machine) and [_LSSVM_](http://www.worldscientific.com/worldscibooks/10.1142/5089)
2. Quantify covariance between input patterns: [_Gaussian Processes_](http://www.gaussianprocess.org/gpml/)
3. Represent degree of 'closeness' or affinity in unsupervised learning: [_Kernel Spectral Clustering_](http://arxiv.org/pdf/1505.00477.pdf)

For an in depth review of the various applications of kernels in the machine learning domain, refer to [Scholkopf et. al](http://www.kernel-machines.org/publications/pdfs/0701907.pdf)

## Kernel API

The kernel class hierarchy all stems from a simple trait shown here.

```scala
trait Kernel[T, V] {
  def evaluate(x: T, y: T): V
}
```

This outlines only one key feature for kernel functions i.e. their evaluation functional which takes two inputs from $\mathcal{X}$ and yields a scalar value.

!!! note "`#!scala Kernel` vs `#!scala CovarianceFunction`"
    <br/>
    For practical purposes, the [`#!scala  Kernel[T, V]`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/index.html#io.github.mandar2812.dynaml.kernels.Kernel) trait does not have enough functionality for usage in varied models like [_Gaussian Processes_](/core/core_gp.md), [_Student's T Processes_](/core/core_stp.md), [_LS-SVM_](core_lssvm.md) etc.

    For this reason there is the [`#!scala  CovarianceFunction[T, V, M]`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/index.html#io.github.mandar2812.dynaml.kernels.CovarianceFunction) abstract class. It contains methods to construct kernel matrices, keep track of hyper-parameter assignments among other things.

## Creating arbitrary kernel functions

Apart from off the shelf kernel functions, it is also possible to create custom kernels on the fly by using the `#!scala CovarianceFunction` object.

### Constructing kernels via feature maps

It is known from [Mercer's theorem](https://en.wikipedia.org/wiki/Mercer%27s_theorem) that any valid kernel function must be decomposable as a dot product between certain _basis function_ representation of the inputs. This translates mathematically into.

$$
\begin{align}
	& K(\mathbf{x}, \mathbf{y}) = \varphi^{T}(\mathbf{x}) . \varphi(\mathbf{y}) \\
	& \varphi(.): \mathcal{X} \rightarrow \mathbb{R}^n
\end{align}
$$

The function $\varphi(.)$ is some higher (possibly infinite) dimensional representation of the input features of a data point. Note that the input space $\mathcal{X}$ could be any of the following (but not limited to).

* The space of all connection graphs with specific number of nodes.

* A multi-dimensional vector.

* The space of all character sequences (binary or otherwise) up to a certain length.

* The set of all integer tuples e.g. $(1,2), (6,10), \cdots$

 We can use any function from some domain $\mathcal{X}$ yielding a `#!scala DenseVector[Double]` to define a particular inner product/kernel function.

```scala
// First create a function mapping from some input space to
// Breeze dense vectors.

val mapFunc = (vec: DenseVector[Double]) => {
	val mat = vec * vec.t
	mat.toDenseVector
}

val kernel = CovarianceFunction(mapFunc)

```

### Constructing kernels via direct evaluation

Instead of defining a feature representation like $\varphi(.)$ as in the section above, you can also directly define the evaluation expression of the kernel.

```scala

// Create the expression for the required kernel.
val mapFunc =
(state: Map[String, Double]) =>
  (x: DenseVector[Double], y: DenseVector[Double]) => {
	   state("alpha")*(x dot y) + state("intercept")
  }

//Creates kernel with two hyper-parameters: alpha and intercept
val kernel = CovarianceFunction(mapFunc)(
  Map("alpha" -> 1.5, "intercept" -> 0.01)
)

```


-----

## Creating Composite Kernels

In machine learning it is well known that kernels can be combined to give other valid kernels. The symmetric positive semi-definite property of a kernel is preserved as long as it is added or multiplied to another valid kernel. In DynaML adding and multiplying kernels is elementary.


```scala

val k1 = new RBFKernel(2.5)
val k2 = new RationalQuadraticKernel(2.0)

val k = k1 + k2
val k3 = k*k2
```

-----

!!! seealso "Implementing Custom Kernels"
    For more details on implementing user defined kernels, refer to the [wiki](https://github.com/mandar2812/DynaML/wiki/Kernels).
