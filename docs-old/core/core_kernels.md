!!! summary ""

    The [`dynaml.kernels`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/#io.github.tailhq.dynaml.kernels.package) package has a highly developed API for creating kernel functions for machine learning applications. Here
    we give the user an in-depth introduction to its capabilities.

<br/>

![kernel](../../images/kernel.png)


_Positive definite_ functions or _positive type_ functions occupy an important place in various areas of mathematics, from the construction of covariances of random variables to quantifying distance measures in _Hilbert spaces_. Symmetric positive type functions defined on the cartesian product of a set with itself $K: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$ are also known as _kernel_ functions in machine learning. They are applied extensively in problems such as.

1. Model non-linear behavior in SVM models: [_SVM_](https://en.wikipedia.org/wiki/Support_vector_machine) and [_LSSVM_](http://www.worldscientific.com/worldscibooks/10.1142/5089)
2. Quantify covariance between input patterns: [_Gaussian Processes_](http://www.gaussianprocess.org/gpml/)
3. Represent degree of 'closeness' or affinity in unsupervised learning: [_Kernel Spectral Clustering_](http://arxiv.org/pdf/1505.00477.pdf)

For an in depth review of the various applications of kernels in the machine learning domain, refer to [Scholkopf et. al](http://www.kernel-machines.org/publications/pdfs/0701907.pdf)

!!! note "Nomenclature"
    In the machine learning community the words _kernel_ and _covariance function_ are used interchangeably.

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

    For this reason there is the [`#!scala  CovarianceFunction[T, V, M]`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/index.html#io.github.tailhq.dynaml.kernels.CovarianceFunction) abstract class. It contains methods to construct kernel matrices, keep track of hyper-parameter assignments among other things.

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

!!! note "Feature map kernels"
    Covariance functions constructed using feature mappings as shown above return a special object; an instance of the `#!scala FeatureMapCovariance[T, DenseVector[Double]]` class. In the section on composite kernels we will see why this is important.

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


### Algebraic Operations

In machine learning it is well known that kernels can be combined to give other valid kernels. The symmetric positive semi-definite property of a kernel is preserved as long as it is added or multiplied to another valid kernel. In DynaML adding and multiplying kernels is elementary.

```scala

val k1 = new RBFKernel(2.5)
val k2 = new RationalQuadraticKernel(2.0)

val k = k1 + k2
val k3 = k*k2
```

### Composition

From Mercer's theorem, every kernel can be expressed as a dot product of feature mappings evaluated at the respective data points. We can use this to construct more complex covariances i.e. by successively applying feature mappings.

$$
\begin{align}
C_{a}(\mathbf{x}, \mathbf{y}) &= \varphi_{a}(\mathbf{x})^\intercal \varphi_{a}(\mathbf{y}) \\
C_{b}(\mathbf{x}, \mathbf{y}) &= \varphi_{b}(\mathbf{x})^\intercal \varphi_{b}(\mathbf{y}) \\
C_{b . a}(\mathbf{x}, \mathbf{y}) &= \varphi_{b}(\varphi_{a}(\mathbf{x}))^\intercal \varphi_{b}(\varphi_{a}(\mathbf{y}))
\end{align}
$$

In DynaML, we can create a composite kernel if the kernel represented by the map $\varphi_{a}$, is explicitly of type `#!scala FeatureMapCovariance[T, DenseVector[Double]]`

```scala
val mapFunc = (vec: DenseVector[Double]) => {
	vec/2d
}

val k1 = CovarianceFunction(mapFunc)

val k2 = new RationalQuadraticKernel(2.0)

//Composite kernel
val k3 = k2 > k1
```

### Scaling Covariances

If $C(\mathbf{x}, \mathbf{y})$ is a valid covariance function, then $g(\mathbf{x}) C(\mathbf{x}, \mathbf{y}) g(\mathbf{x})$ is also a valid covariance function, where $g(.): \mathcal{X} \rightarrow \mathbb{R}$ is a non-negative function from the domain of the inputs $\mathcal{X}$ to the real number line. We call these covariances _scaled covariance functions_.

```scala
//Instantiate some kernel
val kernel: LocalScalarKernel[I] = _

val scalingFunction: (I) => Double = _

val scKernel = ScaledKernel(
  kernel, DataPipe(scalingFunction))
```

### Advanced Composite Kernels

Sometimes we would like to express a kernel function as a product (or sum) of component kernels each of which act on
a sub-set of the dimensions (degree of freedom) of the input attributes.

For example; for 4 dimensional input vector, we may define two component kernels acting on the first two and
last two dimensions respectively and combine their evaluations via addition or multiplication. For this purpose the
[`dynaml.kernels`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/#io.github.mandar2812.dynaml.kernels.package) package has the [`#!scala DecomposableCovariance[S]`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-core/#io.github.mandar2812.dynaml.kernels.DecomposableCovariance) class.


In order to create a decomposable kernel you need three components.

 1. The component kernels (order matters)
 2. An [`#!scala Encoder[S, Array[S]]`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-pipes/#io.github.mandar2812.dynaml.pipes.Encoder) instance which splits the input into an array of components
 3. A [`#!scala Reducer`](https://transcendent-ai-labs.github.io/api_docs/DynaML/recent/dynaml-pipes/#io.github.mandar2812.dynaml.pipes.Reducer$) which combines the individual kernel evaluations.

```scala
//Not required in REPL, already imported
import io.github.tailhq.dynaml.DynaMLPipe._
import io.github.tailhq.dynaml.pipes._

val kernel1: LocalScalarKernel[DenseVector[Double]] = _
val kernel2: LocalScalarKernel[DenseVector[Double]] = _

//Default Reducer is addition
val decomp_kernel =
  new DecomposableCovariance[DenseVector[Double]](
    kernel1, kernel2)(
    breezeDVSplitEncoder(2))

val decomp_kernel_mult =
  new DecomposableCovariance[DenseVector[Double]](
    kernel1, kernel2)(
    breezeDVSplitEncoder(2),
    Reducer.:*:)
```

-----

!!! seealso "Implementing Custom Kernels"
    You can implement your own custom kernels by extending the `LocalScalarKernel[T]` interface, for example:
    
    ```scala
    import breeze.linalg.{DenseMatrix, norm, DenseVector}
    
    //You can have any number of constructor parameters
    class MyNewKernel(th: Double = 1.0)
      extends LocalScalarKernel[DenseVector[Double]]
      with Serializable {
      
      //One must specify the names of each hyper-parameter
      override val hyper_parameters = List("theta")
    
      //The state variable stores the 
      //current value of all kernel hyper-parameters
      state = Map("theta" -> th)
    
      // The implementation of the actual kernel function
      override def evaluateAt(
        config: Map[String, Double])(
        x: DenseVector[Double],
        y: DenseVector[Double]): Double = ???
    
      // Return the gradient of the kernel for each hyper-parameter
      // for a particular pair of points x,y
      override def gradientAt(
        config: Map[String, Double])(
        x: DenseVector[Double],
        y: DenseVector[Double]): Map[String, Double] = ???
    }
    
    ``` 

