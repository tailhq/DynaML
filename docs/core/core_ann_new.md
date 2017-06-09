!!! summary
    In v1.4.2, the neural stack API was introduced. It defines some primitives for modular
    construction of neural networks. The main idioms will be computational layers & stacks.  


The neural stack API extends these abstract skeletons by defining two kinds of primitives.

 - Computational layers: Defining how inputs are propagated forward; `#!scala NeuralLayer`
 - Activation functions: `#!scala ActivationFunction`
 - Computational stacks: composed of a number of layers; `#!scala GenericNeuralStack`

!!! note
    The classes `#!scala NeuralLayer` and `#!scala GenericNeuralStack` define layers and stacks in an abstract manner, meaning that the parameters could be in principle of any type.

    The key point to understand is that once a layer or stack is defined, it is immutable i.e. the parameters defining its forward computation can't be changed.

    The API rather provides _factory_ objects which can spawn a particular layer or stack with any parameter assignments.



## Activation Functions

Activation functions are implemented using the `#!scala Activation[I]` object, its `#!scala apply` method requires two
arguments.

  1. Implementation of the activation
  2. Implementation of the derivative of the activation.

```scala
//Define forward mapping
val actFunc: (I) => I = _
//Define derivative of forward mapping
val gradAct: (I) => I = _

val act = Activation(actFunc, gradAct)
```

The `dynaml.models.neuralnets` package also contains implementation of the following activations.

 - Sigmoid $g(x) = \frac{1}{1 + exp(-x)}$

      `#!scala val act = VectorSigmoid`

 - Tanh $g(x) = tanh(x)$

      `#!scala val act = VectorTansig`

 - Linear $g(x) = x$

      `#!scala val act = VectorLinear`

 - Rectified Linear $g(x) = \begin{cases} x & x \geq 0\\0 & else\end{cases}$

      `#!scala val act = VectorRecLin`


## Computational Layers

Computational layers are the most basic unit of neural networks. They define transformations of their inputs and with that define the forward data flow.

Every computational layer generally has a set of parameters describing how this transformation is going to be calculated given the inputs.

In DynaML, the central component of the `#!scala NeuralLayer[Params, Input, Output]` trait is a `#!scala MetaPipe[Params, Input, Output]` (higher order pipe) instance.

### Creating Layers.

Creating an immutable computational layer can be done using the `#!scala NeuralLayer` object.

```scala
import scala.math._

val compute = MetaPipe(
  (params: Double) => (x: Double) => 2d*Pi*params*x
)

val act = Activation(
  (x: Double) => tanh(x),
  (x: Double) => tanh(x)/(sinh(x)*cosh(x)))

val layer = NeuralLayer(compute, act)(0.5)

```

!!! seealso "Vector feed forward layers"
    A common layer is the feed forward vector to vector layer which is given by.
    $$
    \mathbf{h} = \sigma(\mathbf{W} \mathbf{x} + \mathbf{b})
    $$



### Layer Factories.

Since the computation and activation are the only two relevant inputs required to spawn any computational layer, the `#!scala NeuralLayerFactory[Params, Input, Output]` class is the _factory_ for creating layers on the fly. Layer factories are data pipes which take the layer parameters as input and create computational layers on demand.

A layer factory can be created as follows.

```scala
val fact = NeuralLayerFactory(compute, act)

val layer1 = fact(0.25)
```

!!! seealso "Vector layer factory"
    Vector layers can be created using the `#!scala Vec2VecLayerFactory`

    ```scala
    val layerFactory =
      new Vec2VecLayerFactory(VectorTansig)(inDim = 4, outDim = 5)
    ```  

## Neural Stacks

A neural stack is a sequence of computational layers. Every layer represents some computation, so the neural stack is nothing but a sequence of computations or forward data flow. The top level class for neural stacks is `#!scala GenericNeuralStack`. Extending the base class there are two stack implementations.

  - Eagerly evaluated stack: Layers are spawned as soon as the stack is created.

    ```scala
    val layers: Seq[NeuralLayer[P, I, I]] = _

    //Variable argument apply function
    //so the elements of the sequence
    //must be enumerated.
    val stack = NeuralStack(layers:_*)
    ```

  - Lazy stack: Layers are spawned only as needed, but once created they are [_memoized_](https://en.wikipedia.org/wiki/Memoization).

    ```scala
    val layers_func: (Int) => NeuralLayer[P, I, I] = _

    val stack = LazyNeuralStack(layers_func, num_layers = 4)
    ```


### Stack Factories

Stack factories like layer factories are pipe lines, which take as input a sequence of layer parameters and return a neural stack of the spawned layers.

```scala
val layerFactories: Seq[NeuralLayerFactory[P, I, I]] = _
//Create a stack factory from a sequence of layer factories
val stackFactory = NeuralStackFactory(layerFactories:_*)

//Create a stack factory that creates
//feed forward neural stacks that take as inputs
//breeze vectors.

//Input, Hidden, Output
val num_units_by_layer = Seq(5, 8, 3)
val acts = Seq(VectorSigmoid, VectorTansig)
val breezeStackFactory = NeuralStackFactory(num_units_by_layer)(acts)
```
