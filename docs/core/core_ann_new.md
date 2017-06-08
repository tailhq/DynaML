!!! summary
    In v1.4.2, the neural stack API was introduced. It defines some primitives for modular
    construction of neural networks. The main idioms will be computational layers & stacks.  


The new neural network API extends the same top level traits as the old API, i.e.
`#!scala NeuralNet[Data, BaseGraph, Input, Output,Graph <: NeuralGraph[BaseGraph, Input, Output]]` which itself extends
the `#!scala ParameterizedLearner[Data, Graph, Input, Output, Stream[(Input, Output)]]` trait.

!!! tip
    To learn more about `#!scala ParameterizedLearner` and other major model classes, refer to the [model hierarchy](/core/core_model_hierarchy.md) specification.

In the case of `#!scala NeuralNet`, the _parameters_ are of a generic (unknown) type `#!scala Graph` which has to be an extension of `#!scala NeuralGraph[BaseGraph, Input, Output]]` trait.

The neural stack API extends these abstract skeletons by defining two kinds of primitives.

 - Computational layers: Defining how inputs are propagated forward; `#!scala NeuralLayer`
 - Activation functions: `#!scala ActivationFunction`
 - Computational stacks: composed of a number of layers; `#!scala GenericNeuralStack`

!!! note
    The classes `#!scala NeuralLayer` and `#!scala GenericNeuralStack` define layers and stacks in an abstract manner, meaning that the parameters could be in principle of any type.

    The key point to understand is that once a layer or stack is defined, it is immutable i.e. the parameters defining its forward computation can't be changed.

    The API rather provides _factory_ objects which can spawn a particular layer or stack with any parameter assignments.

## Computational Layers

Computational layers are the most basic unit of neural networks. They define transformations of their inputs and with that define the forward data flow.

Every computational layer generally has a set of parameters describing how this transformation is going to be calculated given the inputs.

In DynaML, the central component of the `#!scala NeuralLayer[Params, Input, Output]` trait is a `#!scala MetaPipe[Params, Input, Output]` (higher order pipe) instance.

### Creating Layers.

Creating an immutable computational layer can be done using the `#!scala NeuralLayer` object.

```scala
import scala.math._

val compute = MetaPipe(
  (params: Double) => (x: Double) => 2d*Pi*params*x)

val act = Activation(
  (x: Double) => tanh(x),
  (x: Double) => tanh(x)/(sinh(x)*cosh(x)))

val layer = NeuralLayer(compute, act)(0.5)

```

### Layer Factories.

```scala
val fact = NeuralLayerFactory(compute, act)

val layer1 = fact(0.25)
```

## Activation Functions

```scala
//Define forward mapping
val actFunc: (I) => I = _
//Define derivative of forward mapping
val gradAct: (I) => I = _

val act = Activation(actFunc, gradAct)
```

## Neural Stacks

### Stack Factories
