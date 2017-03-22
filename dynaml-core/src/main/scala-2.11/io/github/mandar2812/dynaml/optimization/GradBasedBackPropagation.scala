package io.github.mandar2812.dynaml.optimization

import io.github.mandar2812.dynaml.models.neuralnets.NeuralStack

/**
  * Created by mandar on 23/03/2017.
  */
abstract class GradBasedBackPropagation[LayerP, I] extends
  RegularizedOptimizer[NeuralStack[LayerP, I], I, I, Stream[(I, I)]] {

}
