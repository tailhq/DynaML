package org.kuleuven.esat.neuralNetworks

import breeze.linalg.{DenseVector, DenseMatrix}

/**
 * Represents the template of a Feed Forward Neural Network
 * backed by an underlying graph.
 */
abstract class FeedForwardNetwork[G]
  extends NeuralNetwork[G]{
  override protected val params =
    (1 to hiddenLayers + 1).map((i) => {
    if(i == 1){
      DenseMatrix.ones[Double](i, this.inputDimensions)
    } else if (i == hiddenLayers + 1) {
      DenseMatrix.ones[Double](this.outputDimensions, i)
    } else {
      DenseMatrix.ones[Double](neuronCounts(i), neuronCounts(i-1))
    }
  }).toList

  val feedForward = FeedForwardNetwork.feedForwardFunc(activations)(params)

}

object FeedForwardNetwork {
  def feedForwardFunc(activations: List[(Double) => Double])(params: List[DenseMatrix[Double]]):
  (DenseVector[Double]) => List[DenseVector[Double]] = {
    def feedForwardInner(x: DenseVector[Double]): List[DenseVector[Double]] = {
      var partial_activation: DenseVector[Double] = x
      List.range(1, params.length).map((i) => {
        val res: DenseVector[Double] = params(i)*partial_activation
        partial_activation = res.map((j) => activations(i)(j))
        partial_activation
      })
    }
    feedForwardInner
  }
}