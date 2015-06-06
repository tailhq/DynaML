package org.kuleuven.esat.neuralNetworks

import breeze.linalg.{DenseVector, DenseMatrix}

/**
 * Represents the template of a Feed Forward Neural Network
 * backed by an underlying graph.
 */
abstract class FeedForwardNetwork[G]
  extends NeuralNetwork[G]{
  override protected var params =
    FeedForwardNetwork.initializeWeights(hiddenLayers,
      inputDimensions,
      outputDimensions,
      neuronCounts)

  val feedForward = FeedForwardNetwork.feedForwardFunc(activations)(params)

}

object FeedForwardNetwork {
  def feedForwardFunc(activations: List[(Double) => Double])(params: List[DenseMatrix[Double]]):
  (DenseVector[Double]) => List[(DenseVector[Double], DenseVector[Double])] = {
    def feedForwardInner(x: DenseVector[Double]): List[(DenseVector[Double], DenseVector[Double])] = {
      var partial_activation: DenseVector[Double] = x
      List.range(1, params.length).map((i) => {
        val res: DenseVector[Double] = params(i)*partial_activation
        partial_activation = res.map((j) => activations(i)(j))
        (res, partial_activation)
      })
    }
    feedForwardInner
  }

  def initializeWeights(hiddenLayers: Int,
                        inputDimensions: Int,
                        outputDimensions: Int,
                        neuronCounts: List[Int]) =
    (1 to hiddenLayers + 1).map((i) => {
    if(i == 1){
      DenseMatrix.ones[Double](i, inputDimensions + 1)
    } else if (i == hiddenLayers + 1) {
      DenseMatrix.ones[Double](outputDimensions, i)
    } else {
      DenseMatrix.ones[Double](neuronCounts(i), neuronCounts(i-1))
    }
  }).toList

}