package io.github.mandar2812.dynaml.models.neuralnets

import breeze.linalg.DenseVector
import com.tinkerpop.frames.FramedGraphFactory

/**
 * Represents the template of a Feed Forward Neural Network
 * backed by an underlying graph.
 */
abstract class BasicFeedForwardNetwork[D](netgraph: NeuralGraph)
  extends NeuralNetwork[D, NeuralGraph]{
  this.params = netgraph
  val feedForward = BasicFeedForwardNetwork.feedForwardFunc(params) _

  /**
   * Get the value of the parameters
   * of the model.
   **/
  override val outputDimensions: Int
  override val hiddenLayers: Int
  override val activations: List[(Double) => Double]
  override val neuronCounts: List[Int]
  override val inputDimensions: Int

  override def initParams(): NeuralGraph

  /**
   * Learn the parameters
   * of the model which
   * are in a node of the
   * graph.
   *
   **/
  override def learn(): Unit = {}
}

object BasicFeedForwardNetwork {

  val manager: FramedGraphFactory = new FramedGraphFactory

  def apply[G](num_inputs: Int, num_outputs: Int,
               hidden_layers: Int,
               activations: List[String],
               neuronCounts:List[Int] = List()) = {


  }

  def feedForwardFunc(networkGraph: NeuralGraph)
                     (inputPattern: DenseVector[Double]): Unit = {}

  def initializeWeights(hiddenLayers: Int,
                        inputDimensions: Int,
                        outputDimensions: Int,
                        neuronCounts: List[Int]) = {}

}