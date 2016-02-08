package io.github.mandar2812.dynaml.models.neuralnets

import breeze.linalg.DenseVector
import com.tinkerpop.blueprints.Graph
import com.tinkerpop.frames.FramedGraph
import io.github.mandar2812.dynaml.optimization.BackPropogation
import io.github.mandar2812.dynaml.pipes.DataPipe

/**
 * Represents the template of a Feed Forward Neural Network
 * backed by an underlying graph.
 */
class FeedForwardNetwork[D](
  data: D, netgraph: FFNeuralGraph,
  transform: DataPipe[D, Stream[(DenseVector[Double], DenseVector[Double])]])
  extends NeuralNetwork[D, FramedGraph[Graph], FFNeuralGraph,
    (DenseVector[Double], DenseVector[Double])]{

  override protected val g = data

  val num_points:Int = dataAsStream(g).length

  override protected var params: FFNeuralGraph = netgraph

  val feedForward = params.forwardPass

  /**
   * Get the value of the parameters
   * of the model.
   **/
  override val outputDimensions: Int = params.num_outputs

  override val hiddenLayers: Int = params.hidden_layers

  override val activations: List[(Double) => Double] =
    params.activations.map(TransferFunctions.getActivation)

  override val neuronCounts: List[Int] =
    List.tabulate[Int](hiddenLayers)(i => params.getLayer(i+1).size)
  override val inputDimensions: Int = params.num_inputs

  override def initParams(): FFNeuralGraph = FFNeuralGraph(
    inputDimensions, outputDimensions,
    hiddenLayers, params.activations,
    neuronCounts)

  override protected val optimizer =
    new BackPropogation()
      .setNumIterations(100)
      .setStepSize(0.01)

  override def dataAsStream(d: D) = transform.run(d)

  /**
   * Learn the parameters
   * of the model which
   * are in a node of the
   * graph.
   *
   **/
  override def learn(): Unit = {
    params = optimizer.optimize(num_points, dataAsStream(g), initParams())
  }

  def test(d: D): Stream[(DenseVector[Double], DenseVector[Double])] = {
    dataAsStream(d).map(rec => (feedForward(rec._1), rec._2))
  }
}
