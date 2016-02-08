package io.github.mandar2812.dynaml.models.neuralnets

import breeze.linalg.DenseVector
import com.tinkerpop.blueprints.{GraphFactory, Graph}
import com.tinkerpop.frames.{FramedGraphFactory, FramedGraph}
import io.github.mandar2812.dynaml.graphutils.{Neuron, Synapse}

import scala.collection.JavaConversions
import scala.collection.JavaConversions._
import scala.util.Random

/**
 * Represents the underlying graph of a neural
  * model.
 */
class FFNeuralGraph(baseGraph: FramedGraph[Graph],
                    hidden: Int = 1,
                    act: List[String])
  extends NeuralGraph[FramedGraph[Graph]]{

  override protected val g = baseGraph

  val hidden_layers = hidden

  val activations = act

  def getLayer(layer: Int) = JavaConversions.iterableAsScalaIterable(
    g.getVertices[Neuron]("layer", layer, classOf[Neuron])
  )

  def getLayerSynapses(layer: Int) = JavaConversions.iterableAsScalaIterable(
    g.getEdges[Synapse]("layer", layer, classOf[Synapse])
  )

  override val num_inputs: Int = getLayer(0).size - 1

  override val num_outputs: Int = getLayer(hidden_layers+1).size

  override val forwardPass: (DenseVector[Double]) => DenseVector[Double] = (pattern) => {
    //Set the pattern as input to layer 0
    val inputnodes = getLayer(0) filter (_.getNeuronType() == "input")

    inputnodes.foreach(node => {
      val id = node.getNID()
      node.setValue(pattern(id-1))
    })

    //println("Output Neurons: "+getLayer(hidden_layers+1).toString())
    val outputs:Map[Int, Double] = getLayer(hidden_layers+1)
      .map(outputNeuron => (outputNeuron.getNID(), Neuron.getLocalField(outputNeuron)._1))
      .toMap

    DenseVector.tabulate[Double](num_outputs)(i => outputs(i+1))
  }

}


object FFNeuralGraph {
  val manager: FramedGraphFactory = new FramedGraphFactory

  def apply(num_inputs: Int, num_outputs: Int,
            hidden_layers: Int = 1,
            activations: List[String],
            nCounts:List[Int] = List()): FFNeuralGraph = {

    val neuronCounts:List[Int] = if(nCounts.isEmpty)
      List.tabulate[Int](hidden_layers+1)(i => {
        if(i <= hidden_layers) 3 else num_outputs
      })
    else nCounts

    val graphconfig = Map("blueprints.graph" ->
      "com.tinkerpop.blueprints.impls.tg.TinkerGraph")

    val fg = manager.create(GraphFactory.open(mapAsJavaMap(graphconfig)))

    (0 to hidden_layers+1).foreach((layer) => {
      //For each layer create neurons
      if(layer == 0) {
        (1 to num_inputs).foreach(inputnode => {
          //create input node
          val inNode: Neuron = fg.addVertex((0,inputnode), classOf[Neuron])
          inNode.setLayer(0)
          inNode.setNID(inputnode)
          inNode.setNeuronType("input")
        })
        //Create Bias unit
        val biasInput: Neuron = fg.addVertex((0,num_inputs+1), classOf[Neuron])
        biasInput.setLayer(0)
        biasInput.setNeuronType("bias")
        biasInput.setNID(num_inputs+1)

      } else {
        val num_neurons = if(layer == hidden_layers+1) num_outputs else neuronCounts(layer-1)
        (1 to num_neurons).foreach(neuronID => {

          //create neuron
          val neuron: Neuron = fg.addVertex((layer, neuronID), classOf[Neuron])
          neuron.setLayer(layer)
          neuron.setNID(neuronID)
          neuron.setActivationFunc(activations(layer-1))
          neuron.setNeuronType("perceptron")

          //Wire incoming synapses
          val n = fg.getVertices[Neuron]("layer", layer-1, classOf[Neuron])
          n.foreach(vertex => {
            val synapse: Synapse =
              fg.addEdge((layer, vertex.getNID(), neuron.getNID()),
                vertex.asVertex(), neuron.asVertex(), "synapse", classOf[Synapse])
            synapse.setLayer(layer)
            synapse.setWeight(Random.nextDouble())
            synapse.setPrevWeightUpdate(0.0)
          })
        })

        //Create Bias unit for layer if it is not an output layer
        if(layer < hidden_layers+1) {
          val biasLayerL: Neuron = fg.addVertex((layer, num_neurons+1), classOf[Neuron])
          biasLayerL.setLayer(layer)
          biasLayerL.setNeuronType("bias")
          biasLayerL.setNID(num_neurons+1)
        }
      }
    })

    new FFNeuralGraph(fg, hidden_layers, activations)
  }
}