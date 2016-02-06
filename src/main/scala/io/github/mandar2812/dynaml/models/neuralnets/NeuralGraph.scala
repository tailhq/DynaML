package io.github.mandar2812.dynaml.models.neuralnets

import breeze.linalg.DenseVector
import com.tinkerpop.blueprints.{GraphFactory, Graph}
import com.tinkerpop.frames.{FramedGraphFactory, FramedGraph}
import io.github.mandar2812.dynaml.graphutils.{Synapse, InputNode}

import scala.collection.JavaConversions._

/**
 * Created by mandar on 8/10/15.
 */
class NeuralGraph(baseGraph: FramedGraph[Graph]){

  protected val g = baseGraph

  def getLayer(layer: Int) = g.getVertices("layer", layer)

  val forwardPassFunc: (DenseVector[Double]) => DenseVector[Double] = (pattern) => {

    pattern
  }

}


object NeuralGraph {
  val manager: FramedGraphFactory = new FramedGraphFactory

  private def apply(num_inputs: Int, num_outputs: Int,
                    hidden_layers: Int = 1,
                    activations: List[String],
                    nCounts:List[Int] = List()): NeuralGraph = {

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
          val inNode = fg.addVertex(classOf[InputNode])
          inNode.setProperty("id", inputnode)
        })
      } else {
        (1 to neuronCounts(layer)).foreach(neuronID => {
          //create input node
          val neuron = fg.addVertex(classOf[InputNode])
          neuron.setProperty("id", neuronID)
          val n = fg.getVertices("layer", layer-1)
          n.foreach(vertex => {
            val synapse = fg.addEdge(
              (layer,vertex.getProperty[Int]("id"), neuron.getProperty[Int]("id")),
              vertex, neuron, "synapse", classOf[Synapse])
            synapse.setWeight(1.0)
          })

        })
      }
    })

    new NeuralGraph(fg)
  }
}