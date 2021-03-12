/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
* */
package io.github.tailhq.dynaml.graph

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Uniform
import com.tinkerpop.blueprints.{Graph, GraphFactory}
import com.tinkerpop.frames.{FramedGraph, FramedGraphFactory}
import io.github.tailhq.dynaml.graph.utils.{Neuron, Synapse}
import org.apache.log4j.Logger

import scala.collection.JavaConversions
import scala.collection.JavaConversions._

/**
  * Represents the underlying graph of a
  * feed-forward neural network.
  *
  * @param baseGraph The base graph object, [[FFNeuralGraph]] encapsulates
  *                  an existing graph object of type [[FramedGraph]] and builds
  *                  upon it by defining a set of behaviours expected from Neural Network
  *                  graphs
  *
  * @param hidden The number of hidden layers in the network.
  *
  * @param act A list of Strings containing the activations for each layer.
  *            Options for activation functions are:
  *            1) "logsig" or "sigmoid"
  *            2) "tansig"
  *            3) "linear"
  *            4) "recLinear"
  * */
class FFNeuralGraph(baseGraph: FramedGraph[Graph], act: List[String], hidden: Int = 1)
  extends NeuralGraph[FramedGraph[Graph], DenseVector[Double], DenseVector[Double]]{

  override protected val g = baseGraph

  val hidden_layers = hidden

  val activations = act

  /**
    * Get as a scala [[Iterable]] the neurons for a particular layer.
    *
    * @param layer The layer number, can vary from 0 (input layer)
    *              to hidden_layers + 1 (output layer)
    *
    * @return The neurons in the particular layer as [[Neuron]] objects
    *
    * */
  def getLayer(layer: Int) = JavaConversions.iterableAsScalaIterable(
    g.getVertices[Neuron]("layer", layer, classOf[Neuron])
  )

  /**
    * Get as a scala [[Iterable]] the synapses between layer l and l-1.
    *
    * @param layer The layer number, can vary from 1 (input layer synapses)
    *              to hidden_layers + 1 (output layer synapses)
    *
    * @return The respective synapses as [[Synapse]] objects
    *
    * */
  def getLayerSynapses(layer: Int) = JavaConversions.iterableAsScalaIterable(
    g.getEdges[Synapse]("layer", layer, classOf[Synapse])
  )

  val num_inputs: Int = getLayer(0).size - 1

  val num_outputs: Int = getLayer(hidden_layers+1).size

  /**
    * Perform a forward pass through the network to
    * calculate the predicted output.
    * */
  override val forwardPass: (DenseVector[Double]) => DenseVector[Double] = (pattern) => {
    //Set the pattern as input to layer 0
    val inputnodes = getLayer(0) filter (_.getNeuronType() == "input")

    inputnodes.foreach(node => {
      val id = node.getNID()
      node.setValue(pattern(id-1))
    })

    val outputs:Map[Int, Double] = getLayer(hidden_layers+1)
      .map(outputNeuron => (outputNeuron.getNID(), Neuron.getLocalField(outputNeuron)._1))
      .toMap

    DenseVector.tabulate[Double](num_outputs)(i => outputs(i+1))
  }

  /**
    * Get as a breeze [[DenseMatrix]] the synapses between layer l and l-1.
    *
    * @param layer The layer number, can vary from 1 (input layer synapses)
    *              to hidden_layers + 1 (output layer synapses)
    *
    * @return The respective synapses as elements of a matrix
    *
    * */
  def getSynapsesAsMatrix(layer: Int): DenseMatrix[Double] = {
    val synapses = getLayerSynapses(layer)

    val inN = getLayer(layer-1).toList.length
    val outN =
      if(layer <= hidden_layers) getLayer(layer).toList.length-1
      else getLayer(layer).toList.length

    val synapsesMap: Map[(Int, Int), Double] =
      synapses.map(s => (
        (s.getPostSynapticNeuron().getNID(),
          s.getPreSynapticNeuron().getNID()),
        s.getWeight())).toMap

    DenseMatrix.tabulate[Double](outN, inN)((i,j) => synapsesMap((i+1, j+1)))
  }

  /**
    * Perform a forward pass through the network to
    * calculate the predicted output for a batch of
    * input points.
    *
    * @param procInputs The input batch as a List of Lists
    *                   where each level of the top level List
    *                   represents an input node. On the other hand
    *                   each element of the lower level list represents
    *                   a particular dimension of a particular data point
    *                   in the data set.
    * */
  def predictBatch(procInputs: List[List[Double]]) = {

    getLayer(0).foreach(node => node.getNeuronType() match {
      case "input" =>
        node.setValueBuffer(procInputs(node.getNID() - 1).toArray)
        node.setLocalFieldBuffer(procInputs(node.getNID() - 1).toArray)
      case "bias" =>
        node.setValueBuffer(Array.fill[Double](procInputs.head.length)(1.0))
        node.setLocalFieldBuffer(Array.fill[Double](procInputs.head.length)(1.0))
    })

    (1 to hidden_layers).foreach(layer => {
      getLayer(layer).foreach(node => node.getNeuronType() match {
        case "perceptron" =>
          val (locfield, field) = Neuron.getLocalFieldBuffer(node)
          node.setLocalFieldBuffer(locfield)
          node.setValueBuffer(field)
        case "bias" =>
          node.setValueBuffer(Array.fill[Double](procInputs.head.length)(1.0))
          node.setLocalFieldBuffer(Array.fill[Double](procInputs.head.length)(1.0))
      })
    })

    getLayer(hidden_layers+1)
      .map(node => (node.getNID()-1, Neuron.getLocalFieldBuffer(node)._1.zipWithIndex.map(_.swap).toMap))
      .toMap
  }
}


object FFNeuralGraph {
  val manager: FramedGraphFactory = new FramedGraphFactory

  private val logger = Logger.getLogger(this.getClass)

  /**
    * Create a [[FFNeuralGraph]] object with
    * [[FramedGraph]] as the base graph.
    *
    * @param num_inputs Number of input dimensions
    * @param num_outputs Number of input dimensions
    * @param hidden_layers Number of hidden layers
    * @param nCounts The number of neurons in each hidden layer
    * @param activations The activation functions for each layer
    * @param biasFlag Indicates if bias unit is to be created.
    *
    * */
  def apply(num_inputs: Int, num_outputs: Int,
            hidden_layers: Int = 1,
            activations: List[String],
            nCounts:List[Int] = List(),
            biasFlag: Boolean = true): FFNeuralGraph = {

    val uni = new Uniform(-1.0, 1.0)

    val neuronCounts:List[Int] = if(nCounts.isEmpty)
      List.tabulate[Int](hidden_layers+1)(i => {
        if(i <= hidden_layers) 3 else num_outputs
      })
    else nCounts

    val graphconfig = Map("blueprints.graph" ->
      "com.tinkerpop.blueprints.impls.tg.TinkerGraph")

    val fg = manager.create(GraphFactory.open(mapAsJavaMap(graphconfig)))

    (0 to hidden_layers+1).foreach((layer) => {
      logger.info("Initializing layer "+layer)
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
        if(biasFlag) {
          val biasInput: Neuron = fg.addVertex((0,num_inputs+1), classOf[Neuron])
          biasInput.setLayer(0)
          biasInput.setNeuronType("bias")
          biasInput.setNID(num_inputs+1)
        }

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
            synapse.setWeight(uni.draw)
            synapse.setPrevWeightUpdate(0.0)
          })
        })

        //Create Bias unit for layer if it is not an output layer
        if(layer < hidden_layers+1 && biasFlag) {
          val biasLayerL: Neuron = fg.addVertex((layer, num_neurons+1), classOf[Neuron])
          biasLayerL.setLayer(layer)
          biasLayerL.setNeuronType("bias")
          biasLayerL.setNID(num_neurons+1)
        }
      }
    })

    new FFNeuralGraph(fg, activations, hidden_layers)
  }
}