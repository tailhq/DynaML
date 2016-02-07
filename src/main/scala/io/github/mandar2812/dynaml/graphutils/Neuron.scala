package io.github.mandar2812.dynaml.graphutils

import com.tinkerpop.blueprints.Direction
import com.tinkerpop.frames.{Incidence, Property, VertexFrame}
import io.github.mandar2812.dynaml.models.neuralnets.TransferFunctions

import scala.collection.JavaConversions

/**
 * Outline of a Neuron
 */
trait Neuron extends VertexFrame {

  @Property("activationFunc")
  def getActivationFunc(): String

  @Property("activationFunc")
  def setActivationFunc(a: String): Unit

  @Property("nID")
  def getNID(): Int

  @Property("nID")
  def setNID(id: Int): Unit

  @Property("layer")
  def getLayer(): Int

  @Property("layer")
  def setLayer(l: Int): Unit

  @Property("value")
  def setValue(v: Double): Unit

  @Property("value")
  def getValue(): Double

  @Property("NeuronType")
  def setNeuronType(v: String): Unit

  @Property("NeuronType")
  def getNeuronType(): String

  @Incidence(label = "synapse", direction = Direction.IN)
  def getIncomingSynapses(): java.lang.Iterable[Synapse]

  @Incidence(label = "synapse", direction = Direction.OUT)
  def getOutgoingSynapses(): java.lang.Iterable[Synapse]

}

object Neuron {

  def getLocalField(neuron: Neuron): Double = neuron.getNeuronType() match {
    case "input" => neuron.getValue()
    case "bias" => 1.0
    case "perceptron" =>
      val connections = JavaConversions.iterableAsScalaIterable(neuron.getIncomingSynapses())
      val activationFunc = TransferFunctions.getActivation(neuron.getActivationFunc())
      activationFunc(connections.map(synapse => {
        synapse.getWeight() * getLocalField(synapse.getPreSynapticNeuron())
      }).sum)
  }
}
