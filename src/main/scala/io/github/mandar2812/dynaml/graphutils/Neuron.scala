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

  def getLocalField(neuron: Neuron): (Double, Double) = neuron.getNeuronType() match {
    case "input" => (neuron.getValue(), neuron.getValue())
    case "bias" => (1.0, 1.0)
    case "perceptron" =>
      val connections = JavaConversions.iterableAsScalaIterable(neuron.getIncomingSynapses())
      val activationFunc = TransferFunctions.getActivation(neuron.getActivationFunc())
      val field = connections.map(synapse => {
        synapse.getWeight() * getLocalField(synapse.getPreSynapticNeuron())._1
      }).sum

      (activationFunc(field), field)
  }

  def getLocalGradient(neuron: Neuron, hidden: Int): Double = {

    def localGradientRec(n: Neuron, hidden_layers: Int): Double =
      n.getLayer() - hidden_layers match {
        case 1 =>
          val DtransFunc = TransferFunctions.getDiffActivation(n.getActivationFunc())
          val (localfield, field) = Neuron.getLocalField(n)
          (n.getValue() - localfield)*DtransFunc(field)

        case _ =>
          val DtransFunc = TransferFunctions.getDiffActivation(n.getActivationFunc())
          val (_, field) = Neuron.getLocalField(n)

          val outCon = JavaConversions.iterableAsScalaIterable(n.getOutgoingSynapses())
          val sum = outCon.map(synapse =>{
            val weight = synapse.getWeight()
            val preN = synapse.getPostSynapticNeuron()
            weight*localGradientRec(preN, hidden_layers)
          }).sum

          sum*DtransFunc(field)
      }

    localGradientRec(neuron, hidden)
  }
}
