package io.github.mandar2812.dynaml.graph.utils

import com.tinkerpop.blueprints.Direction
import com.tinkerpop.frames.{Incidence, Property, VertexFrame}
import io.github.mandar2812.dynaml.models.neuralnets.TransferFunctions._
import org.apache.log4j.Logger

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

  @Property("valueBuffer")
  def getValueBuffer(): Array[Double]

  @Property("valueBuffer")
  def setValueBuffer(v: Array[Double]): Unit

  @Property("LocalGradBuffer")
  def getLocalGradBuffer(): Array[Double]

  @Property("LocalGradBuffer")
  def setLocalGradBuffer(v: Array[Double]): Unit

  @Property("LocalFieldBuffer")
  def getLocalFieldBuffer(): Array[Double]

  @Property("LocalFieldBuffer")
  def setLocalFieldBuffer(v: Array[Double]): Unit

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

  private val logger = Logger.getLogger(this.getClass)

  def getLocalField(neuron: Neuron): (Double, Double) = neuron.getNeuronType() match {
    case "input" => (neuron.getValue(), neuron.getValue())
    case "bias" => (1.0, 1.0)
    case "perceptron" =>
      val connections = JavaConversions.iterableAsScalaIterable(neuron.getIncomingSynapses())
      val activationFunc = getActivation(neuron.getActivationFunc())
      val field = connections.map(synapse => {
        synapse.getWeight() * getLocalField(synapse.getPreSynapticNeuron())._1
      }).sum

      (activationFunc(field), field)
  }

  def getLocalFieldBuffer(neuron: Neuron): (Array[Double], Array[Double]) = neuron.getNeuronType() match {
    case "input" => (neuron.getValueBuffer(), neuron.getValueBuffer())
    case "bias" => (neuron.getValueBuffer(), neuron.getValueBuffer())
    case "perceptron" =>
      val connections = JavaConversions.iterableAsScalaIterable(neuron.getIncomingSynapses())
      val activationFunc = getActivation(neuron.getActivationFunc())
      val field = connections.map(synapse => {
        synapse.getPreSynapticNeuron().getLocalFieldBuffer().map(synapse.getWeight()*_)
      }).reduce((c1,c2) => c1.zip(c2).map(c => c._1+c._2))
      (field.map(activationFunc), field)
  }


  def getLocalGradient(neuron: Neuron, hidden: Int): Double = {

    def localGradientRec(n: Neuron, hidden_layers: Int): Double =
      n.getLayer() - hidden_layers match {
        case 1 =>
          val DtransFunc = getDiffActivation(n.getActivationFunc())
          val (localfield, field) = getLocalField(n)
          (n.getValue() - localfield)*DtransFunc(field)

        case _ =>
          val DtransFunc = getDiffActivation(n.getActivationFunc())
          val (_, field) = getLocalField(n)

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

  def getLocalGradientBuffer(neuron: Neuron, hidden: Int,
                             rho: Double = 0.005,
                             sparsityWeight: Double = 0.0): Array[Double] =
    neuron.getLayer() - hidden match {
      case 1 =>
        val DtransFunc = getDiffActivation(neuron.getActivationFunc())
        val (localfield, field) = getLocalFieldBuffer(neuron)
        neuron.getValueBuffer()
          .zip(localfield).map(c => c._1 - c._2)
          .zip(field).map(c => -1.0 * c._1 * DtransFunc(c._2))

      case _ =>
        val DtransFunc = neuron.getNeuronType() match {
          case "perceptron" => getDiffActivation(neuron.getActivationFunc())
          case "bias" => Dlin
        }

        val field = neuron.getValueBuffer()

        val activationFunc = getActivation(neuron.getActivationFunc())

        val rho_neuron = neuron.getActivationFunc() match {
          case SIGMOID => field.map(v => {
            activationFunc(v)
          }).sum/field.length

          case TANH => field.map(v => {
            (activationFunc(v/2.0)+1.0)/2.0
          }).sum/field.length

          case _ => 0.0
        }

        val KL_Term = -1.0*rho/rho_neuron + (1.0 - rho)/(1.0 - rho_neuron)

        val outCon = JavaConversions.iterableAsScalaIterable(neuron.getOutgoingSynapses())

        val sum = outCon.map(synapse =>{
          val weight = synapse.getWeight()
          val postN = synapse.getPostSynapticNeuron()
          postN.getLocalGradBuffer().map(_*weight)
        }).reduce((c1,c2) => c1.zip(c2).map(c => c._1+c._2))

        sum.zip(field).map(c => (c._1 + sparsityWeight*KL_Term)*DtransFunc(c._2))
      }



}