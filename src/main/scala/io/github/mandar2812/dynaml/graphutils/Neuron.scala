package io.github.mandar2812.dynaml.graphutils

import com.tinkerpop.blueprints.Direction
import com.tinkerpop.frames.{Incidence, Property, VertexFrame}

/**
 * Outline of a Neuron
 */
trait Neuron extends VertexFrame {
  @Property("localfield")
  def getLocalField(): Double

  @Property("activationFunc")
  def getActivationFunc(): String

  @Property("layer")
  def getLayer(): Int

  @Property("layer")
  def setLayer(l: Int): Unit

  @Incidence(label = "synapse", direction = Direction.IN)
  def getIncomingSynapses(): java.lang.Iterable[Synapse]

  @Incidence(label = "synapse", direction = Direction.OUT)
  def getOutgoingSynapses(): java.lang.Iterable[Synapse]

}
