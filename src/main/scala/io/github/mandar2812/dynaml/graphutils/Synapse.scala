package io.github.mandar2812.dynaml.graphutils

import com.tinkerpop.frames.{Property, InVertex, OutVertex, EdgeFrame}

/**
 * Frames implementation of a Neural Network
 * Synapse
 */
trait Synapse extends EdgeFrame {
  @OutVertex
  def getPreSynapticNeuron(): Neuron

  @InVertex
  def getPostSynapticNeuron(): Neuron

  @Property("weight")
  def getWeight(): Double

  @Property("weight")
  def setWeight(w: Double): Unit

}
