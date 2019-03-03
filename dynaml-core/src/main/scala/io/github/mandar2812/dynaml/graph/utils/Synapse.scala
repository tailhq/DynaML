package io.github.mandar2812.dynaml.graph.utils

import com.tinkerpop.frames.{EdgeFrame, InVertex, OutVertex, Property}

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

  @Property("PrevWeightUpdate")
  def getPrevWeightUpdate(): Double

  @Property("PrevWeightUpdate")
  def setPrevWeightUpdate(w: Double): Unit

  @Property("layer")
  def getLayer(): Int

  @Property("layer")
  def setLayer(l: Int): Unit

}
