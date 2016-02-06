package io.github.mandar2812.dynaml.graphutils

import com.tinkerpop.frames.{VertexFrame, Property}

/**
 * Input nodes of a Neural Network
 */
trait InputNode extends VertexFrame {
  @Property("value")
  def setValue(v: Double): Unit

  @Property("value")
  def getValue(): Double

  @Property("layer")
  def getLayer(): Int = 1

  @Property("id")
  def getId(): Int

}
