package io.github.mandar2812.dynaml.graphutils

import com.tinkerpop.blueprints.Direction
import com.tinkerpop.frames.{Adjacency, VertexFrame, Property}

/**
 * Defines the [[VertexFrame]] for the
 * parameter node in the graph.
 */
trait Parameter extends VertexFrame {
  @Property("slope")
  def getSlope(): Array[Double]

  @Property("slope")
  def setSlope(slope: Array[Double]): Unit

  @Adjacency(label = "controls", direction = Direction.OUT)
  def getControlledPointLabels(): java.lang.Iterable[Label]
}
