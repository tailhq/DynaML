package org.kuleuven.esat.graphUtils

import com.tinkerpop.blueprints.Direction
import com.tinkerpop.frames.{Adjacency, VertexFrame, Property}

/**
 * Defines the [[VertexFrame]] for the
 * parameter node in the graph.
 */
trait Parameter[T] extends VertexFrame {
  @Property("slope")
  def getSlope(): T

  @Property("slope")
  def setSlope(slope: T): Unit

  @Adjacency(label = "controls", direction = Direction.OUT)
  def getControlledPointLabels(): java.lang.Iterable[Label[T]]
}
