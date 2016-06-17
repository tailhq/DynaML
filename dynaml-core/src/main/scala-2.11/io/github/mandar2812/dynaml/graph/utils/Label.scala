package io.github.mandar2812.dynaml.graph.utils

import com.tinkerpop.blueprints.Direction
import com.tinkerpop.frames.{Adjacency, Property, VertexFrame}

/**
 * Defines the [[VertexFrame]] for the
 * target label node in the graph.
 */
trait Label extends VertexFrame{
  @Property("value")
  def getValue(): Double

  @Property("value")
  def setValue(value: Double): Unit

  @Adjacency(label = "causes", direction = Direction.IN)
  def getCausedByPoint(): java.lang.Iterable[Point]
}
