package org.kuleuven.esat.graphUtils

import com.tinkerpop.blueprints.Direction
import com.tinkerpop.frames.{Adjacency, VertexFrame, Property}

/**
 * Defines the [[VertexFrame]] for the
 * input features nodes in the graph.
 */
trait Point[T] extends VertexFrame {
  @Property("value")
  def getValue(): T

  @Property("value")
  def setValue(value: T): Unit

  @Property("featureMap")
  def getFeatureMap(): T

  @Property("featureMap")
  def setFeatureMap(value: T): Unit

  @Adjacency(label = "causes", direction = Direction.OUT)
  def getLabel(): java.lang.Iterable[Label[T]]
}
