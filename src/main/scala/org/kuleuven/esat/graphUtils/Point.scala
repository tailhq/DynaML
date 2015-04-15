package org.kuleuven.esat.graphUtils

import com.tinkerpop.blueprints.Direction
import com.tinkerpop.frames.{Adjacency, VertexFrame, Property}

/**
 * Defines the [[VertexFrame]] for the
 * input features nodes in the graph.
 */
trait Point extends VertexFrame {
  @Property("value")
  def getValue(): Array[Byte]

  @Property("value")
  def setValue(value: Array[Byte]): Unit

  @Property("featureMap")
  def getFeatureMap(): Array[Byte]

  @Property("featureMap")
  def setFeatureMap(value: Array[Byte]): Unit

  @Adjacency(label = "causes", direction = Direction.OUT)
  def getLabel(): java.lang.Iterable[Label]
}
