package io.github.tailhq.dynaml.graph.utils

import com.tinkerpop.blueprints.Direction
import com.tinkerpop.frames.{Adjacency, Property, VertexFrame}

/**
 * Defines the [[VertexFrame]] for the
 * input features nodes in the graph.
 */
trait Point extends VertexFrame {
  @Property("value")
  def getValue(): Array[Double]

  @Property("value")
  def setValue(value: Array[Double]): Unit

  @Property("featureMap")
  def getFeatureMap(): Array[Double]

  @Property("featureMap")
  def setFeatureMap(value: Array[Double]): Unit

  @Adjacency(label = "causes", direction = Direction.OUT)
  def getLabel(): java.lang.Iterable[Label]
}
