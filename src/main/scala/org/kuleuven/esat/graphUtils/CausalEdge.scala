package org.kuleuven.esat.graphUtils

import com.tinkerpop.frames.{Property, OutVertex, InVertex, EdgeFrame}

/**
 * Defines the [[EdgeFrame]] for the
 * edges going out from the input data
 * node.
 */
trait CausalEdge[T] extends EdgeFrame {
  @OutVertex
  def getPoint(): Point[T]

  @InVertex
  def getLabel(): Label[T]

  @Property("relation")
  def getRelation(): String

  @Property("relation")
  def setRelation(value: String): Unit
}
