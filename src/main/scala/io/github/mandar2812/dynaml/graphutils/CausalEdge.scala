package io.github.mandar2812.dynaml.graphutils

import com.tinkerpop.frames.{Property, OutVertex, InVertex, EdgeFrame}

/**
 * Defines the [[EdgeFrame]] for the
 * edges going out from the input data
 * node.
 */
trait CausalEdge extends EdgeFrame {
  @OutVertex
  def getPoint(): Point

  @InVertex
  def getLabel(): Label

  @Property("relation")
  def getRelation(): String

  @Property("relation")
  def setRelation(value: String): Unit
}
