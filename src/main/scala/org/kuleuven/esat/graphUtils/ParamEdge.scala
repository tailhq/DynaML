package org.kuleuven.esat.graphUtils

import com.tinkerpop.frames.{InVertex, OutVertex, EdgeFrame}

/**
 * Defines the [[EdgeFrame]] for the
 * edges going out from the parameter
 * node.
 */
trait ParamEdge[T] extends EdgeFrame {
  @OutVertex
  def getParameter(): Parameter[T]

  @InVertex
  def getLabel(): Label[T]
}
