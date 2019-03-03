package io.github.mandar2812.dynaml.graph.utils

import com.tinkerpop.frames.{EdgeFrame, InVertex, OutVertex}

/**
 * Defines the [[EdgeFrame]] for the
 * edges going out from the parameter
 * node.
 */
trait ParamEdge extends EdgeFrame {
  @OutVertex
  def getParameter(): Parameter

  @InVertex
  def getLabel(): Label
}
