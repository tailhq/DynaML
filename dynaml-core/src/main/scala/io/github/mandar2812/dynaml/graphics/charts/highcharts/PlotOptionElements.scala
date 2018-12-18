package io.github.mandar2812.dynaml.graphics.charts.highcharts

/**
 * User: austin
 * Date: 12/12/14
 */
object Stacking {
  type Type = String
  val (normal, percent) = ("normal", "percent")
  def values = Set(normal, percent)
}
