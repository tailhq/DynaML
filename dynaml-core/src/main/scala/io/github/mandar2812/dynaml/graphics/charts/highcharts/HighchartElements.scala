package io.github.mandar2812.dynaml.graphics.charts.highcharts

import scala.collection.mutable
import scala.language.implicitConversions

/**
 * User: austin
 * Date: 9/10/13
 */

object Color {
  type Type = String
  implicit def javaColorToHex(color: java.awt.Color): Type = "#"+Integer.toHexString(color.getRGB()).substring(2)
  implicit def javaColorToHexOption(color: java.awt.Color): Option[Type] = Some("#"+Integer.toHexString(color.getRGB()).substring(2))

}

object Alignment {
  type Type = String
  val (left, center, right) = ("left", "center", "right")
  def values = mutable.LinkedHashSet() ++ Seq(center, left, right)
}

object VerticalAlignment {
  type Type = String
  val (top, middle, bottom) = ("top", "middle", "bottom")
  def values = Set(top, middle, bottom)
}

object AxisAlignment {
  type Type = String
  val (middle, low, high) = ("middle", "low", "high")
  def values = Set(middle, low, high)
}

object Layout {
  type Type = String
  val (horizontal, vertical) = ("horizontal", "vertical")
  def values = mutable.LinkedHashSet() ++ Seq(horizontal, vertical)
}

object AxisType {
  type Type = String
  val (category, datetime, linear, logarithmic) = ("category", "datetime", "linear", "logarithmic")
  def values = Set(category, datetime, linear, logarithmic)
}

object SeriesType {
  type Type = String
  val (area, areaspline, bar, boxplot, column, line, pie, scatter, spline) = ("area", "areaspline", "bar", "boxplot", "column", "line", "pie", "scatter", "spline")
  def values = mutable.LinkedHashSet() ++ Seq(area, areaspline, bar, column, line, pie, scatter, spline)
}

object Overflow {
  type Type = String
  val justify = "justify"
  def values = Set(justify)
}

object Zoom {
  type Type = String
  val (x, y, xy) = ("x", "y", "xy")
  def values = Set(x, y, xy)
}

object FontWeight {
  type Type = String
  val (normal, lighter, bold, bolder, inherit) = ("normal", "lighter", "bold", "bolder", "inherit")
  def values = Set(normal, lighter, bold, bolder, inherit)
}

// WIP
case class CSSObject(
                      color: Option[Color.Type] = None,
                      fontWeight: Option[FontWeight.Type] = None
                      ) {
  override def toString() = {
    Seq(
      ("color", color),
      ("fontWeight", fontWeight)
    ).collect{case(name, Some(v)) => "%s: '%s'".format(name, v)}.mkString(",\n")
  }
}
