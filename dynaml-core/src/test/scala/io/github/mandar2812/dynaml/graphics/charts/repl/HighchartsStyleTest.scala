package io.github.mandar2812.dynaml.graphics.charts.repl

import io.github.mandar2812.dynaml.graphics.charts.highcharts._
import org.scalatest.{Matchers, FunSuite}
import io.github.mandar2812.dynaml.graphics.charts.Highcharts._

/**
 * User: austin
 * Date: 3/25/15
 */
class HighchartsStyleTest extends FunSuite with Matchers {

  test("legend") {
    val hc = Highchart(Seq(Series(Seq(Data(1, 2))), Series(Seq(Data(2, 3)))), xAxis = None, yAxis = None)
    val styled_hc = legend(hc, List("Legend"))
    styled_hc.series.map(_.name) should be(Seq(Some("Legend"), None))
  }

  test("stacking") {
    val hc = Highchart(Seq(Series(Seq(Data(1, 2))), Series(Seq(Data(2, 3)))), xAxis = None, yAxis = None)
    val stacked_hc = stack(hc, Stacking.normal)
    stacked_hc.plotOptions.head.series.head.stacking should be(Some(Stacking.normal))
    val unstacked_hc = unstack(stacked_hc)
    unstacked_hc.plotOptions.head.series.head.stacking should be(None)
  }

  test("x-axis") {
    val hc = Highchart(Seq(Series(Seq(Data(1, 2))), Series(Seq(Data(2, 3)))), xAxis = None, yAxis = None)
    val axistype_hc = xAxisType(hc, AxisType.logarithmic)
    axistype_hc should not be (None)
    val axiscategories_hc = xAxisCategories(axistype_hc, List("First", "Second"))
    axiscategories_hc should not be (None)
    val axistitle_hc = xAxis(axiscategories_hc, "Label")

    println("=============" + axistitle_hc.toJson + "=============")

    axistitle_hc.xAxis.get.apply(0).title should be(Some(AxisTitle("Label")))
    axistitle_hc.xAxis.get.apply(0).axisType should be(Some(AxisType.category))
    axistitle_hc.xAxis.get.apply(0).categories.get.mkString(",") should be("First,Second")
  }
}

