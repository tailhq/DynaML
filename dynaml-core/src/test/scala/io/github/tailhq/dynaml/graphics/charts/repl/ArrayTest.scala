package io.github.tailhq.dynaml.graphics.charts.repl

import io.github.tailhq.dynaml.graphics.charts.highcharts._
import org.scalatest.{Matchers, FunSuite}
import Highchart._


/**
 * User: austin
 * Date: 3/25/15
 */
class ArrayTest extends FunSuite with Matchers {

  test("Array - Single point Highchart to json") {
    val hc = Highchart(Array(Series(Seq(Data(1, 2)))), chart = Chart(zoomType = Zoom.xy), xAxis = None, yAxis = None).toServiceFormat

    hc should be ("highcharts",
      Map(
        "series" -> List(Map("data" -> List(Map("x" -> 1, "y" -> 2)), "type" -> "line")),
        "chart" -> Map("zoomType" -> "xy"),
        "exporting" -> Map("filename" -> "chart"),
        "plotOptions" -> Map(
          "line" -> Map("turboThreshold" -> 0)
        ),
        "credits" -> Map(
          "href" -> "",
          "text" -> ""
        ),
        "title" -> Map("text" -> "")
      )
    )
  }
}
