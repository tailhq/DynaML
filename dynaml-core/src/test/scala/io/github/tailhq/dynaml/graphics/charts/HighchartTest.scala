package io.github.tailhq.dynaml.graphics.charts

import org.scalatest.Matchers
import org.scalatest.FunSuite
import io.github.tailhq.dynaml.graphics.charts.highcharts._
import Highchart._

/**
 * User: austin
 * Date: 10/4/13
 */
class HighchartTest extends FunSuite with Matchers {

  test("Single point Highchart to json") {
    val hc = Highchart(Seq(Series(Seq(Data(1, 2)))), chart = Chart(zoomType = Zoom.xy), xAxis = None, yAxis = None).toServiceFormat

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
