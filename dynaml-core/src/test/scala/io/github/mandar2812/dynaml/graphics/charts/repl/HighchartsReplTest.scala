package io.github.mandar2812.dynaml.graphics.charts.repl

import io.github.mandar2812.dynaml.graphics.charts.Highcharts._
import org.scalatest.Matchers
import org.scalatest.FunSuite

/**
 * User: austin
 * Date: 12/15/14
 */
class HighchartsReplTest extends FunSuite with Matchers  {

  test("Pie repl") {
    disableOpenWindow // prevents server from starting
    pie(1 to 4).toJson should be(
        """{"series":[""" +
        """{"data":[{"x":0,"y":1},{"x":1,"y":2},{"x":2,"y":3},{"x":3,"y":4}],"type":"pie"}],""" +
        """"exporting":{"filename":"chart"},""" +
        """"yAxis":[{"title":{"text":""}}],""" +
        """"plotOptions":{},""" +
        """"credits":{"href":"","text":""},""" +
        """"chart":{"zoomType":"xy"},""" +
        """"title":{"text":""},""" +
        """"xAxis":[{"title":{"text":""}}]}"""
    )
  }

}
