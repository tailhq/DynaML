package io.github.tailhq.dynaml.graphics.charts.highcharts

/**
 * User: austin
 * Date: 12/20/14
 */
object Histogram {
  def histogram(binCounts: Seq[(String, Double)]) = {
    // Does not rely on implicit imports - use import Highchart._ in an application!
    val series = Series(binCounts.zipWithIndex.map{case((bucket, count), index) => Data(index, count, name = Some(bucket))}, chart = Some(SeriesType.column))
    val plotOptions = Some(PlotOptions(series = Some(PlotOptionKey(
      groupPadding = Some(0),
      pointPadding = Some(0)
    ))))

    val numBins = binCounts.size
    val sqrt = math.sqrt(numBins).toInt

    val categories = binCounts.map(_._1).zipWithIndex.map{case(key, index) =>
      if(binCounts.size < 42 || index % sqrt == 0) key else ""
    }.toArray
    val xAxis = Some(Array(Axis(labels = Some(AxisLabel(rotation = Some(-45))), categories = Some(categories))))

    val hc = Highchart(series = Seq(series), plotOptions = plotOptions, xAxis = xAxis)

    hc
  }
}
