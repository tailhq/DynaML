package io.github.tailhq.dynaml.graphics.charts.highcharts

import io.github.tailhq.dynaml.graphics.charts.Highcharts._
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression
import io.github.tailhq.dynaml.graphics.charts.highcharts.Highchart._

/**
 * User: jannis
 * Date: 12/12/14
 *
 * Plots both the scatter of the data points as well as the regression line for
 * the given data
 */

object LeastSquareRegression {

  def leastSquareRegression(xSeq: Seq[Double], ySeq: Seq[Double]) = {
    // regress the data
    val target: Array[Double]            = ySeq.toArray
    val predictor: Array[Array[Double]]  = xSeq.map(Array(_)).toArray
    val model = {
      val mod = new OLSMultipleLinearRegression()
      mod.newSampleData(target, predictor)
      mod
    }
    val params= model.estimateRegressionParameters
    val b  = params(0)
    val m = params(1)
    val residualRSquared = model.calculateAdjustedRSquared()

    
    // make the plot 
    val xMin = xSeq.min
    val xMax = xSeq.max
    val data = Series(xSeq.zip(ySeq).map{case (x,y) => Data(x,y)}, name = "Datapoints", chart = "scatter")
    val line = Series(
      data = List(Data(xMin, b + xMin * m), Data(xMax, b + xMax * m)),
      color = data.color,
      name = "y = " + f"$b%1.5f" + " + " + f"$m%1.5f" + " * x"
    )

    plot(Highchart(List(data,line), Some(Title("r² = " + f"$residualRSquared%1.5f"))))
  }
}
