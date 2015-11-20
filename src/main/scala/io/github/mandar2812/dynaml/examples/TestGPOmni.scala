package io.github.mandar2812.dynaml.examples

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels.{RBFKernel, RBFCovFunc, FBMCovFunction}
import io.github.mandar2812.dynaml.models.gp.{GPRegression, GPTimeSeries}
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.utils

/**
  * Created by mandar on 19/11/15.
  */
object TestGPOmni {
  def apply(year: Int, bandwidth: Double = 0.5): Unit = {
    //Load Omni data into a stream
    //Extract the time and Dst values
    //separate data into training and test
    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots

    val processpipe = DataPipe(utils.textFileToStream _) >
      DataPipe((s: Stream[String]) => s.map(utils.replace("\\s+")(","))) >
      DataPipe((l: Stream[String]) => utils.extractColumns(l, ",", List(40,16,21,23,24),
        Map(16 -> "999.9", 21 -> "999.9", 24 -> "9999.", 23 -> "999.9", 40 -> "99999"))) >
      DataPipe((lines: Stream[String]) => lines.filter(line => !line.contains(",,"))) >
      DataPipe((lines: Stream[String]) => lines.map(line => {
        val split = line.split(",")
        (DenseVector(split.tail.map(_.toDouble)), split.head.toDouble)
      })) >
      /*DataPipe((lines: Stream[String]) => lines.map{line =>
        val splits = line.split(",")
        val timestamp = splits(1).toDouble * 24 + splits(2).toDouble
        (timestamp, splits(3).toDouble)
      }) >*/
      DataPipe((data: Stream[(DenseVector[Double], Double)]) => {
        (data.take(100), data.take(200).takeRight(50))
      }) >
      DataPipe((trainTest: (Stream[(DenseVector[Double], Double)], Stream[(DenseVector[Double], Double)])) => {
        val model = new GPRegression(new RBFKernel(bandwidth), trainTest._1.toSeq)
        val res = model.test(trainTest._2.toSeq)
        val metrics = new RegressionMetrics(res.map(i => (i._3, i._2)).toList, res.length)
        println(res.map(i => (i._2, i._3)).toList)
        metrics.generatePlots()
      })

    processpipe.run("data/omni2_"+year+".csv")

  }
}
