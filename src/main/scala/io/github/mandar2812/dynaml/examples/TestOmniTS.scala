package io.github.mandar2812.dynaml.examples

import java.text.{DateFormat, SimpleDateFormat}

import breeze.linalg.{DenseVector, DenseMatrix}
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.gp.{GPRegression, GPTimeSeries}
import io.github.mandar2812.dynaml.optimization.{GPMLOptimizer, GridSearch}
import io.github.mandar2812.dynaml.pipes.{DynaMLPipe, StreamDataPipe, DataPipe}
import io.github.mandar2812.dynaml.utils

/**
  * Created by mandar on 22/11/15.
  */
object TestOmniTS {

  def apply(year: Int, kernel: CovarianceFunction[Double, Double, DenseMatrix[Double]],
            bandwidth: Double,
            noise: CovarianceFunction[Double, Double, DenseMatrix[Double]],
            num_training: Int, num_test: Int,
            column: Int, grid: Int,
            step: Double, globalOpt: String,
            stepSize: Double,
            maxIt: Int): Unit =
    runExperiment(year, kernel, bandwidth, noise,
      num_training, num_test, column, grid, step, globalOpt,
      Map("tolerance" -> "0.0001",
        "step" -> stepSize.toString,
        "maxIterations" -> maxIt.toString))

  def runExperiment(year: Int = 2006,
                    kernel: CovarianceFunction[Double, Double, DenseMatrix[Double]],
                    bandwidth: Double = 0.5,
                    noise: CovarianceFunction[Double, Double, DenseMatrix[Double]],
                    num_training: Int = 200, num_test: Int = 50,
                    column: Int = 40, grid: Int = 5,
                    step: Double = 0.2, globalOpt: String = "ML",
                    opt: Map[String, String]): Unit = {
    //Load Omni data into a stream
    //Extract the time and Dst values
    //separate data into training and test
    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots

    val modelTrainTest =
      (trainTest: ((Stream[(Double, Double)],
        Stream[(Double, Double)]),
        (DenseVector[Double], DenseVector[Double]))) => {
        val model = new GPTimeSeries(kernel, noise, trainTest._1._1.toSeq)
        val gs = globalOpt match {
          case "GS" => new GridSearch[model.type](model)
            .setGridSize(grid)
            .setStepSize(step)
            .setLogScale(false)

          case "ML" => new GPMLOptimizer[Double,
            Seq[(Double, Double)],
            GPTimeSeries](model)
        }
        val (_, conf) = gs.optimize(kernel.state ++ noise.state, opt)

        model.setState(conf)

        val res = model.test(trainTest._1._2.toSeq)
        val scoresAndLabelsPipe =
          DataPipe((res: Seq[(Double, Double, Double, Double, Double)]) =>
            res.map(i => (i._3, i._2)).toList) >
            DataPipe((list: List[(Double, Double)]) =>
              list.map{l => (l._1*trainTest._2._2(1) + trainTest._2._1(1),
                l._2*trainTest._2._2(1) + trainTest._2._1(1))}
            )

        val scoresAndLabels = scoresAndLabelsPipe.run(res)

        val metrics = new RegressionMetrics(scoresAndLabels,
          scoresAndLabels.length)

        //println(scoresAndLabels)
        metrics.print()
        metrics.generatePlots()
      }

    val processpipe = DynaMLPipe.fileToStream >
      DynaMLPipe.replaceWhiteSpaces >
      DynaMLPipe.extractTrainingFeatures(
        List(0,1,2,column),
        Map(16 -> "999.9", 21 -> "999.9",
          24 -> "9999.", 23 -> "999.9",
          40 -> "99999", 22 -> "9999999.",
          25 -> "999.9", 28 -> "99.99",
          27 -> "9.999", 39 -> "999")) >
      DynaMLPipe.removeMissingLines >
      DynaMLPipe.extractTimeSeries((year,day,hour) => (day * 24) + hour)



    val trainTestPipe = DynaMLPipe.duplicate(processpipe) >
      DynaMLPipe.duplicate(
        DataPipe((l: Stream[(Double, Double)]) =>
          l.map(d => (DenseVector(d._1), d._2)))
      ) >
      DynaMLPipe.splitTrainingTest(num_training, num_test) >
      DynaMLPipe.trainTestGaussianStandardization >
      DataPipe((X: ((Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)]),
        (DenseVector[Double], DenseVector[Double]))) => {
        ((X._1._1.map(c => (c._1(0), c._2)),
          X._1._2.map(c => (c._1(0), c._2))),
          X._2)
      }) >
      DataPipe(modelTrainTest)

    trainTestPipe.run(("data/omni2_"+year+".csv",
      "data/omni2_"+year+".csv"))


  }
}
