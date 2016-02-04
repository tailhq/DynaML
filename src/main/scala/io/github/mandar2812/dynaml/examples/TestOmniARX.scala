package io.github.mandar2812.dynaml.examples

import java.io.File

import breeze.linalg.{DenseMatrix, DenseVector}
import com.github.tototoshi.csv.CSVWriter
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.gp.{GPNarXModel, GPRegression}
import io.github.mandar2812.dynaml.optimization.{GPMLOptimizer, GridSearch}
import io.github.mandar2812.dynaml.pipes.{DynaMLPipe, DataPipe}
import com.quantifind.charts.Highcharts._

/**
  * @author mandar2812 on 22/11/15.
  *
  * Test a GP-NARX model on the Omni Data set
  */
object TestOmniARX {

  def apply(year: Int, yeartest:Int,
            kernel: CovarianceFunction[DenseVector[Double],
              Double, DenseMatrix[Double]],
            delta: Int, stepAhead: Int, bandwidth: Double,
            noise: CovarianceFunction[DenseVector[Double],
              Double, DenseMatrix[Double]],
            num_training: Int, num_test: Int,
            column: Int, exoInputColumns: List[Int] = List(24),
            grid: Int, step: Double, globalOpt: String,
            stepSize: Double = 0.05,
            maxIt: Int = 200): Unit =
    runExperiment(year, yeartest, kernel, delta, stepAhead, bandwidth, noise,
      num_training, num_test, column, exoInputColumns,
      grid, step, globalOpt,
      Map("tolerance" -> "0.0001",
        "step" -> stepSize.toString,
        "maxIterations" -> maxIt.toString))

  def runExperiment(year: Int = 2006, yearTest:Int = 2007,
                    kernel: CovarianceFunction[DenseVector[Double],
                      Double, DenseMatrix[Double]],
                    deltaT: Int = 2, stepPred: Int = 3,
                    bandwidth: Double = 0.5,
                    noise: CovarianceFunction[DenseVector[Double],
                      Double, DenseMatrix[Double]],
                    num_training: Int = 200, num_test: Int = 50,
                    column: Int = 40, ex: List[Int] = List(24), grid: Int = 5,
                    step: Double = 0.2, globalOpt: String = "ML",
                    opt: Map[String, String]): Seq[Seq[AnyVal]] = {
    //Load Omni data into a stream
    //Extract the time and Dst values
    //separate data into training and test
    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots
    val modelTrainTest =
      (trainTest: ((Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)]),
        (DenseVector[Double], DenseVector[Double]))) => {
        val model = new GPNarXModel(deltaT, ex.length,
          kernel, noise, trainTest._1._1.toSeq)

        val gs = globalOpt match {
          case "GS" => new GridSearch[model.type](model)
            .setGridSize(grid)
            .setStepSize(step)
            .setLogScale(false)

          case "ML" => new GPMLOptimizer[DenseVector[Double],
            Seq[(DenseVector[Double], Double)],
            GPRegression](model)
        }

        val startConf = kernel.state ++ noise.state
        val (_, conf) = gs.optimize(startConf, opt)

        model.setState(conf)

        val res = model.test(trainTest._1._2.toSeq)

        val deNormalize = DataPipe((list: List[(Double, Double)]) =>
          list.map{l => (l._1*trainTest._2._2(-1) + trainTest._2._1(-1),
            l._2*trainTest._2._2(-1) + trainTest._2._1(-1))})

        val scoresAndLabelsPipe =
          DataPipe(
            (res: Seq[(DenseVector[Double], Double, Double, Double, Double)]) =>
              res.map(i => (i._3, i._2)).toList) > deNormalize

        val scoresAndLabels = scoresAndLabelsPipe.run(res)

        val metrics = new RegressionMetrics(scoresAndLabels,
          scoresAndLabels.length)


        metrics.print()
        metrics.generatePlots()

        //Plotting time series prediction comparisons
        line((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._2))
        hold()
        line((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._1))
        legend(List("Time Series", "Predicted Time Series (one hour ahead)"))
        unhold()
        val timeObs = scoresAndLabels.map(_._2).zipWithIndex.min._2
        val timeModel = scoresAndLabels.map(_._1).zipWithIndex.min._2


        Seq(
          Seq(year, yearTest, deltaT, ex.length, 1, num_training, num_test,
            metrics.mae, metrics.rmse, metrics.Rsq,
            metrics.corr, metrics.modelYield,
            timeObs.toDouble - timeModel.toDouble)
        )

      }

    val preProcessPipe = DynaMLPipe.fileToStream >
      DynaMLPipe.replaceWhiteSpaces >
      DynaMLPipe.extractTrainingFeatures(
        List(0,1,2,column)++ex,
        Map(
          16 -> "999.9", 21 -> "999.9",
          24 -> "9999.", 23 -> "999.9",
          40 -> "99999", 22 -> "9999999.",
          25 -> "999.9", 28 -> "99.99",
          27 -> "9.999", 39 -> "999",
          45 -> "99999.99", 46 -> "99999.99",
          47 -> "99999.99")
      ) >
      DynaMLPipe.removeMissingLines >
      DynaMLPipe.extractTimeSeriesVec((year,day,hour) => (day * 24) + hour) >
      DynaMLPipe.deltaOperationVec(deltaT)

    val trainTestPipe = DynaMLPipe.duplicate(preProcessPipe) >
      DynaMLPipe.splitTrainingTest(num_training, num_test) >
      DynaMLPipe.gaussianStandardization >
      DataPipe(modelTrainTest)


    trainTestPipe.run(("data/omni2_"+year+".csv",
      "data/omni2_"+yearTest+".csv"))



  }
}

object DstARXExperiment {

  def apply(years: List[Int] = (2007 to 2014).toList,
            testYears: List[Int] = (2000 to 2015).toList,
            modelSizes: List[Int] = List(50, 100, 150),
            deltas: List[Int] = List(1, 2, 3), exogenous: List[Int] = List(24),
            stepAhead: Int, bandwidth: Double,
            noise: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]],
            num_test: Int, column: Int, grid: Int, step: Double) = {

    val writer = CSVWriter.open(new File("data/OmniNARXRes.csv"), append = true)

    years.foreach((year) => {
      testYears.foreach((testYear) => {
        deltas.foreach((delta) => {
          modelSizes.foreach((modelSize) => {
            TestOmniARX.runExperiment(year, testYear, new FBMKernel(1.05),
              delta, stepAhead, bandwidth, noise,
              modelSize, num_test, column, exogenous,
              grid, step, "GS",
              Map("tolerance" -> "0.0001",
                "step" -> "0.1",
                "maxIterations" -> "100"))
              .foreach(res => writer.writeRow(res))
          })
        })
      })
    })

    writer.close()
  }
}