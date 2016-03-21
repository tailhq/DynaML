package io.github.mandar2812.dynaml.examples



import breeze.linalg.DenseVector
import com.quantifind.charts.Highcharts._
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels.LocalSVMKernel
import io.github.mandar2812.dynaml.models.svm.DLSSVM
import io.github.mandar2812.dynaml.pipes.{DynaMLPipe, DataPipe}
import org.apache.log4j.Logger

/**
  * Created by mandar on 11/2/16.
  */
object TestLSSVMOmni {

  def apply(year: Int, yeartest:Int,
            kern: LocalSVMKernel[DenseVector[Double]],
            delta: Int, timeLag:Int, stepAhead: Int,
            num_training: Int, num_test: Int,
            column: Int, regularization: Double = 0.5): Unit =
    runExperiment(year, yeartest, kern,
      delta, timeLag, stepAhead,
      num_training, num_test, column,
      Map("regularization" -> regularization.toString))

  def runExperiment(year: Int = 2006, yearTest:Int = 2007,
                    kernel: LocalSVMKernel[DenseVector[Double]],
                    deltaT: Int = 2, timelag:Int = 0, stepPred: Int = 3,
                    num_training: Int = 200, num_test: Int = 50,
                    column: Int = 40, opt: Map[String, String]): Seq[Seq[AnyVal]] = {
    //Load Omni data into a stream
    //Extract the time and Dst values

    val logger = Logger.getLogger(this.getClass)

    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots
    val modelTrainTest =
      (trainTest: ((Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)]),
        (DenseVector[Double], DenseVector[Double]))) => {

        val model = new DLSSVM(trainTest._1._1, num_training, kernel)

        model.setRegParam(opt("regularization").toDouble).learn()

        val res = trainTest._1._2.map(testpoint => (model.predict(testpoint._1), testpoint._2))

        val scoresAndLabelsPipe = DataPipe((list: List[(Double, Double)]) =>
            list.map{l => (l._1*trainTest._2._2(-1) + trainTest._2._1(-1),
              l._2*trainTest._2._2(-1) + trainTest._2._1(-1))})

        val scoresAndLabels = scoresAndLabelsPipe.run(res.toList)

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

        Seq(
          Seq(year, yearTest, deltaT, 1, num_training, num_test,
            metrics.mae, metrics.rmse, metrics.Rsq,
            metrics.corr, metrics.modelYield)
        )
      }

    val preProcessPipe = DynaMLPipe.fileToStream >
      DynaMLPipe.replaceWhiteSpaces >
      DynaMLPipe.extractTrainingFeatures(
        List(0,1,2,column),
        Map(
          16 -> "999.9", 21 -> "999.9",
          24 -> "9999.", 23 -> "999.9",
          40 -> "99999", 22 -> "9999999.",
          25 -> "999.9", 28 -> "99.99",
          27 -> "9.999", 39 -> "999",
          45 -> "99999.99", 46 -> "99999.99",
          47 -> "99999.99")
      ) > DynaMLPipe.removeMissingLines >
      DynaMLPipe.extractTimeSeries((year,day,hour) => (day * 24) + hour) >
      DynaMLPipe.deltaOperation(deltaT, timelag)

    val trainTestPipe = DynaMLPipe.duplicate(preProcessPipe) >
      DynaMLPipe.splitTrainingTest(num_training, num_test) >
      DynaMLPipe.trainTestGaussianStandardization >
      DataPipe(modelTrainTest)

    trainTestPipe.run(("data/omni2_"+year+".csv",
      "data/omni2_"+yearTest+".csv"))

  }
}

