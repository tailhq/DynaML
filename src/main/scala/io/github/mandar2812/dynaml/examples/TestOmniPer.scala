package io.github.mandar2812.dynaml.examples

import breeze.linalg.{DenseMatrix, DenseVector}
import com.quantifind.charts.Highcharts._
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.pipes.{StreamDataPipe, DataPipe}
import io.github.mandar2812.dynaml.utils
import org.apache.log4j.Logger

/**
  * Created by mandar on 29/1/16.
  */
object TestOmniPer {
  def apply(yearTest:Int = 2007, num_test:Int = 2000, column:Int = 40): Seq[Seq[AnyVal]] = {
    //Load Omni data into a stream
    //Extract the time and Dst values
    //separate data into training and test
    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots
    val logger = Logger.getLogger(this.getClass)
    val replaceWhiteSpaces = (s: Stream[String]) => s.map(utils.replace("\\s+")(","))

    val extractTrainingFeatures = (l: Stream[String]) =>
      utils.extractColumns(l, ",", List(0,1,2,column),
        Map(16 -> "999.9", 21 -> "999.9",
          24 -> "9999.", 23 -> "999.9",
          40 -> "99999", 22 -> "9999999.",
          25 -> "999.9", 28 -> "99.99",
          27 -> "9.999", 39 -> "999"))

    val extractTimeSeries = (lines: Stream[String]) => lines.map{line =>
      val splits = line.split(",")
      val timestamp = splits(1).toDouble * 24 + splits(2).toDouble
      (timestamp, splits(3).toDouble)
    }


    val deltaOperation = (lines: Stream[(Double, Double)]) =>
      lines.toList.sliding(2).map((history) => {
        val features = DenseVector(history.take(history.length - 1).map(_._2).toArray)
        //assert(history.length == deltaT + 1, "Check one")
        //assert(features.length == deltaT, "Check two")
        (features, history.last._2)
      }).toStream

    val modelTrainTest =
      (trainTest: (Stream[(DenseVector[Double], Double)])) => {

        val scoresAndLabels = trainTest.map(x => (x._1(0), x._2)).toList

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


        logger.info("Printing One Step Ahead (OSA) Performance Metrics")
        metrics.print()
        val timeObs = scoresAndLabels.map(_._2).zipWithIndex.min._2
        val timeModel = scoresAndLabels.map(_._1).zipWithIndex.min._2
        logger.info("Timing Error; OSA Prediction: "+(timeObs-timeModel))


        line((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._2))
        hold()
        line((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._1))
        unhold()
        Seq(
          Seq(yearTest, 1, num_test,
            metrics.mae, metrics.rmse, metrics.Rsq,
            metrics.corr, metrics.modelYield,
            timeObs.toDouble - timeModel.toDouble)
        )

      }

    val preProcessPipe = DataPipe(utils.textFileToStream _) >
      DataPipe(replaceWhiteSpaces) >
      DataPipe(extractTrainingFeatures) >
      StreamDataPipe((line: String) => !line.contains(",,")) >
      DataPipe(extractTimeSeries) >
      DataPipe(deltaOperation) >
      DataPipe((data: Stream[(DenseVector[Double], Double)]) => data.takeRight(num_test))

    val trainTestPipe = preProcessPipe > DataPipe(modelTrainTest)

    trainTestPipe.run("data/omni2_"+yearTest+".csv")

  }
}
