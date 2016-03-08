package io.github.mandar2812.dynaml.examples

import java.io.File
import java.text.SimpleDateFormat
import java.util.{Calendar, GregorianCalendar, Date}

import breeze.linalg.{DenseMatrix, DenseVector}
import com.github.tototoshi.csv.CSVWriter
import com.quantifind.charts.Highcharts._
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels.{DiracKernel, CovarianceFunction}
import io.github.mandar2812.dynaml.pipes.{DynaMLPipe, StreamDataPipe, DataPipe}
import io.github.mandar2812.dynaml.utils
import org.apache.log4j.Logger

/**
  * Created by mandar on 29/1/16.
  */
object TestOmniPer {
  def apply(start: String = "2006/12/28/00",
            end: String = "2006/12/29/23",
            column:Int = 40): Seq[Seq[Double]] = {

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

    val names = Map(
      24 -> "Solar Wind Speed",
      16 -> "I.M.F Bz",
      40 -> "Dst",
      41 -> "AE",
      38 -> "Kp",
      39 -> "Sunspot Number",
      28 -> "Plasma Flow Pressure",
      23 -> "Proton Density"
    )

    val sdf: SimpleDateFormat = new SimpleDateFormat("yyyy/MM/dd/HH")
    val dateS: Date = sdf.parse(start)
    val dateE: Date = sdf.parse(end)

    val greg: GregorianCalendar = new GregorianCalendar()
    greg.setTime(dateS)
    val dayStart = greg.get(Calendar.DAY_OF_YEAR)
    val hourStart = greg.get(Calendar.HOUR_OF_DAY)
    val stampStart = (dayStart * 24) + hourStart
    val yearTest = greg.get(Calendar.YEAR)


    greg.setTime(dateE)
    val dayEnd = greg.get(Calendar.DAY_OF_YEAR)
    val hourEnd = greg.get(Calendar.HOUR_OF_DAY)
    val stampEnd = (dayEnd * 24) + hourEnd

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
        val (timeObs, timeModel, peakValuePred, peakValueAct) = names(column) match {
          case "Dst" =>
            (scoresAndLabels.map(_._2).zipWithIndex.min._2,
              scoresAndLabels.map(_._1).zipWithIndex.min._2,
              scoresAndLabels.map(_._1).min,
              scoresAndLabels.map(_._2).min)
          case _ =>
            (scoresAndLabels.map(_._2).zipWithIndex.max._2,
              scoresAndLabels.map(_._1).zipWithIndex.max._2,
              scoresAndLabels.map(_._1).max,
              scoresAndLabels.map(_._2).max)
        }

        logger.info("Timing Error; OSA Prediction: "+(timeObs-timeModel))

        line((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._2))
        hold()
        line((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._1))
        unhold()
        Seq(
          Seq(yearTest.toDouble, 1.0, scoresAndLabels.length.toDouble,
            metrics.mae, metrics.rmse, metrics.Rsq,
            metrics.corr, metrics.modelYield,
            timeObs.toDouble - timeModel.toDouble,
            peakValuePred,
            peakValueAct)
        )

      }

    val preProcessPipe = DataPipe(utils.textFileToStream _) >
      DataPipe(replaceWhiteSpaces) >
      DataPipe(extractTrainingFeatures) >
      StreamDataPipe((line: String) => !line.contains(",,")) >
      DataPipe(extractTimeSeries) >
      StreamDataPipe((couple: (Double, Double)) => couple._1 >= stampStart && couple._1 <= stampEnd) >
      DataPipe(deltaOperation)

    val trainTestPipe = preProcessPipe > DataPipe(modelTrainTest)

    trainTestPipe.run("data/omni2_"+yearTest+".csv")

  }
}

object DstPersistExperiment {

  def apply() = {
    val writer =
      CSVWriter.open(
        new File("data/OmniPerStormsRes.csv"),
        append = true)

      val stormsPipe =
        DynaMLPipe.fileToStream >
          DynaMLPipe.replaceWhiteSpaces >
          StreamDataPipe((stormEventData: String) => {
            val stormMetaFields = stormEventData.split(',')

            val eventId = stormMetaFields(0)
            val startDate = stormMetaFields(1)
            val startHour = stormMetaFields(2).take(2)

            val endDate = stormMetaFields(3)
            val endHour = stormMetaFields(4).take(2)

            val minDst = stormMetaFields(5).toDouble

            val stormCategory = stormMetaFields(6)

            val res = TestOmniPer(
              startDate+"/"+startHour, endDate+"/"+endHour)

            val row = Seq(
              eventId, stormCategory, 1.0,
              0.0, res.head(4), res.head(6),
              res.head(9)-res.head(10),
              res.head(10), res.head(8)
            )

            writer.writeRow(row)
          })

      stormsPipe.run("data/geomagnetic_storms.csv")
    }
}