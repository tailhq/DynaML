/*
 * Copyright (c) 2016. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
 * Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
 * Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
 * Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
 * Vestibulum commodo. Ut rhoncus gravida arcu.
 */

package io.github.mandar2812.dynaml.examples

import java.io.File
import java.text.SimpleDateFormat
import java.util.{Calendar, Date, GregorianCalendar}

import breeze.linalg.DenseVector
import com.github.tototoshi.csv.CSVWriter
import com.quantifind.charts.Highcharts._
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.pipes.{DataPipe, DynaMLPipe, StreamDataPipe}
import io.github.mandar2812.dynaml.utils
import org.apache.log4j.Logger

/**
  * Created by mandar on 8/4/16.
  */
object TestOmniNarmax {
  def apply(start: String = "2006/12/28/00",
            end: String = "2006/12/29/23"): Seq[Seq[Double]] = {

    val logger = Logger.getLogger(this.getClass)

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

    val narmax_params = DenseVector(0.8335, -3.083e-4, -6.608e-7, 0.13112,
      -2.1584e-10, 2.8405e-5, 1.5255e-10,
      7.3573e-5, 0.73433, 1.545e-4)


    val preProcessPipe = DynaMLPipe.fileToStream >
      DynaMLPipe.replaceWhiteSpaces > DynaMLPipe.extractTrainingFeatures(
      List(0, 1, 2, 40, 24, 15, 16, 28),
      Map(
        16 -> "999.9", 16 -> "999.9",
        21 -> "999.9", 24 -> "9999.",
        23 -> "999.9", 40 -> "99999",
        22 -> "9999999.", 25 -> "999.9",
        28 -> "99.99", 27 -> "9.999", 39 -> "999",
        45 -> "99999.99", 46 -> "99999.99",
        47 -> "99999.99")) >
      DynaMLPipe.removeMissingLines >
      DynaMLPipe.extractTimeSeriesVec((year,day,hour) => (day * 24) + hour) >
      StreamDataPipe((couple: (Double, DenseVector[Double])) => {
        val features = couple._2
        //Calculate the coupling function p^0.5 V^4/3 Bt sin^6(theta)
        val Bt = math.sqrt(math.pow(features(2), 2) + math.pow(features(2), 2))
        val sin_theta6 = math.pow(features(2)/Bt,6)
        val p = features(4)
        val v = features(1)
        val couplingFunc = math.sqrt(p)*math.pow(v, 4/3.0)*Bt*sin_theta6
        val Dst = features(0)
        (couple._1, DenseVector(Dst, couplingFunc))
      }) > StreamDataPipe((couple: (Double, DenseVector[Double])) =>
      couple._1 >= stampStart && couple._1 <= stampEnd) >
      DynaMLPipe.deltaOperationARX(List(2, 3)) >
      StreamDataPipe((couple: (DenseVector[Double], Double)) => {
        val vec = couple._1
        val Dst_t_1 = vec(0)
        val Dst_t_2 = vec(1)

        val couplingFunc_t_1 = vec(2)
        val couplingFunc_t_2 = vec(3)
        val couplingFunc_t_3 = vec(4)

        val finalFeatures = DenseVector(Dst_t_1, couplingFunc_t_1, couplingFunc_t_1*Dst_t_1,
          Dst_t_2, math.pow(couplingFunc_t_2, 2.0),
          couplingFunc_t_3, math.pow(couplingFunc_t_1, 2.0),
          couplingFunc_t_2, 1.0, math.pow(Dst_t_1, 2.0))


        (narmax_params dot finalFeatures, couple._2)
      }) > DataPipe((scoresAndLabels: Stream[(Double, Double)]) => {

        val metrics = new RegressionMetrics(scoresAndLabels.toList,
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
        val (timeObs, timeModel, peakValuePred, peakValueAct) = names(40) match {
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


      })


    preProcessPipe.run("data/omni2_"+yearTest+".csv")
  }
}


object TestOmniTL {

  val logger = Logger.getLogger(this.getClass)

  def apply(start: String = "2006/12/28/00",
            end: String = "2006/12/29/23") = {



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


    val yearStart = greg.get(Calendar.YEAR).toString

    val monthStart = if(greg.get(Calendar.MONTH) < 9) {
      "0"+(greg.get(Calendar.MONTH)+1).toString
    } else {
      (greg.get(Calendar.MONTH)+1).toString
    }

    val fileNameS = "dst_"+yearStart+"_"+monthStart+".txt"




    greg.setTime(dateE)
    val dayEnd = greg.get(Calendar.DAY_OF_YEAR)
    val hourEnd = greg.get(Calendar.HOUR_OF_DAY)
    val stampEnd = (dayEnd * 24) + hourEnd







  }

  def prepareTLFiles() = {

    val stormsPipe =
      DynaMLPipe.fileToStream >
        DynaMLPipe.replaceWhiteSpaces >
        StreamDataPipe((stormEventData: String) => {
          val stormMetaFields = stormEventData.split(',')

          val startDate = stormMetaFields(1)
          val endDate = stormMetaFields(3)

          val sdf: SimpleDateFormat = new SimpleDateFormat("yyyy/MM/dd")
          val greg: GregorianCalendar = new GregorianCalendar()

          greg.setTime(sdf.parse(startDate))
          val yearStart = greg.get(Calendar.YEAR).toString

          val monthStart = if(greg.get(Calendar.MONTH) < 9) {
            "0"+(greg.get(Calendar.MONTH)+1).toString
          } else {
            (greg.get(Calendar.MONTH)+1).toString
          }

          val fileNameS = "dst_"+yearStart+"_"+monthStart+".txt"

          greg.setTime(sdf.parse(endDate))

          val yearEnd = greg.get(Calendar.YEAR).toString

          val monthEnd = if(greg.get(Calendar.MONTH) < 9) {
            "0"+(greg.get(Calendar.MONTH)+1).toString
          } else {
            (greg.get(Calendar.MONTH)+1).toString
          }

          val fileNameE = "dst_"+yearEnd+"_"+monthEnd+".txt"

          if(fileNameE == fileNameS) {
            logger.info("Same Month")
            utils.downloadURL(
              "http://lasp.colorado.edu/space_weather/dsttemerin/archive/"+fileNameS,
              "data/"+fileNameS)
          } else {
            logger.info("Different Months!")
            utils.downloadURL(
              "http://lasp.colorado.edu/space_weather/dsttemerin/archive/"+fileNameS,
              "data/"+fileNameS)

            utils.downloadURL(
              "http://lasp.colorado.edu/space_weather/dsttemerin/archive/"+fileNameE,
              "data/"+fileNameE)
          }

        })

    stormsPipe.run("data/geomagnetic_storms.csv")

  }

}


object DstNMExperiment {

  def apply() = {
    val writer =
      CSVWriter.open(
        new File("data/OmniNMStormsRes.csv"),
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

          val res = TestOmniNarmax(startDate+"/"+startHour, endDate+"/"+endHour)

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