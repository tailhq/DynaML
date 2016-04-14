/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
* */
package io.github.mandar2812.dynaml.examples

import java.io.File
import java.text.SimpleDateFormat
import java.util.{Calendar, GregorianCalendar, Date}

import breeze.linalg.{DenseMatrix, DenseVector}
import com.github.tototoshi.csv.CSVWriter
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.gp.{GPNarXModel, GPRegression}
import io.github.mandar2812.dynaml.optimization.{GPMLOptimizer, GridSearch}
import io.github.mandar2812.dynaml.pipes.{StreamDataPipe, DynaMLPipe, DataPipe}
import com.quantifind.charts.Highcharts._
import org.apache.log4j.Logger

/**
  * @author mandar2812
  * date: 22/11/15.
  *
  * Test a GP-NARX model on the Omni Data set
  */
object TestOmniARX {

  def apply(trainstart: String = "2008/01/01/00",
            trainend: String = "2008/01/10/23",
            start: String = "2006/12/28/00",
            end: String = "2006/12/29/23",
            kernel: CovarianceFunction[DenseVector[Double],
              Double, DenseMatrix[Double]],
            delta: List[Int],
            noise: CovarianceFunction[DenseVector[Double],
              Double, DenseMatrix[Double]],
            column: Int,
            exoInputColumns: List[Int] = List(24),
            grid: Int, step: Double, globalOpt: String,
            stepSize: Double = 0.05,
            maxIt: Int = 200, action: String = "test") =
    runExperiment(
      trainstart, trainend, start, end, kernel, delta,
      stepPred = 0, noise,
      column, exoInputColumns,
      grid, step, globalOpt,
      Map("tolerance" -> "0.0001",
        "step" -> stepSize.toString,
        "maxIterations" -> maxIt.toString,
        "Use VBz" -> "false"),
      action)

  def runExperiment(trainstart: String, trainend: String,
                    start: String, end: String,
                    kernel: CovarianceFunction[DenseVector[Double],
                      Double, DenseMatrix[Double]],
                    deltaT: List[Int], stepPred: Int = 3,
                    noise: CovarianceFunction[DenseVector[Double],
                      Double, DenseMatrix[Double]],
                    column: Int = 40,
                    ex: List[Int] = List(24), grid: Int = 5,
                    step: Double = 0.2, globalOpt: String = "ML",
                    opt: Map[String, String],
                    action: String = "test"): Seq[Seq[Double]] = {

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

    val greg: GregorianCalendar = new GregorianCalendar()
    val sdf: SimpleDateFormat = new SimpleDateFormat("yyyy/MM/dd/HH")

    val trainDateS: Date = sdf.parse(trainstart)
    val trainDateE: Date = sdf.parse(trainend)

    greg.setTime(trainDateS)
    val traindayStart = greg.get(Calendar.DAY_OF_YEAR)
    val trainhourStart = greg.get(Calendar.HOUR_OF_DAY)
    val trainstampStart = (traindayStart * 24) + trainhourStart
    val yearTrain = greg.get(Calendar.YEAR)


    greg.setTime(trainDateE)
    val traindayEnd = greg.get(Calendar.DAY_OF_YEAR)
    val trainhourEnd = greg.get(Calendar.HOUR_OF_DAY)
    val trainstampEnd = (traindayEnd * 24) + trainhourEnd

    val dateS: Date = sdf.parse(start)
    val dateE: Date = sdf.parse(end)

    greg.setTime(dateS)
    val dayStart = greg.get(Calendar.DAY_OF_YEAR)
    val hourStart = greg.get(Calendar.HOUR_OF_DAY)
    val stampStart = (dayStart * 24) + hourStart
    val yearTest = greg.get(Calendar.YEAR)

    greg.setTime(dateE)
    val dayEnd = greg.get(Calendar.DAY_OF_YEAR)
    val hourEnd = greg.get(Calendar.HOUR_OF_DAY)
    val stampEnd = (dayEnd * 24) + hourEnd

    //Load Omni data into a stream
    //Extract the time and Dst values
    //separate data into training and test
    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots

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
      DynaMLPipe.extractTimeSeriesVec((year,day,hour) => (day * 24) + hour)

    val processTraining = if(opt("Use VBz").toBoolean) {
      preProcessPipe > StreamDataPipe((point: (Double, DenseVector[Double])) => {
        (point._1, DenseVector(point._2.toArray ++ Array(point._2(1) * point._2(2))))
      }) > StreamDataPipe((couple: (Double, DenseVector[Double])) =>
        couple._1 >= trainstampStart && couple._1 <= trainstampEnd) >
        DynaMLPipe.deltaOperationARX(deltaT)
    } else {
      preProcessPipe > StreamDataPipe((couple: (Double, DenseVector[Double])) =>
        couple._1 >= trainstampStart && couple._1 <= trainstampEnd) >
        DynaMLPipe.deltaOperationARX(deltaT)
    }

    val processTest = if(opt("Use VBz").toBoolean){
      preProcessPipe >
        StreamDataPipe((point: (Double, DenseVector[Double])) => {
          (point._1, DenseVector(point._2.toArray ++ Array(point._2(1) * point._2(2))))
        }) >
        StreamDataPipe((couple: (Double, DenseVector[Double])) =>
          couple._1 >= stampStart && couple._1 <= stampEnd) >
        DynaMLPipe.deltaOperationARX(deltaT)
    } else {
      preProcessPipe >
        StreamDataPipe((couple: (Double, DenseVector[Double])) =>
          couple._1 >= stampStart && couple._1 <= stampEnd) >
        DynaMLPipe.deltaOperationARX(deltaT)
    }


    val modelTrainTest =
      (trainTest: ((Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)]),
        (DenseVector[Double], DenseVector[Double]))) => {
        val model = new GPNarXModel(deltaT.max, ex.length,
          kernel, noise, trainTest._1._1)
        val num_training = trainTest._1._1.length

        // If a validation set is specified, process it
        // using the above pre-process Data Pipes and
        // feed it into the model instance.

        if(opt.contains("validationStart") && opt.contains("validationEnd")) {
          val validationDateS: Date = sdf.parse(opt("validationStart"))
          val validationDateE: Date = sdf.parse(opt("validationEnd"))

          greg.setTime(validationDateS)
          val valdayStart = greg.get(Calendar.DAY_OF_YEAR)
          val valhourStart = greg.get(Calendar.HOUR_OF_DAY)
          val valstampStart = (valdayStart * 24) + valhourStart
          val yearVal = greg.get(Calendar.YEAR)


          greg.setTime(validationDateE)
          val valdayEnd = greg.get(Calendar.DAY_OF_YEAR)
          val valhourEnd = greg.get(Calendar.HOUR_OF_DAY)
          val valstampEnd = (valdayEnd * 24) + valhourEnd

          val processValidation = if(opt("Use VBz").toBoolean) {
            preProcessPipe >
              StreamDataPipe((couple: (Double, DenseVector[Double])) =>
                couple._1 >= valstampStart && couple._1 <= valstampEnd) >
              StreamDataPipe((point: (Double, DenseVector[Double])) => {
                (point._1, DenseVector(point._2.toArray ++ Array(point._2(1) * point._2(2))))
              }) >
              DynaMLPipe.deltaOperationARX(deltaT)
          } else {
            preProcessPipe >
              StreamDataPipe((couple: (Double, DenseVector[Double])) =>
                couple._1 >= valstampStart && couple._1 <= valstampEnd) >
              DynaMLPipe.deltaOperationARX(deltaT)
          }

          val featureDims = trainTest._2._1.length - 1

          val meanFeatures = trainTest._2._1(0 until featureDims)
          val stdDevFeatures = trainTest._2._2(0 until featureDims)

          val meanTargets = trainTest._2._1(-1)
          val stdDevTargets = trainTest._2._2(-1)

          // Set processTargets to a data pipe
          // which re scales the predicted outputs and actual
          // outputs to their orignal scales using the calculated
          // mean and standard deviation of the targets.
          model.processTargets = StreamDataPipe((predictionCouple: (Double, Double)) =>
            (predictionCouple._1*stdDevTargets + meanTargets,
              predictionCouple._2*stdDevTargets + meanTargets)
          )

          /*model.scoresToEnergy = DataPipe((scoresAndLabels) => {
            scoresAndLabels.map((couple) => math.abs(couple._1-couple._2)).max
          })*/

          val standardizeValidationInstances = StreamDataPipe(
            (instance: (DenseVector[Double], Double)) => {

              ((instance._1 - meanFeatures) :/ stdDevFeatures,
                (instance._2 - meanTargets)/stdDevTargets)
            })

          model.validationSet =
            (processValidation > standardizeValidationInstances) run
              "data/omni2_"+yearTrain+".csv"
        }



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

        if (action != "energyLandscape") {
          val (_, conf) = gs.optimize(startConf, opt)

          model.setState(conf)

          val res = model.test(trainTest._1._2)

          val deNormalize1 = DataPipe((list: List[(Double, Double, Double, Double)]) =>
            list.map{l => (l._1*trainTest._2._2(-1) + trainTest._2._1(-1),
              l._2*trainTest._2._2(-1) + trainTest._2._1(-1),
              l._3*trainTest._2._2(-1) + trainTest._2._1(-1),
              l._4*trainTest._2._2(-1) + trainTest._2._1(-1))})

          val scoresAndLabelsPipe =
            DataPipe(
              (res: Seq[(DenseVector[Double], Double, Double, Double, Double)]) =>
                res.map(i => (i._3, i._2, i._4, i._5)).toList) > deNormalize1


          val scoresAndLabels = scoresAndLabelsPipe.run(res)

          val metrics = new RegressionMetrics(scoresAndLabels.map(i => (i._1, i._2)),
            scoresAndLabels.length)

          val (name, name1) =
            if(names.contains(column)) (names(column), names(column))
            else ("Value","Time Series")

          metrics.setName(name)

          metrics.print()

          //Plotting time series prediction comparisons
          line((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._2))
          hold()
          line((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._1))
          spline((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._3))
          hold()
          spline((1 to scoresAndLabels.length).toList, scoresAndLabels.map(_._4))
          legend(List(name1, "Predicted "+name1+" (one hour ahead)", "Lower Bar", "Higher Bar"))
          unhold()

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

          action match {
            case "test" =>
              Seq(
                Seq(yearTrain.toDouble, yearTest.toDouble, deltaT.head.toDouble,
                  ex.length.toDouble, 1.0, num_training.toDouble,
                  trainTest._1._2.length.toDouble,
                  metrics.mae, metrics.rmse, metrics.Rsq,
                  metrics.corr, metrics.modelYield,
                  timeObs.toDouble - timeModel.toDouble,
                  peakValuePred,
                  peakValueAct)
              )
            case "predict" => scoresAndLabels.map(i => Seq(i._2, i._1))

          }
        } else {
          gs.getEnergyLandscape(startConf, opt).map(k => {
            Seq(k._1) ++
              kernel.hyper_parameters.map(k._2(_)) ++
              noise.hyper_parameters.map(k._2(_))
          }).toSeq
        }
      }

    val trainTestPipe = DataPipe(processTraining, processTest) >
      DynaMLPipe.trainTestGaussianStandardization >
      DataPipe(modelTrainTest)


    trainTestPipe.run(("data/omni2_"+yearTrain+".csv",
      "data/omni2_"+yearTest+".csv"))
  }
}

object DstARXExperiment {

  def apply(years: List[Int] = (2007 to 2014).toList,
            testYears: List[Int] = (2000 to 2015).toList,
            modelSizes: List[Int] = List(50, 100, 150),
            deltas: List[Int] = List(1, 2, 3), exogenous: List[Int] = List(24),
            stepAhead: Int, bandwidth: Double,
            noise: Double, num_test: Int,
            column: Int, grid: Int,
            step: Double) = {

    val writer = CSVWriter.open(new File("data/OmniNARXRes.csv"), append = true)

    years.foreach((year) => {
      testYears.foreach((testYear) => {
          modelSizes.foreach((modelSize) => {
            TestOmniARX.runExperiment(
              year.toString+"/01/01/00",
              year.toString+"/01/10/23",
              testYear+"/12/12:00",
              testYear+"/12/12:23",
              new FBMKernel(bandwidth),
              deltas, stepAhead, new DiracKernel(noise),
              column, exogenous,
              grid, step, "GS",
              Map("tolerance" -> "0.0001",
                "step" -> "0.1",
                "maxIterations" -> "100",
                "Use VBz" -> "false"))
              .foreach(res => writer.writeRow(res))
          })
      })
    })

    writer.close()
  }

  def apply(trainstart: String, trainend: String,
            kernel: CovarianceFunction[DenseVector[Double],
              Double, DenseMatrix[Double]],
            deltas: List[Int],
            column: Int, ex: List[Int],
            options: Map[String, String]) = {
    val writer =
      CSVWriter.open(new File("data/"+
        options("fileID")+
        "OmniARXStormsRes.csv"), append = true)

    val initialKernelState = kernel.state

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
            kernel.setHyperParameters(initialKernelState)

            val res = TestOmniARX.runExperiment(
              trainstart, trainend,
              startDate+"/"+startHour,
              endDate+"/"+endHour,
              kernel, deltas, 0,
              new DiracKernel(2.0),
              column, ex,
              options("grid").toInt,
              options("step").toDouble,
              options("globalOpt"),
              options, action = options("action"))

            if(options("action") == "test") {
              val row = Seq(
                eventId, stormCategory,
                deltas.head.toDouble,
                res.head(5), res.head(8),
                res.head(10), res.head(13)-res.head(14),
                res.head(14), res.head(12))

              writer.writeRow(row)
            } else {
              writer.writeAll(res)
            }


          })

      stormsPipe.run("data/geomagnetic_storms.csv")
  }
}