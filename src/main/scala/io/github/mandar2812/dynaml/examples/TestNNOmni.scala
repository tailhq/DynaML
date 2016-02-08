package io.github.mandar2812.dynaml.examples

import java.io.File

import breeze.linalg.{DenseMatrix, DenseVector}
import com.github.tototoshi.csv.CSVWriter
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.models.neuralnets.{FeedForwardNetwork, FFNeuralGraph}
import io.github.mandar2812.dynaml.pipes.{DynaMLPipe, DataPipe}
import org.apache.log4j.Logger


/**
  * @author mandar2812 datum 22/11/15.
  *
  * Test a GP-NAR model on the Omni Data set
  */
object TestNNOmni {

  def apply(year: Int, yeartest:Int,
            hidden: Int = 2, nCounts:List[Int] = List(), acts:List[String],
            delta: Int, timeLag:Int, stepAhead: Int,
            num_training: Int, num_test: Int,
            column: Int, grid: Int,
            step: Double, globalOpt: String = "GS",
            stepSize: Double = 0.05,
            maxIt: Int = 200, mini: Double = 1.0,
            alpha: Double = 0.0, regularization: Double = 0.5): Unit =
    runExperiment(year, yeartest, hidden, nCounts, acts,
      delta, timeLag, stepAhead,
      num_training, num_test, column,
      grid, step, globalOpt,
      Map("tolerance" -> "0.0001",
        "step" -> stepSize.toString,
        "maxIterations" -> maxIt.toString,
        "miniBatchFraction" -> mini.toString,
        "momentum" -> alpha.toString,
        "regularization" -> regularization.toString
      ))

  def runExperiment(year: Int = 2006, yearTest:Int = 2007,
                    hidden: Int = 2, nCounts:List[Int] = List(), act:List[String],
                    deltaT: Int = 2, timelag:Int = 0, stepPred: Int = 3,
                    num_training: Int = 200, num_test: Int = 50,
                    column: Int = 40, grid: Int = 5,
                    step: Double = 0.2, globalOpt: String = "ML",
                    opt: Map[String, String]): Seq[Seq[AnyVal]] = {
    //Load Omni data into a stream
    //Extract the time and Dst values

    val logger = Logger.getLogger(this.getClass)

    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots
    val modelTrainTest =
      (trainTest: ((Stream[(DenseVector[Double], Double)],
        Stream[(DenseVector[Double], Double)]),
        (DenseVector[Double], DenseVector[Double]))) => {

        val gr = FFNeuralGraph(trainTest._1._1.head._1.length, 1, hidden,
          act, nCounts)

        val transform = DataPipe((d: Stream[(DenseVector[Double], Double)]) =>
          d.map(el => (el._1, DenseVector(el._2))))

        val model = new FeedForwardNetwork[Stream[(DenseVector[Double], Double)]](trainTest._1._1, gr, transform)

        model.setLearningRate(opt("step").toDouble)
          .setMaxIterations(opt("maxIterations").toInt)
          .setBatchFraction(opt("miniBatchFraction").toDouble)
          .setMomentum(opt("momentum").toDouble)
          .setRegParam(opt("regularization").toDouble)
          .learn()

        val res = model.test(trainTest._1._2)
        val scoresAndLabelsPipe =
          DataPipe(
            (res: Seq[(DenseVector[Double], DenseVector[Double])]) =>
              res.map(i => (i._1(0), i._2(0))).toList) > DataPipe((list: List[(Double, Double)]) =>
            list.map{l => (l._1*trainTest._2._2(-1) + trainTest._2._1(-1),
              l._2*trainTest._2._2(-1) + trainTest._2._1(-1))})

        val scoresAndLabels = scoresAndLabelsPipe.run(res)

        val metrics = new RegressionMetrics(scoresAndLabels,
          scoresAndLabels.length)

        metrics.print()
        metrics.generatePlots()

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
      DynaMLPipe.gaussianStandardization >
      DataPipe(modelTrainTest)

    trainTestPipe.run(("data/omni2_"+year+".csv",
      "data/omni2_"+yearTest+".csv"))

  }
}

object DstARNNExperiment {

  def apply(years: List[Int] = (2007 to 2014).toList,
            testYears: List[Int] = (2000 to 2015).toList,
            modelSizes: List[Int] = List(50, 100, 150),
            deltas: List[Int] = List(1, 2, 3),
            stepAhead: Int, bandwidth: Double,
            noise: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]],
            num_test: Int, column: Int, grid: Int, step: Double) = {

    val writer = CSVWriter.open(new File("data/OmniRes.csv"), append = true)

    years.foreach((year) => {
      testYears.foreach((testYear) => {
        deltas.foreach((delta) => {
          modelSizes.foreach((modelSize) => {
            TestOmniAR.runExperiment(year, testYear, new FBMKernel(1.05),
              delta, 0, stepAhead, bandwidth, noise,
              modelSize, num_test, column,
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