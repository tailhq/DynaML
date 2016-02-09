package io.github.mandar2812.dynaml.examples

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.models.neuralnets.{FFNeuralGraph, FeedForwardNetwork}
import io.github.mandar2812.dynaml.pipes.{DynaMLPipe, DataPipe}

/**
  * Created by mandar on 15/12/15.
  */
object TestNNHousing {

  def apply(hidden: Int = 2, nCounts:List[Int] = List(), acts:List[String], trainFraction: Double = 0.75,
            columns: List[Int] = List(13,0,1,2,3,4,5,6,7,8,9,10,11,12),
            stepSize: Double = 0.01, maxIt: Int = 300, mini: Double = 1.0): Unit =
    runExperiment(hidden, nCounts, acts,
      (506*trainFraction).toInt, columns,
      Map("tolerance" -> "0.0001",
        "step" -> stepSize.toString,
        "maxIterations" -> maxIt.toString,
        "miniBatchFraction" -> mini.toString
      )
    )

  def runExperiment(hidden: Int = 2, nCounts:List[Int] = List(), act: List[String],
                    num_training: Int = 200, columns: List[Int] = List(40,16,21,23,24,22,25),
                    opt: Map[String, String]): Unit = {

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
      }

    //Load Housing data into a stream
    //Extract the time and Dst values
    //separate data into training and test
    //pipe training data to model and then generate test predictions
    //create RegressionMetrics instance and produce plots

    val preProcessPipe = DynaMLPipe.fileToStream >
      DynaMLPipe.trimLines >
      DynaMLPipe.replaceWhiteSpaces >
      DynaMLPipe.extractTrainingFeatures(columns, Map()) >
      DynaMLPipe.splitFeaturesAndTargets

    val trainTestPipe = DynaMLPipe.duplicate(preProcessPipe) >
      DynaMLPipe.splitTrainingTest(num_training, 506-num_training) >
      DynaMLPipe.gaussianStandardization >
      DataPipe(modelTrainTest)

    trainTestPipe.run(("data/housing.data", "data/housing.data"))

  }

}
