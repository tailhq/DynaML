package org.kuleuven.esat

import breeze.linalg.DenseMatrix
import org.kuleuven.esat.graphicalModels.{MarkovChain}

/**
 * Hello world!
 *
 */
object AppMarkov extends App {

  override def main(args: Array[String]): Unit = {
    //Read csv file

    val tr: DenseMatrix[Double] =
      DenseMatrix(
        (1.0,0.0,0.0,0.0),
        (0.0,1.0,0.0,0.0),
        (0.001,0.001,0.0,0.998),
        (0.2,0.3,0.4,0.1)
      )
    val model = MarkovChain(tr)

    model.setLearningRate(0.001)
      .setMaxIterations(args.apply(0).toInt)
      .learn

    println(model.stationaryDistribution)
  }

}
