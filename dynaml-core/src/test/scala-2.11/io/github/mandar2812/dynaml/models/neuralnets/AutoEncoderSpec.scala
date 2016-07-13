package io.github.mandar2812.dynaml.models.neuralnets

import breeze.linalg.{DenseVector, sum}
import breeze.stats.distributions.{Gaussian, Uniform}
import io.github.mandar2812.dynaml.evaluation.MultiRegressionMetrics
import org.scalatest.{FlatSpec, Matchers}
import io.github.mandar2812.dynaml.models.neuralnets.TransferFunctions._

/**
  * Created by mandar on 11/7/16.
  */
class AutoEncoderSpec extends FlatSpec with Matchers {

  /*"An auto-encoder"*/ ignore should "be able to learn a continuous, "+
    "invertible identity map x = g(h(x))" in {

    val uni = new Uniform(0.0, 1.0)
    //Create synthetic data set of x,y values

    val noise = new Gaussian(0.0, 0.02)

    val numPoints:Int = 4000
    val epsilon = 0.85

    val data = (1 to numPoints).map(_ => {
      val features = DenseVector.tabulate[Double](2)(_ => uni.draw)
      val augFeatures = DenseVector(
        math.pow(features(0)+0.85*features(1), 3) + noise.draw,
        math.pow(features(0)-0.5*features(1), 2) + noise.draw,
        math.pow(features(0)+features(1), 3) + noise.draw,
        math.pow(features(0)-features(1), 2) + noise.draw,
        math.pow(features(0)+0.4*features(1), 1.5) + noise.draw,
        math.pow(features(0)+0.5*features(1), 1.5) + noise.draw)

      (augFeatures, augFeatures)
    })

    val (trainingData, testData) = (data.take(3000), data.takeRight(1000))

    val enc = new AutoEncoder(6, 3, List(SIGMOID, LIN))

    enc.optimizer
      .setRegParam(0.1)
      .setStepSize(1.2)
      .setNumIterations(200)
      .setMomentum(0.8)
      .setSparsityWeight(0.1)

    enc.learn(trainingData.toStream)


    val metrics = new MultiRegressionMetrics(
      testData.map(c => (enc.i(enc(c._1)), c._2)).toList,
      testData.length)

    metrics.print()
    assert(sum(metrics.corr)/metrics.corr.length >= epsilon)

  }

}
