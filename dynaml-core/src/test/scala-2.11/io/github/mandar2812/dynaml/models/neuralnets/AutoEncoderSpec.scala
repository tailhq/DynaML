package io.github.mandar2812.dynaml.models.neuralnets

import breeze.linalg.{DenseVector, sum}
import breeze.stats.distributions.{Gaussian, Uniform}
import io.github.mandar2812.dynaml.evaluation.MultiRegressionMetrics
import org.scalatest.{FlatSpec, Matchers}
import io.github.mandar2812.dynaml.models.neuralnets.TransferFunctions._
import io.github.mandar2812.dynaml.optimization.BackPropagation

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
    val epsilon = 0.05

    val data = (1 to numPoints).map(_ => {
      val features = DenseVector.tabulate[Double](2)(_ => uni.draw)
      val augFeatures = DenseVector(
        math.pow(0.85*features(1), 2.5) + noise.draw,
        math.pow(0.45*features(0), 3.2) + noise.draw,
        math.pow(features(0)+0.85*features(1), 3) + noise.draw,
        math.pow(features(0)-0.5*features(1), 2) + noise.draw,
        math.pow(features(0)+features(1), 3) + noise.draw,
        math.pow(features(0)-features(1), 2) + noise.draw,
        math.pow(features(0)+0.4*features(1), 1.5) + noise.draw,
        math.pow(features(0)+0.5*features(1), 1.5) + noise.draw)

      (augFeatures, augFeatures)
    })

    val (trainingData, testData) = (data.take(3000), data.takeRight(1000))

    val enc = new AutoEncoder(8, 4, List(SIGMOID, LIN))

    BackPropagation.rho = 0.5
    enc.optimizer
      .setRegParam(0.0)
      .setStepSize(1.5)
      .setNumIterations(200)
      .setMomentum(0.4)
      .setSparsityWeight(0.9)

    enc.learn(trainingData.toStream)


    val metrics = new MultiRegressionMetrics(
      testData.map(c => (enc.i(enc(c._1)), c._2)).toList,
      testData.length)

    println("Corr: "+metrics.corr)
    assert(sum(metrics.mae)/metrics.corr.length <= epsilon)

  }

}
