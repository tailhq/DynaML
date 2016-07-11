package io.github.mandar2812.dynaml.models.neuralnets

import breeze.linalg.{DenseVector, sum}
import breeze.stats.distributions.Uniform
import io.github.mandar2812.dynaml.evaluation.MultiRegressionMetrics
import org.scalatest.{FlatSpec, Matchers}

/**
  * Created by mandar on 11/7/16.
  */
class AutoEncoderSpec extends FlatSpec with Matchers {

  /*"An auto-encoder"*/ ignore should "be able to learn a continuous, "+
    "invertible identity map x = g(h(x))" in {

    val uni = new Uniform(0.0, 1.0)
    //Create synthetic data set of x,y values

    val numPoints:Int = 5000
    val epsilon = 0.85

    val data = (1 to numPoints).map(_ => {
      val features = DenseVector.tabulate[Double](2)(_ => uni.draw)
      val augFeatures =
        DenseVector(
          features.toArray ++
            Array(
              math.pow(features(0)+features(1), 3),
              math.pow(features(0)+features(1), 2)))
      (augFeatures, augFeatures)
    })

    val enc = new AutoEncoder(4, 2, List("logsig", "logsig"))

    enc.optimizer
      .setRegParam(0.001)
      .setStepSize(1.0)
      .setNumIterations(150)
      .setMomentum(0.5)

    enc.learn(data.toStream)


    val metrics = new MultiRegressionMetrics(
      data.map(c => (enc.i(enc(c._1)), c._2)).toList,
      data.length)

    metrics.print()
    assert(sum(metrics.corr)/metrics.corr.length >= epsilon)

  }

}
