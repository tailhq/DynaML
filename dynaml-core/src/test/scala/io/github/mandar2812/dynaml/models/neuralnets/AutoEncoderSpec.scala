package io.github.mandar2812.dynaml.models.neuralnets

import breeze.linalg.{DenseVector, sum}
import breeze.stats.distributions.{Gaussian, Uniform}
import io.github.mandar2812.dynaml.evaluation.MultiRegressionMetrics
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability.RandomVariable
import spire.implicits._

import org.scalatest.{FlatSpec, Matchers}

/**
  * Created by mandar on 11/7/16.
  */
class AutoEncoderSpec extends FlatSpec with Matchers {

  /*"An auto-encoder"*/ ignore should "be able to learn a continuous, "+
    "invertible identity map x = g(h(x))" in {

    val uni = new Uniform(-math.Pi, math.Pi)
    val theta = RandomVariable(new Uniform(-math.Pi, math.Pi))
    val circleTransform = DataPipe((t: Double) => (math.cos(t), math.sin(t)))
    val rvOnCircle = theta > circleTransform
    //Create synthetic data set of x,y values

    val noise = new Gaussian(0.0, 0.02)

    val numPoints:Int = 4000
    val epsilon = 0.05

    val data = (1 to numPoints).map(_ => {
      val sample = rvOnCircle.draw
      val features = DenseVector(sample._1, sample._2)
      val augFeatures = DenseVector(
        math.pow(0.85*features(1), 2) + noise.draw,
        math.pow(0.45*features(0), 3) + noise.draw,
        math.pow(features(0)+0.85*features(1), 3) + noise.draw,
        math.pow(features(0)-0.5*features(1), 2) + noise.draw,
        math.pow(features(0)+features(1), 3) + noise.draw,
        math.pow(features(0)-features(1), 2) + noise.draw,
        math.pow(features(0)+0.4*features(1), 2) + noise.draw,
        math.pow(features(0)+0.5*features(1), 3) + noise.draw)

      augFeatures
    })

    val (trainingData, testData) = (data.take(3000), data.takeRight(1000))

    val enc = GenericAutoEncoder(List(8, 4, 4, 8), List(VectorTansig, VectorTansig, VectorTansig))

    //BackPropagation.rho = 0.5

    enc.optimizer.setRegParam(0.0001).setStepSize(0.1).setNumIterations(1000).momentum_(0.5)

    enc.learn(trainingData.toStream)

    val metrics = new MultiRegressionMetrics(
      testData.map(c => (enc.i(enc.f(c)), c)).toList,
      testData.length)

    println("Corr: "+metrics.corr)
    assert(sum(metrics.mae)/metrics.corr.length <= epsilon)

  }

}
