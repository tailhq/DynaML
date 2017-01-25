package io.github.mandar2812.dynaml.models.lm

import breeze.linalg.DenseVector
import breeze.stats.distributions.{Gaussian, Uniform}
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

/**
  * @author mandar2812 date: 11/7/16.
  *
  * Test suite for generalized linear models (GLM)
  */
class GLMSpec extends FlatSpec with Matchers {

  "A regression GLM" should "be able to learn parameters using "+
    "SGD given a basis function set" in {

    //Create synthetic data set of x,y values
    //x is sampled in unit hypercube, y = w.x + noise
    val noise = new Gaussian(0.0, 0.002)
    val uniH = new Uniform(0.0, 1.0)


    val numPoints:Int = 5000

    val phi = (x: DenseVector[Double]) => {
      val (x1, x2) = (x(0), x(1))
      DenseVector(math.sin(x1), math.sin(x2), math.cos(2*x1), math.cos(2*x2))
    }

    val w = DenseVector(0.5, -0.75, 1.0, -0.25)
    val wAug = DenseVector(w.toArray ++ Array(-0.8))

    val data = (1 to numPoints).map(_ => {
      val features = DenseVector.tabulate[Double](2)(_ => uniH.draw)

      val phi_feat = DenseVector(phi(features).toArray ++ Array(1.0))

      (phi(features), (wAug dot phi_feat) + noise.draw())
    }).toStream

    val (trainingData, testData) = (data.take(4000), data.takeRight(1000))

    val epsilon = 0.85

    val model = new RegularizedGLM(trainingData, trainingData.length, phi)

    model.setRegParam(0.001).learn()

    val res = testData.map(p => (model.predict(p._1), p._2)).toList

    val metrics = new RegressionMetrics(res, res.length)

    assert(metrics.Rsq >= epsilon)
  }

}
