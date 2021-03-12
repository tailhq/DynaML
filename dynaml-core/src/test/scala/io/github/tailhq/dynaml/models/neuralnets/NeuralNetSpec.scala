package io.github.tailhq.dynaml.models.neuralnets

import breeze.linalg.{DenseVector, sum}
import breeze.stats.distributions.{Gaussian, Uniform}
import io.github.tailhq.dynaml.DynaMLPipe
import io.github.tailhq.dynaml.evaluation.MultiRegressionMetrics
import io.github.tailhq.dynaml.graph.FFNeuralGraph
import org.scalatest.{FlatSpec, Matchers}

/**
  * @author tailhq date: 10/7/16.
  *
  * Test suite for neural network implementations
  * in DynaML
  */
class NeuralNetSpec extends FlatSpec with Matchers {

  "A feed-forward neural network" should "be able to learn non-linear functions "+
    "on a compact domain" in {
    val uni = new Uniform(0.0, 1.0)
    //Create synthetic data set of x,y values
    //x is sampled in unit hypercube, y = w.x + noise
    val noise = new Gaussian(0.0, 0.002)
    val uniH = new Uniform(0.0, 1.0)


    val numPoints:Int = 5000

    val data = (1 to numPoints).map(_ => {
      val features = DenseVector.tabulate[Double](4)(_ => uniH.draw)

      val (x,y,u,v) = (features(0), features(1), features(2), features(3))

      val target = DenseVector(
        1.0 + x*x + y*y*y + v*u*v + v*u + noise.draw,
        1.0 + x*u + u*y*y + v*v*v + u*u*u + noise.draw)

      (features, target)
    })

    val (trainingData, testData) = (data.take(4000), data.takeRight(1000))

    val epsilon = 0.85

    val model = new FeedForwardNetwork[Stream[(DenseVector[Double], DenseVector[Double])]](trainingData.toStream, FFNeuralGraph(4,2,0,
            List("logsig", "linear"),
            List(10), biasFlag = true))(DynaMLPipe.identityPipe[Stream[(DenseVector[Double], DenseVector[Double])]])

    model.setLearningRate(1.0)
      .setRegParam(0.01)
      .setMomentum(0.8)
      .setMaxIterations(150)
      .learn()

    val res = model.test(testData.toStream)

    val metrics = new MultiRegressionMetrics(res.toList, res.length)
    //println(metrics.Rsq)
    assert(sum(metrics.corr)/metrics.Rsq.length >= epsilon)
  }
}
