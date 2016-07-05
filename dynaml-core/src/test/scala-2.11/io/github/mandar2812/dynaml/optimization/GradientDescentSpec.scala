package io.github.mandar2812.dynaml.optimization

import breeze.linalg.{DenseVector, norm}
import breeze.stats.distributions.{Gaussian, Uniform}
import io.github.mandar2812.dynaml.pipes.DataPipe
import org.scalatest.{FlatSpec, Matchers}

/**
  * Created by mandar on 5/7/16.
  */
class GradientDescentSpec extends FlatSpec with Matchers {

  "Gradient Descent" should "be able to solve Quadratic cost functions "+
    "of the form w^t.w + (w.x-y)^2 " in {
    val uni = new Uniform(0.0, 1.0)
    //Create synthetic data set of x,y values
    //x is sampled in unit hypercube, y = w.x + noise
    val noise = new Gaussian(0.0, 1.0)

    val w = DenseVector(1.0, -1.0)
    val wAug = DenseVector(1.0, -1.0, 0.0)

    val numPoints:Int = 1000


    val data = (1 to numPoints).map(_ => {
      val features = DenseVector.tabulate[Double](2)(_ => uni.draw)
      val target = (w dot features) //+ noise.draw

      (features, target)
    })

    val FMat = data.map(_._1).map(x => x * x.t).reduce((a,b) => a + b)


    val epsilon = 1E-2

    val transform = DataPipe((s: IndexedSeq[(DenseVector[Double], Double)]) => s.toStream)

    val wApprox = GradientDescent.runSGD(
      numPoints.toLong, 0.0, 1000,
      new SquaredL2Updater, new LeastSquaresGradient,
      1.0, DenseVector(0.0, 0.0, 0.0), data, transform)

    println("Learned W: "+wApprox)
    assert(norm(wApprox - wAug) <= epsilon)

  }
}
