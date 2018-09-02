package io.github.mandar2812.dynaml.optimization

import breeze.linalg.{DenseMatrix, DenseVector, det, norm, trace}
import breeze.numerics.sigmoid
import breeze.stats.distributions.{Gaussian, Uniform}
import io.github.mandar2812.dynaml.DynaMLPipe
import io.github.mandar2812.dynaml.graph.FFNeuralGraph
import io.github.mandar2812.dynaml.models.neuralnets._
import io.github.mandar2812.dynaml.optimization._
import io.github.mandar2812.dynaml.pipes.DataPipe
import org.scalatest.{FlatSpec, Matchers}

/**
  * Created by mandar on 5/7/16.
  */
class QuadraticOptimizationSpec extends FlatSpec with Matchers {

  "Gradient Descent" should "be able to minimize Quadratic cost functions "+
    "of the form w^t.w + (w.x-y)^2 " in {
    val uni = new Uniform(0.0, 1.0)
    //Create synthetic data set of x,y values
    //x is sampled in unit hypercube, y = w.x + noise
    val noise = new Gaussian(0.0, 0.01)

    val w = DenseVector(1.0, -1.0)
    val wAug = DenseVector(1.0, -1.0, 0.0)

    val numPoints:Int = 1000


    val data = (1 to numPoints).map(_ => {
      val features = DenseVector.tabulate[Double](2)(_ => uni.draw)
      val target = (w dot features) + noise.draw

      (features, target)
    })

    val epsilon = 2E-2

    val transform = DataPipe((s: IndexedSeq[(DenseVector[Double], Double)]) => s.toStream)

    val wApprox = GradientDescent.runSGD(
      numPoints.toLong, 0.0, 1000,
      new SquaredL2Updater, new LeastSquaresGradient,
      1.0, DenseVector(0.0, 0.0, 0.0), data, transform)

    //println("Learned W: "+wApprox)
    assert(norm(wApprox - wAug) <= epsilon)

  }

  "Quasi-Newton" should "be able to minimize Quadratic cost functions "+
    "of the form w^t.w + (w.x-y)^2 " in {
    val uni = new Uniform(0.0, 1.0)
    //Create synthetic data set of x,y values
    //x is sampled in unit hypercube, y = w.x + noise
    val noise = new Gaussian(0.0, 0.02)

    val w = DenseVector(1.0, -1.0)
    val wAug = DenseVector(1.0, -1.0, 0.0)

    val numPoints:Int = 1000

    val data = (1 to numPoints).map(_ => {
      val features = DenseVector.tabulate[Double](2)(_ => uni.draw)
      val target = (w dot features) + noise.draw

      (features, target)
    })

    val epsilon = 2E-2

    val transform = DataPipe((s: IndexedSeq[(DenseVector[Double], Double)]) => s.toStream)

    val wApprox = QuasiNewtonOptimizer.run(
      numPoints.toLong, 0.0, 500,
      new SimpleBFGSUpdater, new LeastSquaresGradient,
      1.0, DenseVector(0.0, 0.0, 0.0), data, transform)

    //println("Learned W: "+wApprox)
    assert(norm(wApprox - wAug) <= epsilon)

  }

  "Back-propagation with momentum" should "be able to minimize Quadratic cost functions "+
    "of the form w^t.w + (w.x-y)^2 " in {
    val uni = new Uniform(0.0, 1.0)
    //Create synthetic data set of x,y values
    //x is sampled in unit hypercube, y = w.x + noise
    val noise = new Gaussian(0.0, 0.002)
    val uniH = new Uniform(-1.0, 1.0)

    val w = DenseVector.tabulate[Double](2)(i => uniH.draw)
    val wAug = DenseVector(w.toArray ++ Array(0.0))
    val numPoints:Int = 2000

    val data = (1 to numPoints).map(_ => {
      val features = DenseVector.tabulate[Double](2)(_ => uniH.draw)
      val augFeatures = DenseVector(features.toArray ++ Array(1.0))

      val target = (wAug dot augFeatures) + noise.draw

      (features, target)
    })

    val epsilon = 1E-2

    val transform = DataPipe((s: IndexedSeq[(DenseVector[Double], Double)]) =>
      s.map(p => (p._1, DenseVector(p._2))).toStream)

    val wApprox = BackPropagation.run(
      numPoints.toLong, 0.0, 300,
      1.0, 0.9, 0.5,
      FFNeuralGraph(
        2,1,0, List("linear"),
        List(), biasFlag = true),
      data, transform)

    val err = wApprox.getSynapsesAsMatrix(1).toDenseVector - wAug
    //println("Hidden Layer weights: "+wAug)
    //println("Calculated Hidden Layer weights: "+wApprox.getSynapsesAsMatrix(1).toDenseVector)
    //println("Error in Hidden Layer weights: "+err)
    assert(norm(err) <= epsilon)
  }

  "BackPropagation" should "have working input neuron buffers" in {

    val data = Stream(
      (DenseVector(1.0, 2.0, 3.0), DenseVector(1.0, 1.0)),
      (DenseVector(4.0, 5.0, 6.0), DenseVector(2.0, 2.0)))

    val result = BackPropagation.processDataToNeuronBuffers(data)

    assert(result._1.length == 3 && result._2.length == 2)
    assert(result._1.head == List(1.0, 4.0) && result._2.head == List(1.0, 2.0))
  }


  "BackPropagation" should "be able to minimize Quadratic cost functions "+
    "of the form w^t.w + (w.x-y)^2 " in {
    //Create synthetic data set of x,y values
    //x is sampled in unit hypercube, y = w.x + noise
    val noise = new Gaussian(0.0, 0.002)
    val uniH = new Uniform(-1.0, 1.0)

    val w = DenseVector.tabulate[Double](2)(i => uniH.draw)
    val wAug = DenseVector(w.toArray ++ Array(0.0))
    val numPoints:Int = 500

    val data = (1 to numPoints).map(_ => {
      val features = DenseVector.tabulate[Double](2)(_ => uniH.draw)
      val augFeatures = DenseVector(features.toArray ++ Array(1.0))

      val target = (wAug dot augFeatures) + noise.draw

      (features, target)
    })

    val epsilon = 1E-1

    val transform = DataPipe((s: IndexedSeq[(DenseVector[Double], Double)]) =>
      s.map(p => (p._1, DenseVector(p._2))).toStream)

    val stackfactory = NeuralStackFactory(Seq(2, 1))(Seq(VectorLinear))
    val initial_net = stackfactory(Seq((DenseMatrix((uniH.draw(), uniH.draw())), DenseVector(uniH.draw()))))

    val backprop = new FFBackProp(stackfactory)
      .setNumIterations(1000)
      .setStepSize(0.01)
      .momentum_(0.1)
      .setRegParam(0.0)

    val new_net = backprop.optimize(numPoints, transform(data), initial_net)

    val learned_params = new_net._layers.head.parameters
    val wApprox = DenseVector(learned_params._1.toDenseVector.toArray ++ learned_params._2.toArray)

    val err = wApprox - wAug

    assert(norm(err)/norm(wAug) <= epsilon)
  }

}
