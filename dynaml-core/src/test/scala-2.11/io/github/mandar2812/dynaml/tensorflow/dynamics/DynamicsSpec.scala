package io.github.mandar2812.dynaml.tensorflow.dynamics

import org.scalatest.{FlatSpec, Matchers}
import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.dynaml.tensorflow.pde._
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class DynamicsSpec extends FlatSpec with Matchers {

  val random = new Random()

  def batch(dim: Int, batchSize: Int): Tensor[FLOAT32] = {
    val inputs = ArrayBuffer.empty[Seq[Float]]
    var i = 0
    while (i < batchSize) {
      val input = Seq.tabulate(dim)(_ => random.nextFloat())
      inputs += input
      i += 1
    }

    Tensor(inputs).reshape(Shape(-1, dim))
  }

  val layer: Layer[Output, Output] = new Layer[Output, Output]("Exp") {
    override val layerType = "Exp"

    override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output =
      input.exp
  }

  "Gradient/Jacobian & Hessian operators on single output functions" should " yield correct results" in {

    val input_dim: Int = 3
    val output_dim: Int = 1

    val minibatch: Int = 1


    implicit val mode: Mode = tf.learn.INFERENCE

    val inputs = tf.placeholder(FLOAT32, Shape(-1, input_dim))

    val feedforward: Layer[Output, Output] = new Layer[Output, Output]("Exp") {
      override val layerType = "MatMul"

      override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output =
        tf.matmul(
          input,
          dtf.tensor_f32(input.shape(1), output_dim)(
            Seq.tabulate(input.shape(1), output_dim)((i, j) => if(i == j) 1.0f else 1.0f).flatten:_*
          )
        )
    }

    val linear_exp = layer >> feedforward

    val inter_outputs = layer.forward(inputs)

    val outputs = linear_exp.forward(inputs)

    val grad    = ∇(linear_exp)(inputs)

    val hess    = hessian(linear_exp)(inputs)

    val session = Session()
    session.run(targets = tf.globalVariablesInitializer())

    val trainBatch = batch(dim = input_dim, batchSize = minibatch)

    val feeds = Map(inputs -> trainBatch)

    val results: (Tensor[DataType], Tensor[DataType], Tensor[DataType], Tensor[DataType]) =
      session.run(feeds = feeds, fetches = (outputs, inter_outputs, grad, hess))


    assert(results._3 == results._2)

    for {
      i <- 0 until input_dim
      j <- 0 until input_dim
    } {
      if(i == j) results._4(0, i, j) == results._3(0, i)
      else results._4(0, i, j).scalar.asInstanceOf[Float] == 0f
    }

    session.close()

  }


  "Gradient/Jacobian & Hessian operators on multi-output functions" should " yield correct results" in {

    val input_dim: Int = 3
    val output_dim: Int = 2

    val minibatch: Int = 1


    implicit val mode: Mode = tf.learn.INFERENCE

    val inputs = tf.placeholder(FLOAT32, Shape(-1, input_dim))

    val feedforward: Layer[Output, Output] = new Layer[Output, Output]("Exp") {
      override val layerType = "MatMul"

      override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output =
        tf.matmul(
          input,
          dtf.tensor_f32(input.shape(1), output_dim)(
            Seq.tabulate(input.shape(1), output_dim)((i, j) => if(i == j) 1.0f else 1.0f).flatten:_*
          )
        )
    }

    val linear_exp = layer >> feedforward

    val inter_outputs = layer.forward(inputs)

    val outputs = linear_exp.forward(inputs)

    val grad    = ∇(linear_exp)(inputs)

    val hess    = hessian(linear_exp)(inputs)

    val session = Session()
    session.run(targets = tf.globalVariablesInitializer())

    val trainBatch = batch(dim = input_dim, batchSize = minibatch)

    val feeds = Map(inputs -> trainBatch)

    val results: (Tensor[DataType], Tensor[DataType], Tensor[DataType], Tensor[DataType]) =
      session.run(feeds = feeds, fetches = (outputs, inter_outputs, grad, hess))


    for {
      i <- 0 until input_dim
      z <- 0 until output_dim
    } {
      assert(results._3(0, i, z) == results._2(0, i))
    }

    for {
      i <- 0 until input_dim
      j <- 0 until input_dim
      z <- 0 until output_dim
    } {
      if(i == j) results._4(0, i, j, z) == results._3(0, i, z)
      else results._4(0, i, j, z).scalar.asInstanceOf[Float] == 0f
    }

    session.close()

  }



}
