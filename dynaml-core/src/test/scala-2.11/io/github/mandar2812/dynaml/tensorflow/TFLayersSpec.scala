package io.github.mandar2812.dynaml.tensorflow

import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import io.github.mandar2812.dynaml.tensorflow._
import io.github.mandar2812.dynaml.pipes.DataPipe
import org.platanios.tensorflow.api._
import _root_.io.github.mandar2812.dynaml.probability.GaussianRV
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.{Activation, Layer}
import org.platanios.tensorflow.api.tensors.TensorLike

class TFLayersSpec extends FlatSpec with Matchers {


  "DynaML Tensorflow layers " should "build and compute correctly" in {

    implicit val mode: Mode = tf.learn.INFERENCE

    val vec_input = tf.placeholder(FLOAT32, Shape(-1, 3))

    val image_input = tf.placeholder(FLOAT32, Shape(-1, 3, 3, 4))

    val lin = dtflearn.feedforward(2, false)(0)

    val conv2d = dtflearn.conv2d(2, 4, 2, (1, 1))(0)

    val inception = dtflearn.inception_unit(
      4, Seq(1, 2, 3, 4),
      DataPipe[String, Layer[Output, Output]](dtflearn.identity[Output]),
      false)(0)


    val lin_output = lin.forward(vec_input)

    val conv_output = conv2d.forward(image_input)

    val inception_output = inception.forward(image_input)

    val session = Session()
    session.run(targets = tf.globalVariablesInitializer())


    val sample_input = dtf.fill(FLOAT32, 1, 3)(0f)

    val sample_image = dtf.fill(FLOAT32, 1, 3, 3, 4)(0f)

    val feeds = Map(vec_input -> sample_input, image_input -> sample_image)

    val (lin_tensor, conv_tensor, inception_tensor): (Tensor, Tensor, Tensor) =
      session.run(feeds = feeds, fetches = (lin_output, conv_output, inception_output))

    assert(
      lin.name == "Linear_0" &&
        lin.units == 2 &&
        conv2d.name == "Conv2D_0" &&
        conv2d.filterShape == Shape(2, 2, 4, 2))

    assert(
      lin_tensor.shape == Shape(1, 2) &&
        conv_tensor.shape == Shape(1, 3, 3, 2) &&
        inception_tensor.shape == Shape(1, 3, 3, 10))

    assert(
      lin_tensor.entriesIterator.forall(_.asInstanceOf[Float] == 0f) &&
        conv_tensor.entriesIterator.forall(_.asInstanceOf[Float] == 0f) &&
        inception_tensor.entriesIterator.forall(_.asInstanceOf[Float] == 0.0f)
    )

    session.close()

  }

}
