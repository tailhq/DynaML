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

    val lin_stack = dtflearn.feedforward_stack(
      (i: Int) => dtflearn.identity(s"Id_$i"),
      FLOAT32)(
      Seq(2, 1),
      starting_index = 3,
      useBias = false)

    val lin_stack2 = dtflearn.feedforward_stack(
      (i: Int) => dtflearn.GeneralizedLogistic(s"Id_$i"),
      FLOAT32)(
      Seq(2, 1),
      starting_index = 100,
      useBias = false)

    val conv2d = dtflearn.conv2d(2, 4, 2, (1, 1))(0)

    val conv_pyramid_dropout = dtflearn.conv2d_pyramid(
      2, 4)(
      4, 2)(
      0.1f, true, 0.6f, 20)

    val conv_pyramid_batch = dtflearn.conv2d_pyramid(
      2, 4)(
      4, 2)(
      0.1f, false, 0.6f, 40)

    val inception = dtflearn.inception_unit(
      4, Seq(1, 2, 3, 4),
      DataPipe[String, Layer[Output, Output]](dtflearn.identity[Output]),
      false)(0)

    val inception_stack = dtflearn.inception_stack(
      4, Seq(Seq(1, 2, 3, 4), Seq(1, 1, 1, 1)),
      DataPipe[String, Layer[Output, Output]](dtflearn.identity[Output]),
      true)(10)

    val ctrnn_stack = dtflearn.ctrnn_block(2, 3)(1)
    val ctrnn_stack2 = dtflearn.ctrnn_block(4, 3, 0.5)(10)

    val rbf_l = dtflearn.rbf_layer("RBF_Layer1", 4)

    val lin_output = lin.forward(vec_input)

    val lin_stack_output = lin_stack.forward(vec_input)

    val lin_stack_output2 = lin_stack2.forward(vec_input)

    val conv_output = conv2d.forward(image_input)

    val conv_py_output1 = conv_pyramid_dropout(image_input)
    val conv_py_output2 = conv_pyramid_batch(image_input)

    val (ctrnn_output1, ctrnn_output2) = (ctrnn_stack.forward(vec_input), ctrnn_stack2.forward(vec_input))

    val rbf_output = rbf_l.forward(vec_input)


    val inception_output = inception.forward(image_input)

    val inception_stack_output = inception_stack.forward(image_input)

    val session = Session()
    session.run(targets = tf.globalVariablesInitializer())


    val sample_input = dtf.fill(FLOAT32, 1, 3)(0f)

    val sample_image = dtf.fill(FLOAT32, 1, 3, 3, 4)(0f)

    val feeds = Map(vec_input -> sample_input, image_input -> sample_image)

    val (lin_tensor, lin_stack_tensor, lin_stack_tensor2, conv_tensor, inception_tensor, in_stack_tensor)
    : (Tensor[DataType], Tensor[DataType], Tensor[DataType], Tensor[DataType], Tensor[DataType], Tensor[DataType]) =
      session.run(
        feeds = feeds,
        fetches = (
          lin_output,
          lin_stack_output,
          lin_stack_output2,
          conv_output,
          inception_output,
          inception_stack_output)
    )

    val (conv_batch, conv_dropout, ctrnn_tensor1, ctrnn_tensor2, rbf_tensor)
    : (Tensor[DataType], Tensor[DataType], Tensor[DataType], Tensor[DataType], Tensor[DataType]) = session.run(
      feeds = feeds,
      fetches = (
        conv_py_output2,
        conv_py_output1,
        ctrnn_output1,
        ctrnn_output2,
        rbf_output
      )
    )

    assert(
      lin.name == "Linear_0" &&
        lin.units == 2 &&
        conv2d.name == "Conv2D_0" &&
        conv2d.filterShape == Shape(2, 2, 4, 2))

    assert(
      lin_tensor.shape == Shape(1, 2) &&
        lin_stack_tensor.shape == Shape(1, 1) &&
        lin_stack_tensor2.shape == Shape(1, 1) &&
        conv_tensor.shape == Shape(1, 3, 3, 2) &&
        conv_batch.shape == Shape(1, 1, 1, 4) &&
        conv_dropout.shape == Shape(1, 1, 1, 4) &&
        inception_tensor.shape == Shape(1, 3, 3, 10) &&
        in_stack_tensor.shape == Shape(1, 3, 3, 4) &&
        ctrnn_tensor1.shape == Shape(1, 2, 3) &&
        ctrnn_tensor2.shape == Shape(1, 4, 3) &&
        rbf_tensor.shape == Shape(1, 4))

    assert(
      lin_tensor.entriesIterator.forall(_.asInstanceOf[Float] == 0f) &&
        lin_stack_tensor.entriesIterator.forall(_.asInstanceOf[Float] == 0f) &&
        conv_tensor.entriesIterator.forall(_.asInstanceOf[Float] == 0f) &&
        conv_batch.entriesIterator.forall(_.asInstanceOf[Float] == 0f) &&
        inception_tensor.entriesIterator.forall(_.asInstanceOf[Float] == 0.0f) &&
        in_stack_tensor.entriesIterator.forall(_.asInstanceOf[Float] == 0.0f)
    )



    session.close()

  }

}
