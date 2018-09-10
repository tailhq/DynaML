package io.github.mandar2812.dynaml.tensorflow.dynamics

import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import _root_.io.github.mandar2812.dynaml.graphics.plot3d.DelauneySurface
import _root_.io.github.mandar2812.dynaml.graphics.plot3d
import _root_.io.github.mandar2812.dynaml.utils
import _root_.io.github.mandar2812.dynaml.analysis.implicits._
import _root_.io.github.mandar2812.dynaml.tensorflow._
import _root_.io.github.mandar2812.dynaml.tensorflow.pde._
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api.ops.variables.ConstantInitializer

import scala.util.Random

class WavePDESpec extends FlatSpec with Matchers with BeforeAndAfter {
  val random = new Random()

  def batch(dim: Int, min: Double, max: Double, gridSize: Int): Tensor = {

    val points = utils.combine(Seq.fill(dim)(utils.range(min, max, gridSize)))


    dtf.tensor_f32(Seq.fill(dim)(gridSize).product, dim)(points.flatten:_*)
  }

  val layer = new Layer[Output, Output]("Exp") {
    override val layerType = "Exp"

    override protected def _forward(input: Output)(implicit mode: Mode): Output =
      input.sin
  }


  implicit val mode: Mode = tf.learn.INFERENCE

  val domain = (0.0, 5.0)

  val domain_size = domain._2 - domain._1

  val input_dim: Int = 2

  val output_dim: Int = 1

  val inputs = tf.placeholder(FLOAT32, Shape(-1, input_dim))

  val feedforward = new Layer[Output, Output]("Exp") {
    override val layerType = "MatMul"

    override protected def _forward(input: Output)(implicit mode: Mode): Output =
      tf.matmul(
        input,
        dtf.tensor_f32(input.shape(1), output_dim)(
          Seq.tabulate(input.shape(1), output_dim)(
            (i, j) => if(i == j) math.Pi*2/domain_size else -math.Pi*2/domain_size).flatten:_*
        )
      ).reshape(Shape(input.shape(0)))
  }

  val function  = feedforward >> layer

  val outputs   = function.forward(inputs)

  val df_dt     = d_t(d_t)(function)(inputs)

  val df_ds     = d_s(d_s)(function)(inputs)

  val velocity  = variable[Output]("wave_velocity", FLOAT32, Shape(), ConstantInitializer(1.0f))

  val wave_op   = d_t(d_t) - d_s(d_s)*velocity

  val op_f      = wave_op(function)(inputs)

  var session: Session = _

  var trainBatch: Tensor = _

  var feeds: Map[Output, Tensor] = _

  var results: (Tensor, Tensor, Tensor, Tensor) = _

  before {

    session = Session()

    session.run(targets = tf.globalVariablesInitializer())

    trainBatch = batch(dim = input_dim, 0.0, 5.0, gridSize = 10)

    feeds = Map(inputs -> trainBatch)

    results = session.run(feeds = feeds, fetches = (outputs, df_ds, df_dt, op_f))


  }



  after {
    session.close()
  }

  "Wave Equation " should "have 'wave_velocity' as the epistemic source" in {

    assert(
      wave_op.variables.toSeq.length == 1 &&
        wave_op.variables.keys.toSeq == Seq("wave_velocity") &&
        wave_op.variables.toSeq.head._2.name == "wave_velocity")
  }

  "Plane wave solution" should " satisfy the wave PDE operator exactly" in {
    assert(results._2 == results._3 && results._4 == dtf.fill(FLOAT32, 10*10)(0f))
  }


}
