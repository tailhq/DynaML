{
  import scala.util.Random
  import org.platanios.tensorflow.api._
  import _root_.io.github.mandar2812.dynaml.tensorflow._

  val sess = Session()

  val size = 500

  val (u_init, ut_init) = (
    Seq.tabulate[Double](size*size)(_ => if(Random.nextDouble() <= 0.95) 0d else Random.nextDouble()),
    Seq.fill[Double](size*size)(0d)
  )

  val eps = tf.placeholder(FLOAT32, Shape(), name = "dt")
  val damping = tf.placeholder(FLOAT32, Shape(), name = "damping")


}