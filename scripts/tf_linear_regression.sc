{
  import org.platanios.tensorflow.api._

  import scala.collection.mutable.ArrayBuffer
  import scala.util.Random

  val random = new Random()

  val weight = random.nextFloat()

  def batch(batchSize: Int): (Tensor[Float], Tensor[Float]) = {
    val inputs = ArrayBuffer.empty[Float]
    val outputs = ArrayBuffer.empty[Float]
    var i = 0
    while (i < batchSize) {
      val input = random.nextFloat()
      inputs += input
      outputs += weight * input
      i += 1
    }
    (Tensor(inputs).reshape(Shape(-1, 1)), Tensor(outputs).reshape(Shape(-1, 1)))
  }

  print("Building linear regression model.")
  val inputs = tf.placeholder[Float](Shape(-1, 1))
  val outputs = tf.placeholder[Float](Shape(-1, 1))
  val weights = tf.variable[Float]("weights", Shape(1, 1), tf.ZerosInitializer)
  val predictions = tf.matmul(inputs, weights)
  val loss = tf.sum(tf.square(predictions - outputs))
  val trainOp = tf.train.AdaGrad(1.0f).minimize(loss)

  println("Training the linear regression model.")
  val session = Session()
  session.run(targets = tf.globalVariablesInitializer())
  for (i <- 0 to 50) {
    val trainBatch = batch(10000)
    val feeds = Map(inputs -> trainBatch._1, outputs -> trainBatch._2)
    val trainLoss = session.run(feeds = feeds, fetches = loss, targets = trainOp)
    if (i % 1 == 0)
      println(s"Train loss at iteration $i = ${trainLoss.scalar} " +
        s"(weight = ${session.run(fetches = weights.value).scalar})")
  }

  println(s"Trained weight value: ${session.run(fetches = weights.value).scalar}")
  println(s"True weight value: $weight")
}
