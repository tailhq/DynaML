package io.github.mandar2812.dynaml.kernels

import scalaxy.streams.optimize
import io.github.mandar2812.dynaml.pipes._

/**
  * A kernel/covariance function which can be seen as a combination
  * of base kernels over a subset of the input space.
  *
  * for example K((x1, y1), (x1, y2)) = k1(x1,x2) + k2(y1,y2)
  */
class DecomposableCovariance[S](kernels: LocalScalarKernel[S]*)(
  implicit encoding: Encoder[S, Array[S]],
  reducer: Reducer = Reducer.:+:) extends CompositeCovariance[S] {

  val kernelMap = kernels.map(k => (k.toString.split("\\.").last, k)).toMap

  state = kernels.map(k => {
    val id = k.toString.split("\\.").last
    k.state.map(h => (id+"/"+h._1, h._2))
  }).reduceLeft(_++_)

  val encodingTuple = encoding*encoding

  override val hyper_parameters: List[String] = kernels.map(k => {
    val id = k.toString.split("\\.").last
    k.hyper_parameters.map(h => id+"/"+h)
  }).reduceLeft(_++_)

  blocked_hyper_parameters = kernels.map(k => {
    val id = k.toString.split("\\.").last
    k.blocked_hyper_parameters.map(h => id+"/"+h)
  }).reduceLeft(_++_)


  protected def kernelBind(config: Map[String, Double]) = DataPipe((xy: (Array[S], Array[S])) => {

    val (x, y) = xy

    //Create a collection of kernel functions.
    val kxy: Iterable[(S, S) => Double] = kernelMap.map(k => {
      val kEvalFunc = k._2.evaluateAt(
        config
          .filterKeys(_.contains(k._1))
          .map(CompositeCovariance.truncateHyperParams)) _

      kEvalFunc
    })

    //Bind kernel functions to inputs by zip.
    optimize {
      (x, y, kxy).zipped.map((x, y, k) => k(x, y))
    }
  })

  protected def evaluationDataPipe(config: Map[String, Double]) = encodingTuple > kernelBind(config) > reducer

  override def setHyperParameters(h: Map[String, Double]): DecomposableCovariance.this.type = {
    //Sanity Check
    assert(effective_hyper_parameters.forall(h.contains),
      "All hyper parameters must be contained in the arguments")
    //group the hyper params by kernel id
    h.toSeq.filterNot(_._1.split("/").length == 1).map(kv => {
      val idS = kv._1.split("/")
      (idS.head, (idS.tail.mkString("/"), kv._2))
    }).groupBy(_._1).map(hypC => {
      val kid = hypC._1
      val hyper_params = hypC._2.map(_._2).toMap
      kernelMap(kid).setHyperParameters(hyper_params)
    })
    this
  }

  override def evaluateAt(config: Map[String, Double])(x: S, y: S): Double = evaluationDataPipe(config) run (x,y)

  override def gradientAt(config: Map[String, Double])(x: S, y: S): Map[String, Double] = reducer match {
    case SumReducer =>
      val (xs, ys) = (encoding*encoding)((x,y))

      xs.zip(ys).zip(kernels).map(coupleAndKern => {
        val (u,v) = coupleAndKern._1
        val id = coupleAndKern._1.toString.split("\\.").last
        coupleAndKern._2.gradientAt(config)(u,v).map(c => (id+"/"+c._1, c._2))
      }).reduceLeft(_++_)

    case ProductReducer =>
      val (xs, ys) = (encoding*encoding)((x,y))

      xs.zip(ys).zip(kernels).map(coupleAndKern => {

        val (u,v) = coupleAndKern._1
        val id = coupleAndKern._1.toString.split("\\.").last

        coupleAndKern._2.gradientAt(config)(u,v)
          .map(c => (id+"/"+c._1, c._2))
          .mapValues(_ * evaluateAt(config)(x,y)/coupleAndKern._2.evaluateAt(config)(x,y))
      }).reduceLeft(_++_)

    case _ =>
      super.gradientAt(config)(x, y)
  }
}
