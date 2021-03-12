{
  import breeze.linalg.DenseVector
  import io.github.tailhq.dynaml.analysis.VectorField
  import io.github.tailhq.dynaml.kernels._
  import io.github.tailhq.dynaml.examples.AbottPowerPlant

  implicit val ev = VectorField(24)

  val kernel = new LaplacianKernel(5.5)
  val other_kernel = new RBFKernel(4.5)
  val other_kernel1 = new CauchyKernel(1.0)
  val matern = new GenericMaternKernel[DenseVector[Double]](1.0, 2)

  val otherSumK = kernel + other_kernel
  val noiseKernel = new DiracKernel(1.5)

  noiseKernel.block_all_hyper_parameters

  val mo_result = AbottPowerPlant(
    other_kernel, noiseKernel)(
    deltaT = 3,
      opt = Map(
        "globalOpt" -> "GS", "grid" -> "4",
        "step" -> "0.5", "tolerance" -> "0.0001",
        "maxIterations" -> "10"),
      num_training = 1025, num_test = 1000)



  val so_result = AbottPowerPlant(
    other_kernel, noiseKernel,
    opt = Map(
      "globalOpt" -> "GPC", "grid" -> "2",
      "step" -> "0.5", "tolerance" -> "0.0001",
      "maxIterations" -> "10", "policy" -> "CSA"),
    num_training = 1025, num_test = 1025,
    deltaT = 3, column = 8)

}


