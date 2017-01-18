import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.kernels._
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.examples.AbottPowerPlant

implicit val ev = VectorField(6)
implicit val sp = genericReplicationEncoder[DenseVector[Double]](2)

val sp1 = breezeDVSplitEncoder(2)
val kernel = new LaplacianKernel(1.5)

val other_kernel = new RBFKernel(4.5)
val other_kernel1 = new CauchyKernel(1.0)

val otherSumK = kernel + other_kernel
val sumK2 = new DecomposableCovariance(otherSumK, other_kernel1)(sp1)

AbottPowerPlant(kernel, new DiracKernel(0.09),
  opt = Map("globalOpt" -> "GS", "grid" -> "2", "step" -> "0.004"),
  num_training = 3000, num_test = 1000, deltaT = 2, column = 8)

