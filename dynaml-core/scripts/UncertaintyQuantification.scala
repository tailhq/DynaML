//Uncertainty Quantification Benchmarks
import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.models.gp.GPRegression
import io.github.mandar2812.dynaml.optimization.GridSearch
import breeze.stats.distributions.Uniform
import io.github.mandar2812.dynaml.kernels.PeriodicKernel
import io.github.mandar2812.dynaml.probability.{GaussianRV, IIDRandomVarDistr, ProbabilityModel, RandomVariable}
import spire.implicits._


val num_features = 1
implicit val ev = VectorField(num_features)

val xPrior = RandomVariable(new Uniform(-4.0, 4.0))
val iidXPrior = IIDRandomVarDistr(xPrior) _


val likelihood = DataPipe((x: Double) => GaussianRV(math.atan(1000.0*x*x*x), 0.4))

val model = ProbabilityModel(xPrior, likelihood)

val data: Stream[(DenseVector[Double], Double)] =
  (1 to 500).map(_ => model.sample()).map(c => (DenseVector(c._1), c._2)).toStream

val testData = (1 to 500).map(_ => model.sample()).map(c => (DenseVector(c._1), c._2)).toStream

scatter(testData.map(c => (c._1(0), c._2)))

val kernel = new RBFKernel(4.5)
val perKernel = new PeriodicKernel(1.5, 2.0)
val noise = new DiracKernel(1.0)

val startConf = kernel.state ++ noise.state
val gpModel = new GPRegression(kernel, noise, data)

val gs =
  new GridSearch(gpModel).setGridSize(3).setStepSize(0.2).setLogScale(false)

val (tunedGP, _) = gs.optimize(startConf)

tunedGP.persist()

val gpLikelihood = DataPipe((x: Double) => {
  //val xStream = x.map(DenseVector(_))
  //tunedGP.predictiveDistribution(xStream)
  val pD = tunedGP.predictiveDistribution(Seq(DenseVector(x)))
  GaussianRV(pD.mu(0), pD.covariance(0,0))
})

val gpProbModel = ProbabilityModel(xPrior, gpLikelihood)

scatter((1 to testData.length).map(_ => gpProbModel.sample()).toStream)
