import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.{ContinuousDistr, Gamma}
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.kernels.{DiracKernel, LaplacianKernel, SEKernel}
import io.github.mandar2812.dynaml.modelpipe.GPRegressionPipe
import io.github.mandar2812.dynaml.pipes.{DataPipe, StreamDataPipe}
import io.github.mandar2812.dynaml.probability.MultGaussianRV
import io.github.mandar2812.dynaml.probability.mcmc.HyperParameterMCMC
import io.github.mandar2812.dynaml.utils.GaussianScaler
import com.quantifind.charts.Highcharts._


val deltaT = 4

type Features = DenseVector[Double]
type Data = Stream[(Features, Features)]

implicit val f = VectorField(deltaT)

val kernel = new SEKernel(1.5, 1.5)
val kernel2 = new LaplacianKernel(2.0)
val noise_kernel = new DiracKernel(1.0)

val data_size = 500

val scales_flow_stub = identityPipe[(GaussianScaler, GaussianScaler)]

val prepare_data = {
  fileToStream >
    trimLines >
    extractTrainingFeatures(
      List(0), Map()
    ) >
    DataPipe((lines: Stream[String]) =>
      lines.zipWithIndex.map(couple => (couple._2.toDouble, couple._1.toDouble))
    ) >
    deltaOperation(deltaT, 0) >
    StreamDataPipe((r: (Features, Double)) => (r._1, DenseVector(r._2))) >
    gaussianScaling >
    DataPipe(DataPipe((d: Data) => d.take(data_size)), scales_flow_stub)
}

val create_gp_model = GPRegressionPipe(
  (d: Data) => d.toSeq.map(p => (p._1, p._2(0))),
  kernel2, noise_kernel
)

val model_flow = DataPipe(create_gp_model, scales_flow_stub)

val workflow = prepare_data > model_flow

val (model, scales) = workflow("data/santafelaser.csv")

val num_hyp = model._hyper_parameters.length

val proposal = MultGaussianRV(
  num_hyp)(
  DenseVector.zeros[Double](num_hyp),
  DenseMatrix.eye[Double](num_hyp)*0.001)

val mcmc = HyperParameterMCMC[model.type, ContinuousDistr[Double]](
  model, model._hyper_parameters.map(h => (h, new Gamma(1.0, 2.0))).toMap,
  proposal)

mcmc.burnIn = 50

val samples = mcmc.iid(1000).draw

val samples_se = samples.map(h => (h("beta"), h("noiseLevel")))

scatter(samples_se)
title("x,y ~ P(sigma, a | Data)")
xAxis("sigma")
yAxis("a")
