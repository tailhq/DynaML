import io.github.mandar2812.dynaml.pipes._
import io.github.mandar2812.dynaml.probability.stan._
import io.github.mandar2812.dynaml.probability._
import spire.implicits._
import breeze.math._
import breeze.numerics._
import breeze.linalg._

val xr = UniformRV(-2d, 2d)
val dim = 2

val omega  = DenseVector.tabulate(dim)(_ => scala.util.Random.nextGaussian())
val xr_vec = RandomVariable(() => DenseVector.tabulate(dim)(_ => xr.draw))
val yr_vec = DataPipe(
  (x: DenseVector[Double]) =>
    (GaussianRV(0d, 0.5d) + 10 * sin(omega.t * x) * math.exp(-norm(x, 2)))
)

val x_data = xr_vec.iid(500).draw.toSeq
val y_data = x_data.map(yr_vec(_)).map(_.draw).toSeq

val stan_gp_model = new DynaStan {
  val n = data(int(lower = 1))
  val d = data(int(lower = 1))
  val x = data(vector(d)(n))
  val y = data(vector(n))

  val sigma = parameter(real(lower = 0.0))
  val l = parameter(real(lower = 0.0))
  val sigma_noise = parameter(real(lower = 0.0))

  sigma ~ stan.std_normal()
  sigma_noise ~ stan.std_normal()
  l ~ stan.inv_gamma(5, 5)

  val kernel_mat = local(matrix(n, n))
  val l_kernel_mat = local(matrix(n, n))

  for (i <- range(1, n - 1)) {
    kernel_mat(i, i) := stan.square(sigma) + stan.square(sigma_noise)
    for (j <- range(i + 1, n)) {
      kernel_mat(i, j) := stan.square(sigma) * stan.exp(
        stan.dot_self(x(i) - x(j)) * l * -0.5
      )
      kernel_mat(j, i) := kernel_mat(i, j)
    }
  }

  kernel_mat(n, n) := stan.square(sigma) + stan.square(sigma_noise)

  val mu = stan.rep_vector(0, n)

  l_kernel_mat := stan.cholesky_decompose(kernel_mat)

  y ~ stan.multi_normal_cholesky(mu, l_kernel_mat)
}

val results = stan_gp_model
  .withData(stan_gp_model.n, x_data.length)
  .withData(stan_gp_model.d, x_data.head.size)
  .withData(stan_gp_model.x, x_data.map(_.toArray.toSeq))
  .withData(stan_gp_model.y, y_data)
  .run(chains = 3)
