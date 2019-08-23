import io.github.mandar2812.dynaml.probability.stan._
import breeze.math._
import breeze.numerics._
import breeze.linalg._

val gpCode = """
       data {
         int<lower=1> N;
         real x[N];
         vector[N] y;
       }
       transformed data {
         vector[N] mu = rep_vector(0, N);
       }
       parameters {
         real<lower=0> rho;
         real<lower=0> alpha;
         real<lower=0> sigma;
       }
       model {
         matrix[N, N] L_K;
         matrix[N, N] K = cov_exp_quad(x, alpha, rho);
         real sq_sigma = square(sigma);
       
         // diagonal elements
         for (n in 1:N)
           K[n, n] = K[n, n] + sq_sigma;
       
         L_K = cholesky_decompose(K);
       
         rho ~ inv_gamma(5, 5);
         alpha ~ std_normal();
         sigma ~ std_normal();
       
         y ~ multi_normal_cholesky(mu, L_K);
       }
       """

val xr = UniformRV(-2d, 2d)
val yr = DataPipe(
  (x: Double) => (GaussianRV(0d, 0.5d) + 10 * math.sin(x) * math.exp(-x * x))
)

val xs = xr.iid(100).draw.toSeq
val ys = xs.map(x => yr(x).draw)

val gp_model = DynaStan.from_code(gpCode)

val N = gp_model.data(gp_model.int(lower = 1))
val x = gp_model.data(gp_model.vector(N))
val y = gp_model.data(gp_model.vector(N))

val results = DynaStan
  .compile(gp_model)
  .withData(N, xs.length)
  .withData(x, xs)
  .withData(y, ys)
  .run(chains = 5)

