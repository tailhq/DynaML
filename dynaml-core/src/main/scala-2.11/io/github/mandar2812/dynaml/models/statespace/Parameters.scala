package io.github.mandar2812.dynaml.models.statespace

import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.numerics.exp
import breeze.stats.distributions.{Gaussian, Rand}

sealed trait Parameters {
  override def toString = Parameters.flatten(this).mkString(", ")
  def |+|(that: Parameters): Parameters =
    Parameters.combine(this, that)
  def length: Int = Parameters.length(this)
  def isEmpty: Boolean = Parameters.isEmpty(this)
  def perturb(delta: Double): Rand[Parameters] =
    Parameters.perturb(delta)(this)
  def perturbIndep(delta: Vector[Double]): Rand[Parameters] =
    Parameters.perturbIndep(delta)(this)
}

case class LeafParameter(initParams: StateParameter, scale: Option[Double], sdeParam: SdeParameter) extends Parameters
case class BranchParameter(left: Parameters, right: Parameters) extends Parameters

object LeafParameter {
  def apply(): LeafParameter = {
    LeafParameter(EmptyParameter, None, EmptyStepParameter)
  }
}

object Parameters {
  /**
    * A method to combine parameters
    */
  def combine(lp: Parameters, rp: Parameters): Parameters =
    if (lp.isEmpty) {
      rp
    } else if (rp.isEmpty) {
      lp
    } else {
      BranchParameter(lp, rp)
    }

  /**
    * A zero element, which represents an empty parameter and forms a left and right identity
    */
  def zero: Parameters = LeafParameter()

  /**
    * Checks to see if a parameter is empty
    */
  def isEmpty(p: Parameters): Boolean = p match {
    case LeafParameter(p, v, s) => (p, v, s) match {
      case (EmptyParameter, None, EmptyStepParameter) => true
      case _ => false
    }
    case BranchParameter(lp, rp) => isEmpty(lp) && isEmpty(rp)
  }

  /**
    * Perturbs parameters independently according to a single delta value, ie all parameters are perturbed on the same scale
    */
  def perturb(delta: Double): Parameters => Rand[Parameters] = p => p match {
    case LeafParameter(initParams, v, sdeParams) =>
      for {
        init <- initParams.perturb(delta)
        sde <- sdeParams.perturb(delta)
      } yield LeafParameter(init, v map (x => x * exp(Gaussian(0, delta).draw)), sde)
    case BranchParameter(lp, rp) =>
      for {
        l <- perturb(delta)(lp)
        r <- perturb(delta)(rp)
      } yield BranchParameter(l, r)
  }

  /**
    * Perturb parameters allowing for a different scale for each parameter
    */
  def perturbIndep(delta: Vector[Double]): Parameters => Rand[Parameters] = p => p match {
    case LeafParameter(initParams, v, sdeParams) =>
      v match {
        case Some(x) => 
          for {
            init <- initParams.perturbIndep(delta.take(initParams.length))
            sde <- sdeParams.perturbIndep(delta.drop(initParams.length + 1))
          } yield LeafParameter(init, v map (x => x * exp(Gaussian(0, delta(initParams.length)).draw)), sde)
        case None =>
          for {
            init <- initParams.perturbIndep(delta.take(initParams.length))
            sde <- sdeParams.perturbIndep(delta.drop(initParams.length))
          } yield LeafParameter(init, None, sde)

      }
    case BranchParameter(lp, rp) =>
      for {
        l <- perturbIndep(delta.take(lp.length))(lp)
        r <- perturbIndep(delta.drop(lp.length))(rp)
      } yield BranchParameter(l, r)
  }

  /**
    * Flattens parameters into a Vector of parameters
    * useful for printing
    */
  def flatten(p: Parameters): Vector[Double] = p match {
    case LeafParameter(init, noise, sde) => noise match {
      case Some(v) => init.flatten ++ Vector(v) ++ sde.flatten
      case None => init.flatten ++ sde.flatten
    }
    case BranchParameter(lp, rp) => flatten(lp) ++ flatten(rp)
  }

  def length(p: Parameters): Int = {
    flatten(p).length
  }

  def getSdeParameterNames(p: SdeParameter): IndexedSeq[String] = p match {
    case BrownianParameter(m, s) =>
      val paramSize = (0 to (m.size - 1))
      (paramSize map (i => s"mu$i")) ++ (paramSize map (i => s"sigma$i"))
    case OrnsteinParameter(t, a, s) =>
      val paramSize = (0 to (t.size - 1))
      (paramSize map (i => s"theta$i")) ++ (paramSize map (i => s"alpha$i")) ++ (paramSize map (i => s"sigma$i"))
    case StepConstantParameter(a) => IndexedSeq("a")
  }

  def getInitParamNames(p: StateParameter): IndexedSeq[String] = p match {
    case GaussianParameter(m, s) =>
      val paramSize = (0 to (m.size -1))
      (paramSize map (i => s"m0$i")) ++ (paramSize map (i => s"C0$i"))
  }

  /**
    * Get parameter names using case class names
    * useful for header rows in CSV files etc
    * @param p the parameters to get the names of
    * @return an indexed sequence of strings
    */
  def getParameterNames(p: Parameters): IndexedSeq[String] = p match {
    case LeafParameter(init, noise, sde) => noise match {
      case Some(v) =>
        getInitParamNames(init) ++ Vector("noiseSd") ++ getSdeParameterNames(sde)
      case None => getInitParamNames(init) ++ getSdeParameterNames(sde)
    }
    case BranchParameter(lp, rp) => getParameterNames(lp) ++ getParameterNames(rp)
  }

  /**
    * Transforms a parameter tree to a map of strings to doubles
    * @param p a parameter tree
    * @return a map of parameter names -> parameter value
    */
  def paramsToMap(p: Parameters): Map[String, Double] = {
    (getParameterNames(p), flatten(p)).zipped.map{ case (k, v) => (k -> v) }.toMap
  }
}

sealed trait StateParameter {
  def length: Int = StateParameter.length(this)
  def flatten: Vector[Double] = StateParameter.flatten(this)
  def perturb(delta: Double): Rand[StateParameter]
  def perturbIndep(delta: Vector[Double]): Rand[StateParameter]
}

case class GaussianParameter(m0: DenseVector[Double], c0: DenseMatrix[Double]) extends StateParameter {
  def perturb(delta: Double): Rand[GaussianParameter] = {
    new Rand[GaussianParameter] {
      def draw = {
        GaussianParameter(
          m0 map (Gaussian(_, delta).draw),
          diag(diag(c0) map (x => x * exp(Gaussian(0, delta).draw))))
      }
    }
  }

  def perturbIndep(delta: Vector[Double]): Rand[GaussianParameter] = {
    new Rand[GaussianParameter] {
      def draw = {
        GaussianParameter(
          DenseVector(m0.data.zip(delta.take(m0.length)) map { case (x, d) => Gaussian(x, d).draw }),
          diag(DenseVector(diag(c0).toArray.zip(delta.drop(m0.length)) map { case (x, d) => x * exp(Gaussian(0, d).draw) })))
      }
    }
  }
}
case object EmptyParameter extends StateParameter {
  def perturb(delta: Double): Rand[StateParameter] = ???
  def perturbIndep(delta: Vector[Double]): Rand[StateParameter] = ???
}

object GaussianParameter {
  def apply(m0: Double, c0: Double): GaussianParameter = {
    new GaussianParameter(DenseVector(m0), DenseMatrix(c0))
  }
}

object StateParameter {
  def flatten(p: StateParameter): Vector[Double] = p match {
    case GaussianParameter(m, s) => (m.data ++ diag(s).toArray).toVector
  }
  def length(p: StateParameter): Int = flatten(p).length
}

sealed trait SdeParameter {
  def length: Int = SdeParameter.length(this)
  def flatten: Vector[Double] = SdeParameter.flatten(this)
  def perturb(delta: Double): Rand[SdeParameter]
  def perturbIndep(delta: Vector[Double]): Rand[SdeParameter]
}

object SdeParameter {
  def flatten(p: SdeParameter): Vector[Double] = p match {
    case BrownianParameter(m, s) => m.data.toVector ++ diag(s).data.toVector
    case OrnsteinParameter(theta, alpha, sigma) => theta.data.toVector ++ alpha.data.toVector ++ sigma.data.toVector
    case StepConstantParameter(a) => a.data.toVector
 }

  def length(p: SdeParameter): Int = flatten(p).length  
}

case class BrownianParameter(mu: DenseVector[Double], sigma: DenseMatrix[Double]) extends SdeParameter {
  def perturb(delta: Double): Rand[BrownianParameter] = {
    new Rand[BrownianParameter] {
      def draw = {
        BrownianParameter(
          mu map (Gaussian(_, delta).draw),
          diag(diag(sigma) map (x => x * exp(Gaussian(0, delta).draw))))
      }
    }
  }

  def perturbIndep(delta: Vector[Double]): Rand[BrownianParameter] = {
    new Rand[BrownianParameter] {
      def draw = {
        BrownianParameter(
          DenseVector(mu.data.zip(delta.take(mu.length)) map { case (x, d) => Gaussian(x, d).draw }),
          diag(DenseVector(diag(sigma).toArray.zip(delta.drop(mu.length)) map { case (x, d) => x * exp(Gaussian(0, d).draw) } ))
        )
      }
    }
  }
}

object BrownianParameter {
  def apply(mu: Double, sigma: Double): BrownianParameter = {
    new BrownianParameter(DenseVector(mu), DenseMatrix(sigma))
  }
  def apply(mu: Array[Double], sigma: Array[Double]): BrownianParameter = {
    new BrownianParameter(DenseVector(mu), diag(DenseVector(sigma)))
  }
}
case class OrnsteinParameter(theta: DenseVector[Double], alpha: DenseVector[Double], sigma: DenseVector[Double]) extends SdeParameter {

  def perturb(delta: Double): Rand[OrnsteinParameter] = {
    new Rand[OrnsteinParameter] {
      def draw = {
        // alpha and sigme are non-negative, propose on log-scale
        OrnsteinParameter(
          theta map (Gaussian(_, delta).draw),
          alpha map (x => x * exp(Gaussian(0, delta).draw)),
          sigma map (x => x * exp(Gaussian(0, delta).draw)))
      }
    }
  }

  def perturbIndep(delta: Vector[Double]): Rand[OrnsteinParameter] = {
    new Rand[OrnsteinParameter] {
      def draw = {
        // alpha and sigme are non-negative, propose on log-scale
        OrnsteinParameter(
          theta.data.zip(delta.take(theta.length)) map { case (t, d) => Gaussian(t, d).draw },
          alpha.data.zip(delta.drop(theta.length).take(alpha.length)) map { case (x, d) => x * exp(Gaussian(0, d).draw) },
          sigma.data.zip(delta.drop(theta.length + alpha.length)) map { case (x, d) => x * exp(Gaussian(0, d).draw) })
      }
    }
  }

}

object OrnsteinParameter {
  def apply(theta: Double, alpha: Double, sigma: Double): OrnsteinParameter = {
    new OrnsteinParameter(DenseVector(theta), DenseVector(alpha), DenseVector(sigma))
  }
  def apply(theta: Array[Double], alpha: Array[Double], sigma: Array[Double]): OrnsteinParameter = {
    new OrnsteinParameter(DenseVector(theta), DenseVector(alpha), DenseVector(sigma))
  }
}

case class StepConstantParameter(a: DenseVector[Double]) extends SdeParameter {
  def perturb(delta: Double): Rand[StepConstantParameter] = {
    new Rand[StepConstantParameter] {
      def draw = StepConstantParameter(
        a map (Gaussian(_, delta).draw))
    }
  }

  def perturbIndep(delta: Vector[Double]): Rand[StepConstantParameter] = {
    new Rand[StepConstantParameter] {
      def draw = StepConstantParameter(DenseVector(a.data.zip(delta) map { case (x, d) => Gaussian(x, d).draw }))
    }
  }
}

object StepConstantParameter {
  def apply(a: Double): StepConstantParameter = {
    new StepConstantParameter(DenseVector(a))
  }
}

case object EmptyStepParameter extends SdeParameter {
  def perturb(delta: Double): Rand[SdeParameter] = ???
  def perturbIndep(delta: Vector[Double]): Rand[SdeParameter] = ???
}
