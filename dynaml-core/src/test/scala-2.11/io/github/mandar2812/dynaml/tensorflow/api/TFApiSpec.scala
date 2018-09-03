package io.github.mandar2812.dynaml.tensorflow.api

import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import io.github.mandar2812.dynaml.tensorflow._
import org.platanios.tensorflow.api._

class TFApiSpec extends FlatSpec with Matchers with BeforeAndAfter {

  private var session: Session = _

  before {
    session = Session()
  }

  after {
    session.close()
  }

  //Check creation of tensors
  "Tensors " should "be created as per user-specified shape and data type" in {

    val numbers: Seq[Int] = 1 to 4

    val int16tensor = dtf.tensor_i16(2, 2)(numbers:_*)

    val uint8tensor = dtf.tensor_from("UINT8", 2, 2)(numbers.map(_.toByte))

    val int32tensor = dtf.tensor_i32(2, 2)(numbers:_*)

    val f32tensor = dtf.tensor_f32(2, 2)(numbers.map(_.toDouble):_*)

    val f64tensor = dtf.tensor_f64(2, 2)(numbers.map(_.toDouble):_*)

    assert(uint8tensor.dataType.toString() == "UINT8")
    assert(int16tensor.dataType.toString() == "INT16")
    assert(int32tensor.dataType.toString() == "INT32")
    assert(f32tensor.dataType.toString() == "FLOAT32")
    assert(f64tensor.dataType.toString() == "FLOAT64")
    assert(f64tensor.shape == Shape(2, 2))

  }


  "Tensors " should "stack/concatenate/unstack correctly" in {

    val numbers: Seq[Int] = 1 to 6

    val int16tensor = dtf.tensor_i16(2, 3)(numbers:_*)

    val unstack_tensors = dtf.unstack(int16tensor, axis = -1)

    val st_tensors = dtf.stack(unstack_tensors, axis = -1)

    val concat_tensors1 = dtf.concatenate(unstack_tensors.map(_.reshape(Shape(1, 2))), axis = -1)
    val concat_tensors2 = dtf.concatenate(unstack_tensors, axis = 0)

    assert(unstack_tensors.length == 3)
    assert(unstack_tensors.forall(_.shape == Shape(2)))
    assert(st_tensors.shape == Shape(2, 3))
    assert(concat_tensors1.shape == Shape(1, 6))
    assert(concat_tensors2.shape == Shape(6))

  }




}
