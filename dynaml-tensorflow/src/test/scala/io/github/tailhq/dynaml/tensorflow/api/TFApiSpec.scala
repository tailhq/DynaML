package io.github.tailhq.dynaml.tensorflow.api

import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import io.github.tailhq.dynaml.tensorflow._
import org.platanios.tensorflow.api._
import _root_.io.github.tailhq.dynaml.probability.GaussianRV
import org.platanios.tensorflow.api.core.types.UByte


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

    val int16tensor = dtf.tensor_i16(2, 2)(numbers.map(_.toShort):_*)

    val uint8tensor = dtf.tensor_from[UByte](2, 2)(numbers.map(_.toByte).map(UByte))

    val int32tensor2 = dtf.tensor_from[Int](Shape(2, 2))(numbers:_*)

    val int32tensor = dtf.tensor_i32(2, 2)(numbers:_*)

    val int64tensor = dtf.tensor_i64(2, 2)(numbers.map(_.toLong):_*)

    val f16tensor = dtf.tensor_f16(2, 2)(numbers.map(_.toFloat):_*)

    val f32tensor = dtf.tensor_f32(2, 2)(numbers.map(_.toFloat):_*)

    val f64tensor = dtf.tensor_f64(2, 2)(numbers.map(_.toDouble):_*)

    val f_tensor  = dtf.fill[Float](3, 2)(1f)
    val f_tensor2  = dtf.fill[Float](Shape(3, 2))(1f)

    val f32tensor2 = dtf.tensor_from[Float](100, 100)((1 to 10000).map(_.toFloat))

    val f64tensor2 = dtf.tensor_from[Double](Shape(100, 100))((1 to 10000).map(_.toDouble):_*)
    
    val r_tensor = dtf.random[Double](3, 3)(GaussianRV(0.0, 1.0))

    val b_tensor = dtf.tensor_from_buffer[Int](Shape(5, 5), (0 until 100).map(_.toByte).toArray)

    val b_tensor2 = dtf.tensor_from_buffer[Int](5, 5)((0 until 100).map(_.toByte).toArray)

    assert(uint8tensor.dataType == UINT8 && uint8tensor.shape == Shape(2, 2))
    assert(int16tensor.dataType == INT16 && int16tensor.shape == Shape(2, 2))
    assert(int32tensor.dataType == INT32 && int32tensor.shape == Shape(2, 2))
    assert(int32tensor2.dataType == INT32 && int32tensor2.shape == Shape(2, 2))
    assert(int64tensor.dataType == INT64 && int64tensor.shape == Shape(2, 2))
    assert(f16tensor.dataType == FLOAT16 && f16tensor.shape == Shape(2, 2))
    assert(f32tensor.dataType == FLOAT32 && f32tensor.shape == Shape(2, 2))
    assert(f32tensor2.dataType == FLOAT32 && f32tensor2.shape == Shape(100, 100))
    assert(f64tensor.dataType == FLOAT64 && f64tensor.shape == Shape(2, 2))
    assert(f64tensor2.dataType == FLOAT64 && f64tensor2.shape == Shape(100, 100))
    assert(f_tensor.shape == Shape(3, 2) && f_tensor.dataType == FLOAT32)
    assert(f_tensor.entriesIterator.forall(_ == 1f))

    assert(f_tensor2.shape == Shape(3, 2) && f_tensor2.dataType == FLOAT32)
    assert(f_tensor2.entriesIterator.forall(_ == 1f))

    assert(r_tensor.shape == Shape(3, 3) && r_tensor.dataType == FLOAT64)

    assert(b_tensor.shape == Shape(5, 5) && b_tensor.dataType == INT32)
    assert(b_tensor2.shape == Shape(5, 5) && b_tensor2.dataType == INT32)

  }


  "Tensors " should "stack/concatenate/unstack correctly" in {

    val numbers: Seq[Int] = 1 to 6

    val int16tensor = dtf.tensor_i16(2, 3)(numbers.map(_.toShort):_*)

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
