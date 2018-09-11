package io.github.mandar2812.dynaml.tensorflow.data

import org.scalatest.{FlatSpec, Matchers}
import ammonite.ops._
import com.sksamuel.scrimage.Image
import com.sksamuel.scrimage.filter.GrayscaleFilter
import io.github.mandar2812.dynaml.pipes.DataPipe
import org.scalatest.{FlatSpec, Matchers}
import io.github.mandar2812.dynaml.tensorflow._
import org.platanios.tensorflow.api._

import scala.util.Random


class TFDataApiSpec extends FlatSpec with Matchers {

  private val image_paths =
    ls! pwd/'docs/'images |? (s => s.segments.last.contains("histogram-"))

  "Image datasets" should " be loaded into tensors" in {

    val t = dtfdata.create_image_tensor_buffered(
      buff_size = 2, DataPipe[Image, Array[Byte]](_.copy.subimage(0, 0, 10, 10).argb.flatten.map(_.toByte)),
      image_height = 10,image_width = 10, num_channels = 4)(image_paths, image_paths.length)

    val images_by_source = Iterable(image_paths.grouped(2).zipWithIndex.map(g => (s"Cat_${g._2}", g._1)).toMap)

    val image_process = images_by_source.head.map(
      kv => (kv._1, DataPipe[Image, Image](_.copy.subimage(0, 0, 10, 10).filter(GrayscaleFilter))))

    val images_to_bytes = DataPipe[Seq[Image], Array[Byte]](_.head.argb.map(_.last).map(_.toByte))

    val t2 = dtfdata.create_image_tensor_buffered(
      2, images_by_source.head.keys.toSeq, image_process, images_to_bytes,
      image_height = 10,image_width = 10, num_channels = images_by_source.head.keys.toSeq.length)(
      images_by_source, images_by_source.toSeq.length)

    assert(t.shape == Shape(image_paths.length, 10, 10, 4))

    assert(t2.shape == Shape(images_by_source.toSeq.length, 10, 10, images_by_source.head.keys.toSeq.length))

  }


}
