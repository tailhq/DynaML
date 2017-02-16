package io.github.mandar2812.dynaml.pipes

/**
  * @author mandar date: 16/02/2017.
  *
  * Data Pipes representing functions of multiple arguments
  */
trait DataPipe2[-Source1, -Source2, +Result] extends Serializable {
  self =>

  def run(data1: Source1, data2: Source2): Result

  def apply(data1: Source1, data2: Source2): Result = run(data1, data2)

  def >[Result2](otherPipe: DataPipe[Result, Result2]): DataPipe2[Source1, Source2, Result2] =
    DataPipe2((d1: Source1, d2:Source2) => otherPipe.run(self.run(d1, d2)))

}

object DataPipe2 {
  def apply[Source1, Source2, Result](func2: (Source1, Source2) => Result): DataPipe2[Source1, Source2, Result] =
    new DataPipe2[Source1, Source2, Result] {
      override def run(data1: Source1, data2: Source2) = func2(data1, data2)
    }
}
