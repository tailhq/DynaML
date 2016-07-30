def filterll(data: Seq[Data])(p: Parameters): Loglikelihood = {
  val initFilter = Vector[FilterOut](filter(data.head, p)) // initialise the filter
 
  val filtered = data.tail.foldLeft(initFilter)((acc, nextObservation) => {
    // construct the parameters from the previous step of the filter
    val params = Parameters(p.v, w, acc.head.p.m0, acc.head.p.c0)
 
    // add the filtered observation to the head of the list
    filter(nextObservation, p) +: acc
}).reverse
 
// sum the values of the likelihood
filtered.map(_.likelihood).sum
}
