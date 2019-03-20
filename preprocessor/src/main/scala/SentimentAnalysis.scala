import org.apache.spark.SparkContext

object SentimentAnalysis {

  def setupLogging() = {
    import org.apache.log4j.{Level, Logger}
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
  }

  def main(args: Array[String]): Unit = {
    // val context = new SparkContext("local[*]", "SentimentAnalysis")
    val context = SparkContext.getOrCreate()
    val lines = context.textFile("dbfs:/FileStore/tables/tweets_large_32M.csv")
    val stopwords = context.textFile("dbfs:/FileStore/text/stopwords.txt")
    val outputfile = "dbfs:/FileStore/output/preprocessed_tweets.csv"

    // Removing the @words
    val parsedHash = lines.map(x => x.replaceAll(Utility.REG_HANDLERS, ""))

    // Removing the links
    val parsedLinks = parsedHash.map(x => x.replaceAll(Utility.REG_LINKS, ""))

    // Removing punctuations
    val parsedPunctuations = parsedLinks.map(x => x.replaceAll(Utility.REG_PUNCTUATIONS, ""))

    // Lowercase
    val parsedLowercase = parsedPunctuations.map(x => x.toLowerCase)

    // Expanding contractions
    val parsedContractions = parsedLowercase.map(x => x.split(" ")
      .map(y => if (Utility.contractions.contains(y)) Utility.contractions(y) else y).mkString(" "))

    // Removing stopwords
    val broadcastStopwords = context.broadcast(stopwords.collect.toSet)
    val parsedStopwords = parsedContractions.map(x => x.split(" ")
      .map(y => if (!broadcastStopwords.value.contains(y)) y else "").mkString(" "))

    // Removing extra whitespaces
    val parsedWhitespaces = parsedStopwords.map(x => x.replaceAll(Utility.REG_WHITESPACES, " "))

    // Trimming the string
    val parsedTrim = parsedWhitespaces.map(x => x.replaceAll(Utility.REG_TRIM, ""))

    parsedTrim.saveAsTextFile(outputfile)

    // var count = 0
    // for (result <- parsedTrim.collect() if count < 5) {
    //   println(result)
    //   count = count + 1
    // }
  }
}