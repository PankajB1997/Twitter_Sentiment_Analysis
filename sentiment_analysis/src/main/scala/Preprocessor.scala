import org.apache.spark.sql.SparkSession
import scala.util.matching.Regex

object Preprocessor {

  def setupLogging() = {
    import org.apache.log4j.{Level, Logger}
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.
      master("local[*]")
      .appName("EndToEnd")
      .config("spark.sql.catalogImplementation","hive")
      .enableHiveSupport()
      .getOrCreate()
    import spark.implicits._

    spark.sparkContext.setLogLevel("ERROR")
    val lines = spark.sparkContext.textFile("src/main/resources/tweets_32M.csv")
    val outputfile = "src/main/resources/tweets.preprocessed.csv"

    val stopWords = spark.sparkContext.textFile("src/main/resources/stopwords.txt")

    val pattern = new Regex(Utility.REG_CAMELCASE)

    // Removing the @words
    val parsedHash = lines.map(x => x.replaceAll(Utility.REG_HANDLERS, ""))
    println(parsedHash.count())

    // Removing the links
    val parsedLinks = parsedHash.map(x => x.replaceAll(Utility.REG_LINKS, ""))
    println(parsedLinks.count())

    // Removing punctuations
    val parsedPunctuations = parsedLinks.map(x => x.replaceAll(Utility.REG_PUNCTUATIONS, ""))
    println(parsedPunctuations.count())

    // Splitting camelCase
    val parsedCamelCase = parsedPunctuations.map(x => (pattern findAllIn x).mkString(" "))
    println(parsedCamelCase.count())

    // Lowercase
    val parsedLowercase = parsedCamelCase.map(x => x.toLowerCase)
    println(parsedLowercase.count())

    // Expanding contractions
    val parsedContractions = parsedLowercase.map(x => x.split(" ")
      .map(y => if (Utility.contractions.contains(y)) Utility.contractions(y) else y).mkString(" "))
    println(parsedContractions.count())

    // Removing stopWords
    val broadcastStopWords = spark.sparkContext.broadcast(stopWords.collect.toSet)
    val parsedStopWords = parsedContractions.map(x => x.split(" ")
      .map(y => if (!broadcastStopWords.value.contains(y)) y else "").mkString(" "))
    println(parsedStopWords.count())

    // Removing extra whitespaces
    val parsedWhitespaces = parsedStopWords.map(x => x.replaceAll(Utility.REG_WHITESPACES, " "))
    println(parsedWhitespaces.count())

    // Trimming the string
    val parsedTrim = parsedWhitespaces.map(x => x.trim())
    println(parsedTrim.count())

    // Removing empty tweets
    val parsedEmpty = parsedTrim.filter(x => x.length > 0)
    println(parsedEmpty.count())

    parsedEmpty.saveAsTextFile(outputfile)
  }
}
