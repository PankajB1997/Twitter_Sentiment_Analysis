import org.apache.spark.sql.SparkSession
import scala.util.matching.Regex

object Preprocessor {

  def setupLogging() = {
    import org.apache.log4j.{Level, Logger}
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
  }

  def main(args: Array[String]): Unit = {
    // Initialize Spark Session
    val spark = SparkSession.builder.
      master("local[*]")
      .appName("EndToEnd")
      .config("spark.sql.catalogImplementation","hive")
      .enableHiveSupport()
      .getOrCreate()
    import spark.implicits._
    spark.sparkContext.setLogLevel("ERROR")

    // Load the raw tweets dataset and the stopwords file
    val lines = spark.sparkContext.textFile("src/main/resources/tweets_32M.csv")
    val stopWords = spark.sparkContext.textFile("src/main/resources/stopwords.txt")

    // Removing the @words
    val parsedHash = lines.map(x => x.replaceAll(Utility.REG_HANDLERS, ""))

    // Removing the links
    val parsedLinks = parsedHash.map(x => x.replaceAll(Utility.REG_LINKS, ""))

    // Removing punctuations
    val parsedPunctuations = parsedLinks.map(x => x.replaceAll(Utility.REG_PUNCTUATIONS, ""))

    // Splitting camelCase
    val pattern = new Regex(Utility.REG_CAMELCASE)
    val parsedCamelCase = parsedPunctuations.map(x => (pattern findAllIn x).mkString(" "))

    // Lowercase
    val parsedLowercase = parsedCamelCase.map(x => x.toLowerCase)

    // Expanding contractions
    val parsedContractions = parsedLowercase.map(x => x.split(" ")
      .map(y => if (Utility.contractions.contains(y)) Utility.contractions(y) else y).mkString(" "))

    // Removing stopWords
    val broadcastStopWords = spark.sparkContext.broadcast(stopWords.collect.toSet)
    val parsedStopWords = parsedContractions.map(x => x.split(" ")
      .map(y => if (!broadcastStopWords.value.contains(y)) y else "").mkString(" "))

    // Removing extra whitespaces
    val parsedWhitespaces = parsedStopWords.map(x => x.replaceAll(Utility.REG_WHITESPACES, " "))

    // Trimming the string
    val parsedTrim = parsedWhitespaces.map(x => x.trim())

    // Removing empty tweets
    val parsedEmpty = parsedTrim.filter(x => x.length > 0)

    // Save preprocessed tweets to HDFS
    val outputfile = "src/main/resources/tweets.preprocessed.csv"
    parsedEmpty.saveAsTextFile(outputfile)
  }
}
