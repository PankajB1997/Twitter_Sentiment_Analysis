import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import java.util.Properties
import org.apache.spark.sql.types.{StructField, StructType, StringType, LongType}
import scala.util.matching.Regex

object EndToEnd {

  def main(args: Array[String]): Unit = {
    val hashtag = "got2" // Indicates the set of tweets being tested on; save table name using this hashtag

    val spark = SparkSession.builder.
      master("local[*]")
      .appName("EndToEnd")
      .config("spark.sql.catalogImplementation","hive")
      .enableHiveSupport()
      .getOrCreate()
    import spark.implicits._

    spark.sparkContext.setLogLevel("ERROR")

    // Step 1: Load the CSV file

    val lines = spark.read.option("header", "true")
      .option("multiLine", "true")
      .option("inferSchema", "true")
      .csv("src/main/resources/" + hashtag + "_df.csv")
      .map(x => x.toString()).rdd
    println(lines.count())

    // Step 2: Applying preprocessing on the raw tweets

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

    // Step 3 and 4: Load pretrained models and get predictions

    var preds: Array[DataFrame] = Array()

    val df = parsedEmpty.toDF("tweet")

    for (label <- Utility.LABELS) {
      val model = CrossValidatorModel.load("src/main/ml/model_lr_1.5M_" + label)
      val predictions = model.transform(df)
      preds = preds :+ predictions
    }

    // Step 5: Combine all predictions for each class into one df

    val parsedEmpty_res = parsedEmpty.toDF("tweet").withColumn("id", monotonically_increasing_id())
    val anger_preds = preds(0).select("anger_pred").toDF("anger").withColumn("id", monotonically_increasing_id())
    val anticipation_preds = preds(1).select("anticipation_pred").toDF("anticipation").withColumn("id", monotonically_increasing_id())
    val disgust_preds = preds(2).select("disgust_pred").toDF("disgust").withColumn("id", monotonically_increasing_id())
    val fear_preds = preds(3).select("fear_pred").toDF("fear").withColumn("id", monotonically_increasing_id())
    val joy_preds = preds(4).select("joy_pred").toDF("joy").withColumn("id", monotonically_increasing_id())
    val sadness_preds = preds(5).select("sadness_pred").toDF("sadness").withColumn("id", monotonically_increasing_id())
    val surprise_preds = preds(6).select("surprise_pred").toDF("surprise").withColumn("id", monotonically_increasing_id())
    val trust_preds = preds(7).select("trust_pred").toDF("trust").withColumn("id", monotonically_increasing_id())
    val negative_preds = preds(8).select("negative_pred").toDF("negative").withColumn("id", monotonically_increasing_id())
    val positive_preds = preds(9).select("positive_pred").toDF("positive").withColumn("id", monotonically_increasing_id())

    val id: Seq[String] = Array("id")

    val predictions = parsedEmpty_res.join(anger_preds, id, "outer")
      .join(anticipation_preds, id, "outer")
      .join(disgust_preds, id, "outer")
      .join(fear_preds, id, "outer")
      .join(joy_preds, id, "outer")
      .join(sadness_preds, id, "outer")
      .join(surprise_preds, id, "outer")
      .join(trust_preds, id, "outer")
      .join(negative_preds, id, "outer")
      .join(positive_preds, id, "outer")
      .drop("id")

    // Step 6: Connect to Azure SQL database

    val jdbcUsername = "bigdata_live"
    val jdbcPassword = "twitter_CS4225"
    val jdbcHostname = "cs4225.database.windows.net"
    val jdbcPort = 1433
    val jdbcDatabase ="twitter_CS4225"

    val jdbc_url = s"jdbc:sqlserver://${jdbcHostname}:${jdbcPort};database=${jdbcDatabase};encrypt=true;trustServerCertificate=false;hostNameInCertificate=*.database.windows.net;loginTimeout=60;"
    val connectionProperties = new Properties()
    connectionProperties.put("user", s"${jdbcUsername}")
    connectionProperties.put("password", s"${jdbcPassword}")
    val driverClass = "com.microsoft.sqlserver.jdbc.SQLServerDriver"
    connectionProperties.setProperty("Driver", driverClass)

    // Step 7: Compute Word Count across preprocessed tweets, output is a df with word and frequency columns

    val fields = Array(StructField("tweet", StringType, nullable = true))
    val schema = StructType(fields)

    val tweets = spark.createDataFrame(predictions.select("tweet").rdd, schema)

    val wordcounts = tweets.withColumn("word", explode(split(col("tweet"), " ")))
      .groupBy("word")
      .count()
      .sort($"count".desc)
      .take(150)

    wordcounts.foreach(println)

    // Step 8: Write word count results to Azure Spark SQL database table for that hashtag ("hashtag_wordcount")

    val fields2 = Array(StructField("word", StringType, nullable = false), StructField("count", LongType, nullable = false))
    val schema2 = StructType(fields2)
    spark.createDataFrame(spark.sparkContext.parallelize(wordcounts), schema2)
      .createOrReplaceTempView(hashtag + "_temp_wordcounts_table")
    spark.sql("drop table if exists " + hashtag + "_wordcounts")
    spark.sql("create table " + hashtag + "_wordcounts as select * from " + hashtag + "_temp_wordcounts_table")
    spark.table(hashtag + "_wordcounts").write.jdbc(jdbc_url, hashtag + "_wordcounts", connectionProperties)

    // Step 9: Write sentiment results to Azure Spark SQL database table for that hashtag ("hashtag_sentiments")

    predictions.createOrReplaceTempView(hashtag + "_temp_preds_table")
    spark.sql("drop table if exists " + hashtag + "_preds")
    spark.sql("create table " + hashtag + "_preds as select * from " + hashtag + "_temp_preds_table")
    spark.table(hashtag + "_preds").write.jdbc(jdbc_url, hashtag + "_preds", connectionProperties)

  }
}
