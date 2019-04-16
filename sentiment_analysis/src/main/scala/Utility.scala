object Utility {
  final val REG_HANDLERS = "@\\w+"
  final val REG_LINKS = "(https?:\\/\\/(?:www\\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\\.[^\\s]{2,}|www\\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\\.[^\\s]{2,}|https?:\\/\\/(?:www\\.|(?!www))[a-zA-Z0-9]+\\.[^\\s]{2,}|www\\.[a-zA-Z0-9]+\\.[^\\s]{2,})"
  final val REG_PUNCTUATIONS = "[^\\w\\s]"
  final val REG_WHITESPACES = "\\s+"
  final val REG_TRIM = "^[ \\t]+"
  final val REG_CAMELCASE = "[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))"

  val contractions = Map[String, String](
    "aint" -> "am not",
    "im" -> "i am",
    "ive" -> "i have",
    "whos" -> "who is",
    "arent" -> "are not",
    "cant" -> "cannot",
    "didnt" -> "did not",
    "doesnt" -> "does not",
    "couldnt" -> "could not",
    "dont" -> "do not",
    "hadnt" -> "had not",
    "hasnt" -> "has not",
    "havent" -> "have not",
    "isnt" -> "is not",
    "mightnt" -> "might not",
    "mustnt" -> "must not",
    "neednt" -> "need not",
    "shouldnt" -> "should not",
    "wasnt" -> "was not",
    "werent" -> "were not",
    "wont" -> "will not",
    "werent" -> "were not",
    "hes" -> "he is",
    "shes" -> "she is"
  )

  val LABELS = Array("anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise",
    "trust", "negative", "positive")
}
