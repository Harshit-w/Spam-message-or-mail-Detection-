# Load Required Libraries
suppressPackageStartupMessages({
  library(tm)         
  library(e1071)      
  library(SnowballC)  
})

#  Load the SMS Spam Collection Dataset
# The dataset contains SMS messages labeled as "ham" (not spam) or "spam".
# It is downloaded from the UCI Machine Learning Repository.
temp <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
              temp, mode = "wb")
unzip(temp, "SMSSpamCollection")

# Read the data file: first column = label (ham/spam), second column = message text
sms_data <- read.table("SMSSpamCollection",
                       sep = "\t", header = FALSE, stringsAsFactors = FALSE,
                       quote = "", col.names = c("type", "text"))
unlink(temp)  # Delete temporary file

# Convert the label column into a factor (categorical variable)
sms_data$type <- factor(sms_data$type, levels = c("ham", "spam"))

#  Preprocess and Clean the Text Data
# This step is crucial for improving model accuracy.
prep_corpus <- function(x) {
  corp <- VCorpus(VectorSource(x))                    
  corp <- tm_map(corp, content_transformer(tolower))  
  corp <- tm_map(corp, removeNumbers)                
  corp <- tm_map(corp, removePunctuation)             
  corp <- tm_map(corp, removeWords, stopwords("en"))  
  corp <- tm_map(corp, stemDocument)                  
  corp <- tm_map(corp, stripWhitespace)               
  corp
}

# Apply the cleaning function to the entire SMS text dataset
corpus_clean <- prep_corpus(sms_data$text)

# Create a Document-Term Matrix (DTM)
# The DTM converts text data into a table of word frequencies.
dtm <- DocumentTermMatrix(corpus_clean)

# Reducing Noise
dtm <- removeSparseTerms(dtm, 0.995)

# Helper function: Convert numeric word counts into "yes"/"no" indicators.
convert_counts <- function(m) {
  m <- ifelse(m > 0, "yes", "no")
  m <- as.data.frame(m, stringsAsFactors = TRUE)
  for (j in seq_len(ncol(m))) m[[j]] <- factor(m[[j]], levels = c("no", "yes"))
  m
}

# Split Data into Training and Test Sets
# We use 75% of the data for training and 25% for testing.
# The split is random to ensure balanced representation of ham and spam.
set.seed(123)
n <- nrow(dtm)
train_idx <- sample(seq_len(n), size = floor(0.75 * n))
test_idx  <- setdiff(seq_len(n), train_idx)

# Create training and test sets for both features and labels
dtm_train <- dtm[train_idx, ]
dtm_test  <- dtm[test_idx, ]
y_train <- sms_data$type[train_idx]
y_test  <- sms_data$type[test_idx]

# Convert the word frequencies to "yes"/"no" form for both train and test sets
X_train <- convert_counts(as.matrix(dtm_train))
X_test  <- convert_counts(as.matrix(dtm_test))

# 6) Train the Naive Bayes Model
# Laplace smoothing (laplace = 1) helps handle words that appear in test data but not training data.
sms_classifier <- naiveBayes(X_train, y_train, laplace = 1)

#  Predict and Evaluate the Model
# Predict labels for the test dataset
pred_test <- predict(sms_classifier, X_test)

# Create a confusion matrix to compare actual vs predicted labels
conf_mat <- table(Predicted = pred_test, Actual = y_test)
print(conf_mat)

# Calculate and display overall accuracy
accuracy <- sum(diag(conf_mat)) / sum(conf_mat)
cat("Model Accuracy:", round(accuracy * 100, 2), "%\n")

# Test the Model with a Custom Message
# This function takes any new text message, cleans it using the same steps and predicts whether it is spam or not.
predict_message <- function(text_vec, classifier, train_dtm_terms) {
  corp <- prep_corpus(text_vec)
  msg_dtm <- DocumentTermMatrix(corp, control = list(dictionary = train_dtm_terms))
  msg_df <- convert_counts(as.matrix(msg_dtm))
  predict(classifier, msg_df)
}

# Example 1: Predict if a custom message is spam or ham
test_message <- "Good morning have a nice day"
test_pred <- predict_message(test_message, sms_classifier, Terms(dtm_train))
cat("Message:", test_message, "\nPrediction:", as.character(test_pred), "\n")
