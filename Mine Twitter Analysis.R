#########################################
#Load Data
#The data comes from victorneo. victorneo shows how to do sentiment 
#analysis for tweets using Python. Here, I will demonstrate how to do it in R.
#########################################

setwd("~./Tweets_Text_Sentiment")


happy <- readLines("./happy.txt")
sad <- readLines("./sad.txt")

happy_test <- readLines("./happy_test.txt")
sad_test <- readLines("./sad_test.txt")


tweet <- c(happy,sad)
tweet_test <- c(happy_test,sad_test)

tweet_all <- c(tweet,tweet_test)

sentiment <- c(rep("happy",length(happy)),
               rep("sad",length(sad)))

sentiment_test <- c(rep("happy",length(happy_test)),
                    rep("sad",length(sad_test))) 

sentiment_all <- as.factor(c(sentiment,sentiment_test))
View(sentiment_all)

install.packages("RTextTools")
library(RTextTools)
library(tm)


# Naive Bayes
install.packages("e1071")
library(e1071)
mat <- create_matrix(tweet_all,language = "english",
                     removeStopwords = FALSE,removeNumbers = TRUE,
                     stemWords = FALSE,tm::weightTfIdf)
mat <- as.matrix(mat)

classifier <- naiveBayes(mat[1:160,],as.factor(sentiment_all[1:160]))
predicted <- predict(classifier,mat[161:180,]); predicted

table(sentiment_test,predicted)
recall_accuracy(sentiment_test,predicted)


# the other method 

mat = create_matrix(tweet_all,language = "english",
                    removeStopwords = FALSE,removeNumbers = TRUE,
                    stemWords = FALSE,tm::weightTfIdf)

container <- create_container(mat,as.numeric(sentiment_all),
                              trainSize = 1:160,testSize = 161:180,virgin = FALSE)

models <-  train_models(container, algorithms = c("MAXENT",
                                                  "SVM",
                                                  #"GLMNET",
                                                  "BOOSTING",
                                                  "SLDA","BAGGING",
                                                  "RF",# "NNET",
                                                  "TREE"
                                                  ))

# Test the Models

results = classify_models(container,models)
table(as.numeric(as.numeric(sentiment_all[161:180])),results[,"FORESTS_LABEL"])
recall_accuracy(as.numeric(as.numeric(sentiment_all[161:180])),results[,"FORESTS_LABEL"])


#Here we also want to get the formal test results, including:

#analytics@algorithm_summary: Summary of precision, recall, f-scores, and accuracy sorted by topic code for each algorithm
#analytics@label_summary: Summary of label (e.g. Topic) accuracy
#analytics@document_summary: Raw summary of all data and scoring
#analytics@ensemble_summary: Summary of ensemble precision/coverage. Uses the n variable passed into create_analytics()


###### Formal tests 

analytics <- create_analytics(container, results)
summary(analytics)


head(analytics@algorithm_summary)
head(analytics@label_summary)
head(analytics@document_summary)
analytics@ensemble_summary   # Ensemble Agreement

# Cross Validation 

N = 3
cross_SVM <- cross_validate(container,N,"SVM")
cross_GLMNET <- cross_validate(container,N,"GLMNET")
cross_MAXENT <- cross_validate(container,N,"MAXENT") 

#We Can find that compared with naive Bayes, 
#the other algorithms did a much better job to achieve a recall accuracy higher than 0.95. 

rm(list=ls())