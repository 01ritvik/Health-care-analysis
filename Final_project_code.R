library(MASS) 
library(factoextra)
library(rsample)
library(ROSE)
library(smotefamily)
library(caret)
library(e1071)
library(class)
library(naivebayes)
library(randomForest)
library(mclust)
library(pROC)
library(Boruta)



# Load data set 
data <- read.csv("main.csv")
str(data)




# Changing the class variable into factor
data$o_bullied <- factor(data$o_bullied)
str(data)
summary(data)





# Logistic Regression model (Model No:1)

# Spliting the data set in 70-30 ratio

set.seed(101)
split <- initial_split(data, prop = 0.70, strata = o_bullied)
tr <- training(split)
ts <- testing(split)
write.csv(tr, "LR_initial_train.csv")
write.csv(ts,"LR_initial_test.csv")
barplot(table(tr$o_bullied))

# Feature selection with boruta
boruta_result <- Boruta(o_bullied ~ ., data = tr, doTrace = 2) 
boruta_result
selected_features <- getSelectedAttributes(boruta_result, withTentative = TRUE)
length(selected_features)


# over samplimg the data

over_sampled_data <- ovun.sample(o_bullied ~ ., data = tr[, c(selected_features, "o_bullied")], method = "over")$data
tr = over_sampled_data


# running the logistic regression model on oversampled data

logitModel <- glm(o_bullied ~ ., data = tr[, c(selected_features, "o_bullied")], family = "binomial")
options(scipen = 999)
summary(logitModel)

# testing the resultant model

logitModel.pred <- predict(logitModel, ts[, -90], type = "response")
pred <- factor(ifelse(logitModel.pred >= 0.5, 1, 0))
cm <- table(pred, ts$o_bullied)
cm
accuracy <- round((sum(diag(cm)) / sum(cm)) * 100, 3)
cat("Accuracy: ", accuracy, "%\n")

# plotting the ROC curve and calculating the AUC

roc_score <- roc(ts$o_bullied, logitModel.pred)
roc_curve <- roc.curve(ts$o_bullied, logitModel.pred, plotit = TRUE)
roc_score

# Accuracy of individual class

accuracy_class_0 <- sum(ts$o_bullied == 0 & pred== 0) / sum(ts$o_bullied == 0)
accuracy_class_0
accuracy_class_1 <- sum(ts$o_bullied == 1 & pred== 1) / sum(ts$o_bullied == 1)
accuracy_class_1

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# KNN model (Model No:2)

# Spliting the data set in 70-30 ratio

set.seed(102)
split <- initial_split(data, prop=0.70, strata = o_bullied)
tr <- training(split)
ts <- testing(split)
write.csv(tr, "KNN_initial_train.csv")
write.csv(ts,"KNN_initial_test.csv")
barplot(table(tr$o_bullied))

# Feature selection with PCA

pc <- prcomp(tr[,-ncol(tr)], center = TRUE,scale = TRUE) # PCA excluding last column
summary(pc)

# Over sampling the train data

over_sampled_data <- ovun.sample(o_bullied ~ ., data = data, method = "over")$data
tr = over_sampled_data

# Running the KNN model

k <- 5
knn_model <- knn(train = tr[c(1:51, 90)], test = ts[c(1:51, 90)], cl = tr$o_bullied, k = k)

# testing the resultant model

predicted <- data.frame(predicted = knn_model)
cm <- table(predicted$predicted, ts$o_bullied)
print(cm)
accuracy <- sum(diag(cm)) / sum(cm)
cat("Overall Accuracy: ", round(accuracy * 100, 3), "%\n")

# plotting the ROC curve and measuring the AUC

roc_score=roc.curve(ts[,90], knn_model, plotit = TRUE)
roc_score

# Accuracy of individual class

accuracy_class_0 <- sum(ts$o_bullied == 0 & predicted$predicted == 0) / sum(ts$o_bullied == 0)
accuracy_class_1 <- sum(ts$o_bullied == 1 & predicted$predicted == 1) / sum(ts$o_bullied == 1)
cat("Class 0 Accuracy: ", round(accuracy_class_0 * 100, 3), "%\n")
cat("Class 1 Accuracy: ", round(accuracy_class_1 * 100, 3), "%\n")

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Naive Bayes model (Model No:3)

# Spliting the data set in 70-30 ratio


set.seed(103)
split <- initial_split(data, prop=0.70, strata = o_bullied)
tr <- training(split)
ts <- testing(split)
write.csv(tr, "NB_initial_train.csv")
write.csv(ts,"NB_initial_test.csv")
barplot(table(tr$o_bullied))

# Feature selection with PCA

pc <- prcomp(tr[,-ncol(tr)], center = TRUE,scale = TRUE) 
summary(pc)

# Over sampling the train data

over_sampled_data <- ovun.sample(o_bullied ~ ., data = data, method = "under")$data
tr = over_sampled_data

# Running the Naive Bayes Model

naive_bayes_model <- naive_bayes(o_bullied ~ ., data = tr[c(1:51, 90)])

# testing the resultant model

predicted <- predict(naive_bayes_model, newdata = ts)
cm <- table(predicted, ts$o_bullied)
print(cm)
accuracy <- sum(diag(cm)) / sum(cm)
cat("Overall Accuracy: ", round(accuracy * 100, 3), "%\n")

# plotting the ROC curve and measuring the AUC

roc_score=roc.curve(ts[,90], predicted )
roc_score

# Accuracy of individual class

accuracy_class_0 <- sum(ts$o_bullied == 0 & predicted == 0) / sum(ts$o_bullied == 0)
accuracy_class_1 <- sum(ts$o_bullied == 1 & predicted == 1) / sum(ts$o_bullied == 1)
cat("Class 0 Accuracy: ", round(accuracy_class_0 * 100, 3), "%\n")
cat("Class 1 Accuracy: ", round(accuracy_class_1 * 100, 3), "%\n")


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Support Vector Machine (SVM) Model (Model No:4)

# Splitting the data set in 70-30 ratio

set.seed(104)
split <- initial_split(data, prop=0.70, strata = o_bullied)
tr <- training(split)
ts <- testing(split)
write.csv(tr, "SVM_Iteration_1_initial_train.csv")
write.csv(ts,"SVM_Iteration_1_initial_test.csv")
barplot(table(tr$o_bullied))

# Feature selection with PCA

pc <- prcomp(tr[,-ncol(tr)], center = TRUE,scale = TRUE)
summary(pc)

# Over sampling the train data

over_sampled_data <- ovun.sample(o_bullied ~ ., data = data, method = "over")$data
tr = over_sampled_data

# Running the SVM model

svm_model <- svm(o_bullied ~ ., data = tr[c(1:51, 90)], kernel = "radial", cost = 1, probability = TRUE)

# testing the resultant model

predicted <- predict(svm_model, ts[, -90])
cm <- table(predicted, ts$o_bullied)
print(cm)
accuracy <- sum(diag(cm)) / sum(cm)
cat("Overall Accuracy: ", round(accuracy * 100, 3), "%\n")


# plotting the ROC curve and measuring the AUC


roc_score=roc.curve(ts[,90], predicted )
roc_score






# 2nd iteration of SVM Using Boruta for feature selection 

# Splitting the data set in 70-30 ratio

set.seed(105)
split <- initial_split(data, prop=0.70, strata = o_bullied)
tr <- training(split)
ts <- testing(split)
write.csv(tr, "SVM_Iteration_2_initial_train.csv")
write.csv(ts,"SVM_Iteration_2_initial_test.csv")
barplot(table(tr$o_bullied))

# Feature selection using Boruta


boruta_result <- Boruta(o_bullied ~ ., data = tr, doTrace = 2)
boruta_result
selected_features <- getSelectedAttributes(boruta_result, withTentative = TRUE)
length(selected_features)

# Over sampling the train data

over_sampled_data <- ovun.sample(o_bullied ~ ., data = tr[, c(selected_features, "o_bullied")], method = "over")$data
tr = over_sampled_data

# Running the SVM model 

svm_model <- svm(o_bullied ~ ., data = tr[, c(selected_features, "o_bullied")], kernel = "radial", cost = 1, probability = TRUE)

# testing the resultant model

predicted <- predict(svm_model, ts[, -90])
cm <- table(predicted, ts$o_bullied)
print(cm)
accuracy <- sum(diag(cm)) / sum(cm)

# plotting the ROC curve and measuring the AUC
cat("Overall Accuracy: ", round(accuracy * 100, 3), "%\n")
roc_score=roc.curve(ts[,90], predicted )
roc_score


# Accuracy of individual class

accuracy_class_0 <- sum(ts$o_bullied == 0 & predicted == 0) / sum(ts$o_bullied == 0)
accuracy_class_1 <- sum(ts$o_bullied == 1 & predicted == 1) / sum(ts$o_bullied == 1)
cat("Class 0 Accuracy: ", round(accuracy_class_0 * 100, 3), "%\n")
cat("Class 1 Accuracy: ", round(accuracy_class_1 * 100, 3), "%\n")


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Random Forest model (Model No:5)


# Splitting the data set in 70-30 ratio

set.seed(105)
split <- initial_split(data, prop=0.70, strata = o_bullied)
tr <- training(split)
ts <- testing(split)
write.csv(tr, "RF_Iteration_1_initial_train.csv")
write.csv(ts,"RF_Iteration_1_initial_test.csv")
barplot(table(tr$o_bullied))

# Feature selection with Boruta

pc <- prcomp(tr[,-ncol(tr)], center = TRUE,scale = TRUE) # PCA excluding last column
summary(pc)

# Oversampling the data

over_sampled_data <- ovun.sample(o_bullied ~ ., data = data, method = "over")$data
tr = over_sampled_data

# Running the Random forest model

rf_model <- randomForest(o_bullied ~ ., data = tr[c(1:51, 90)], ntree = 50, mtry = 2)

# testing the resultant model

predicted <- predict(rf_model, ts[, -90])
cm <- table(predicted, ts$o_bullied)
print(cm)
accuracy <- sum(diag(cm)) / sum(cm)
cat("Overall Accuracy: ", round(accuracy * 100, 3), "%\n")

# plotting the ROC curve and measuring the AUC
roc_score=roc.curve(ts[,90], predicted )
roc_score




# second iteration with Boruta 

# Splitting the data set in 70-30 ratio

set.seed(105)
split <- initial_split(data, prop=0.70, strata = o_bullied)
tr <- training(split)
ts <- testing(split)
write.csv(tr, "RF_Iteration_2_initial_train.csv")
write.csv(ts,"RF_Iteration_2_initial_test.csv")
barplot(table(tr$o_bullied))

# Feature selection using Boruta

boruta_result <- Boruta(o_bullied ~ ., data = tr, doTrace = 2)
boruta_result
selected_features <- getSelectedAttributes(boruta_result, withTentative = TRUE)
length(selected_features)

# Oversampling the data

over_sampled_data <- ovun.sample(o_bullied ~ ., data = tr[, c(selected_features, "o_bullied")], method = "over")$data
tr = over_sampled_data

# Running the Random forest model

rf_model <- randomForest(o_bullied ~ ., data = tr[, c(selected_features, "o_bullied")], ntree = 50, mtry = 2)

# testing the resultant model

predicted <- predict(rf_model, ts[, -90])
cm <- table(predicted, ts$o_bullied)
print(cm)
accuracy <- sum(diag(cm)) / sum(cm)
cat("Overall Accuracy: ", round(accuracy * 100, 3), "%\n")

# plotting the ROC curve and measuring the AUC

roc_score=roc.curve(ts[,90], predicted )
roc_score

# Accuracy of individual class

accuracy_class_0 <- sum(ts$o_bullied == 0 & predicted == 0) / sum(ts$o_bullied == 0)
accuracy_class_1 <- sum(ts$o_bullied == 1 & predicted == 1) / sum(ts$o_bullied == 1)
cat("Class 0 Accuracy: ", round(accuracy_class_0 * 100, 3), "%\n")
cat("Class 1 Accuracy: ", round(accuracy_class_1 * 100, 3), "%\n")


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




# Linear Discriminant Analysis (LDA) model (Model No:6)

# Splitting the data set in 70-30 ratio

set.seed(106)
split <- initial_split(data, prop=0.70, strata = o_bullied)
tr <- training(split)
ts <- testing(split)
write.csv(tr, "LDA_initial_train.csv")
write.csv(ts,"LDA_initial_test.csv")
barplot(table(tr$o_bullied))

# Feature selection with PCA

pc <- prcomp(tr[,-ncol(tr)], center = TRUE,scale = TRUE) # PCA excluding last column
summary(pc)

# Oversampling the data
over_sampled_data <- ovun.sample(o_bullied ~ ., data = data, method = "over")$data
tr = over_sampled_data


# Running the LDA model


lda_model <- lda(o_bullied ~ ., data = tr[c(1:51, 90)])

# testing the resultant model

predicted <- predict(lda_model, newdata = ts[, -90])
cm <- table(predicted$class, ts$o_bullied)
print(cm)
accuracy <- sum(diag(cm)) / sum(cm)
cat("Overall Accuracy: ", round(accuracy * 100, 3), "%\n")


# plotting the ROC curve and measuring the AUC


roc_obj <- roc(ts$o_bullied, as.numeric(predicted$class) - 1)  
auc_value <- auc(roc_obj)
plot(roc_obj, main = "ROC Curve for LDA")
cat("AUC: ", round(auc_value, 3), "\n")


# Accuracy of individual class

accuracy_class_0 <- sum(ts$o_bullied == 0 & predicted$class == 0) / sum(ts$o_bullied == 0)
accuracy_class_1 <- sum(ts$o_bullied == 1 & predicted$class == 1) / sum(ts$o_bullied == 1)
cat("Class 0 Accuracy: ", round(accuracy_class_0 * 100, 3), "%\n")
cat("Class 1 Accuracy: ", round(accuracy_class_1 * 100, 3), "%\n")


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#Gaussian Mixture Model (GMM) (Model No:7)

# Splitting the data set in 70-30 ratio

set.seed(107)
split <- initial_split(data, prop=0.70, strata = o_bullied)
tr <- training(split)
ts <- testing(split)
write.csv(tr, "GMM_initial_train.csv")
write.csv(ts,"GMM_initial_test.csv")
barplot(table(tr$o_bullied))

# Feature selection with PCA

pc <- prcomp(tr[,-ncol(tr)], center = TRUE,scale = TRUE)
summary(pc)

# Oversampling the data

over_sampled_data <- ovun.sample(o_bullied ~ ., data = data, method = "over")$data
tr = over_sampled_data


# Running the GMM model

gmm_model <- Mclust(tr[c(1:51, 90)], G = 2)  # Specify the number of components (G = 2 for binary classification)

# testing the resultant model

predicted <- predict(gmm_model, ts[c(1:51, 90)])
cm <- table(predicted$classification, ts$o_bullied)
print(cm)
accuracy <- sum(diag(cm)) / sum(cm)
cat("Overall Accuracy: ", round(accuracy * 100, 3), "%\n")


# plotting the ROC curve and measuring the AUC

roc_obj <- roc(ts$o_bullied, predicted$classification)
auc_value <- auc(roc_obj)
plot(roc_obj, main = "ROC Curve")
cat("AUC: ", round(auc_value, 3), "\n")


# Accuracy of individual class

accuracy_class_0 <- sum(ts$o_bullied == 0 & predicted$classification == 1) / sum(ts$o_bullied == 0)
accuracy_class_1 <- sum(ts$o_bullied == 1 & predicted$classification == 2) / sum(ts$o_bullied == 1)
cat("Class 0 Accuracy: ", round(accuracy_class_0 * 100, 3), "%\n")
cat("Class 1 Accuracy: ", round(accuracy_class_1 * 100, 3), "%\n")




#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# Quadratic Discriminant Analysis (QDA) model (Model No:8)

# Splitting the data set in 70-30 ratio

set.seed(108)
split <- initial_split(data, prop=0.70, strata = o_bullied)
tr <- training(split)
ts <- testing(split)
write.csv(tr, "QDA_initial_train.csv")
write.csv(ts," QDA_initial_test.csv")
barplot(table(tr$o_bullied))


# Feature selection with boruta


boruta_result <- Boruta(o_bullied ~ ., data = tr, doTrace = 2)
boruta_result
selected_features <- getSelectedAttributes(boruta_result, withTentative = TRUE)
length(selected_features)

# Oversampling the data
over_sampled_data <- ovun.sample(o_bullied ~ ., data = tr[, c(selected_features, "o_bullied")], method = "over")$data
tr = over_sampled_data


# Running the QDA model

qda_model <- qda(o_bullied ~ ., data =tr[, c(selected_features, "o_bullied")])

# testing the resultant model

predicted <- predict(qda_model, newdata = ts[, -90])
cm <- table(predicted$class, ts$o_bullied)
print(cm)
accuracy <- sum(diag(cm)) / sum(cm)

# plotting the ROC curve and measuring the AUC

cat("Overall Accuracy: ", round(accuracy * 100, 3), "%\n")
roc_obj <- roc(ts$o_bullied, as.numeric(predicted$class) - 1)  # Convert predicted class to numeric (0 or 1)
auc_value <- auc(roc_obj)
plot(roc_obj, main = "ROC Curve for QDA")
cat("AUC: ", round(auc_value, 3), "\n")

# Accuracy of individual class

accuracy_class_0 <- sum(ts$o_bullied == 0 & predicted$class == 0) / sum(ts$o_bullied == 0)
accuracy_class_1 <- sum(ts$o_bullied == 1 & predicted$class == 1) / sum(ts$o_bullied == 1)
cat("Class 0 Accuracy: ", round(accuracy_class_0 * 100, 3), "%\n")
cat("Class 1 Accuracy: ", round(accuracy_class_1 * 100, 3), "%\n")



