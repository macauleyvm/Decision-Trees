library(caret)
library(rattle)
library(pROC)
library(randomForest)


#get German credit data
data(GermanCredit)
gc_data <- data.frame(GermanCredit)
rm(GermanCredit)
#moved the label variable to the final column for visual clarity 
gc_data <- gc_data[,c(1:9,11:62,10)]


#1----
set.seed(345)
#create a data partition for the training set (70%) and test set (30%) 
train_index <- createDataPartition(gc_data[,ncol(gc_data)], p=0.7, list=FALSE)
gc_train <- gc_data[train_index,]
gc_test <- gc_data[-train_index,]
#set the control that we are going to use for growing and pruning the tree using 10 fold cross validation 
fit_cont <- trainControl(method = "repeatedcv", number=10)

#grow and then prune the tree using CV.
gc_tt_rpart <- train(gc_train[,-ncol(gc_data)],gc_train[,ncol(gc_data)], method="rpart", tuneLength=5, trControl=fit_cont)
gc_tt_rpart$finalModel
#plot the pruned tree
fancyRpartPlot(gc_tt_rpart$finalModel)
#see the test prediction error rate of the final pruned tree
pred_prune <- predict(gc_tt_rpart$finalModel, gc_test[,-ncol(gc_test)], type="class")
mean(pred_prune==gc_test[,ncol(gc_test)])
#71% chance of getting the true prediction of the label from using the pruned tree so we have an error rate of 29%
fancyRpartPlot(gc_tt_rpart$finalModel)
gc_tt_rpart$bestTune
plot(gc_tt_rpart)


#2----
#use rf to determine the optimal sample size that we need to take at each node and this should make our predictions more accurate
gc_tt_rf <- train(Class ~., data=gc_train, method="rf", metric="Accuracy", trControl=fit_cont ,tuneLength=5)
#rf_fit = randomForest(Class~., data=gc_train, importance=TRUE, ntree=1000, replace=TRUE)
#mtry=8
#plot the error rate for the RF samples
plot(gc_tt_rf, main="Error Rate for Diffrent Random Forest Sample Sizes (m)",
     xlab="Random Forest Sample Size (m)", ylab="Cross Validation Accuracy Result")
#final model selected is 16 as has the highest accuracy
gc_tt_rf$finalModel
#get the prediction accuracy for for RF on the test set 
pred_rf <- predict(gc_tt_rf$finalModel, gc_test[,-ncol(gc_test)], type="class")
mean(pred_rf==gc_test[,ncol(gc_test)])
#test accuracy 0.77, higher than the pruned tree

#plot the most important variables from our RF model where m=16
plot(varImp(gc_tt_rf,scale=FALSE))


#3----
#calculate ROC curve need the probability for each prediction classification
pred_rf_prob <- predict(gc_tt_rf$finalModel, gc_test[,-ncol(gc_test)], type="prob")
pred_prune_prob <- predict(gc_tt_rpart$finalModel, gc_test[,-ncol(gc_test)], type="prob")
#calculate roc values for RF and PDT
rf_roc <- roc(gc_test$Class, pred_rf_prob[,1])
pdt_roc <- roc(gc_test$Class, pred_prune_prob[,1])

#plotting both ROC values on the graph 
plot(rf_roc,main="ROC Curve for the Test Data Set for Random Foresting and Pruned Decision Tree")
lines(pdt_roc, col="blue")
legend("bottomright", legend=c("Random Forest", "Pruned Decision Tree"), col=c("black", "blue"), lty=c(1,1),cex=1.5)
#results show that RF is better for prediction as it has a higher area under the curve

