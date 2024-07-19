# Load necessary libraries
library(tidyverse)
library(caret)
library(glmnet)
library(GGally)
library(ggplot2)
library(doBy)
library(FactoMineR)
library(factoextra)
library(corrplot)
library(RColorBrewer)
library(aod)
library(ROCR)
library(rpart)
library(rpart.plot)
library(randomForest)
library(reshape2)

#SET MY PATH
setwd('C:\\Assignment')

#import data
df=read.csv("Airline_customer_satisfaction.csv")
#view data
head(df)


#function to get the type, unique items and the NA count of every column in the dataframe
stats=function(df){
  l=c()
  for(i in 1:length(df)){
    type=class(df[,i])
    unique=length(unique(df[,i]))
    sum_null=sum(is.na(df[,i]))
    l=append(l,c(colnames(df)[i],type,unique,sum_null))
  }
  df_stats=matrix(l,ncol=4,byrow=T)
  colnames(df_stats)=c('column','type','unique','sum_null')
  return (df_stats)
}
#applying the stats function 
stats(df)


#change the NA values with the same value of arrival delay values
df$Arrival.Delay.in.Minutes=ifelse(is.na(df$Arrival.Delay.in.Minutes),df$Departure.Delay.in.Minutes,df$Arrival.Delay.in.Minutes)


#re-check the NA values 
stats(df)

#data summary
summary(df)

#pie chart plot by proportion of satisfaction
data.pie=data.frame(df%>%
                      group_by(satisfaction)%>%
                      summarise(count=n()))
ggplot(data.pie, aes(x="", y=count, fill=satisfaction)) +
  geom_bar(stat="identity", width=1) +
  coord_polar("y", start=0) +
  geom_text(aes(label = count),
            position = position_stack(vjust = 0.5)) +
  theme_void()


#histogram chart of satisfaction by the flight distance
ggplot(df, aes(x=Flight.Distance, fill=satisfaction)) +
  geom_histogram(alpha=0.5,bins=300,position='identity')

#histogram chart of customer type by the flight distance
ggplot(df, aes(x=Flight.Distance, fill=Customer.Type)) +
  geom_histogram(alpha=0.5,bins=300,position='identity')


#density chart of satisfaction by age
ggplot(df, aes(x=Age, color=satisfaction)) +geom_density()

#bar chart of satisfaction by the class of customers
ggplot(df,aes(x=Class,fill=satisfaction))+geom_bar()



# Define the discrete function
discrete = function(x) {
  length(unique(x)) <= 6
}

# Apply the discrete function to each column in the dataframe
sapply(df, discrete)


pca_df=df[,sapply(df,discrete)]
head(pca_df)

pca_df[]=lapply(pca_df,factor)
pca_df[]=lapply(pca_df,as.integer)
head(pca_df)

pca_df2=summaryBy(.~ satisfaction+Gender+Customer.Type+Type.of.Travel+Class,data = pca_df,FUN=c(mean),keep.names = TRUE)
dim(pca_df2)

c=cor(pca_df2)
corrplot(c, type="upper", order="hclust",col=brewer.pal(n=8, name="RdBu"))

acp=PCA(pca_df2,quali.sup = c(1:5),graph=F)
acp$eig

fviz_screeplot(acp, ncp=10)

fviz_pca_ind(acp)

fviz_pca_var(acp)

# Split the data into training and testing sets
set.seed(123)  # For reproducibility
train_index = createDataPartition(df$satisfaction, p = 0.8, list = FALSE)
df_train = df[train_index, ]
df_test = df[-train_index, ]

# Convert the satisfaction column to a binary factor in training and testing sets
df_train$satisfaction = as.factor(ifelse(df_train$satisfaction == 'satisfied', 1, 0))
df_test$satisfaction = as.factor(ifelse(df_test$satisfaction == 'satisfied', 1, 0))

# Check the structure of the training data
str(df_train)

#glm model with the training dataframe
modele_glm=glm(satisfaction ~ .,family='binomial',df_train)


#modele summary
summary(modele_glm)

#stepwise function
step(modele_glm,direction='both')

n_terms <- length(coef(modele_glm))  # Number of terms in the model
for (i in 1:n_terms) {
  for (j in i:n_terms) {
    w = wald.test(b = coef(modele_glm), Sigma = vcov(modele_glm), Terms = i:j)
    if (w$result$chi2[3] > 0.05) {
      print(w$result$chi2[3])
      print(i)
      print(j)
    }
  }
}

# Predict the levels of satisfaction using the predict function with the testing dataframe
pred = predict(modele_glm, df_test, type = 'response')

# Change the predicted values from probabilities to binary (0 or 1)
pred_glm = ifelse(pred > 0.5, 1, 0)

# Calculate the accuracy and the precision of the model
tab = table(df_test$satisfaction, pred_glm)
accuracy_glm = (tab[1, 1] + tab[2, 2]) / sum(tab)
print(paste("The accuracy =", accuracy))

# Print the confusion matrix
print(tab)

precision_glm = tab[2,2]/(tab[1,2]+tab[2,2])
print(paste("The precision = ", precision_glm))

recall_glm = tab[2, 2] / (tab[2, 1] + tab[2, 2])
f1_glm = 2 * ((precision_glm * recall_glm) / (precision_glm + recall_glm))
print(paste("The recall= ", recall_glm))
print(paste("The f1-score= ", f1_glm))

#perform the roc plot on the glm model
predicted=prediction(pred,df_test$satisfaction)
roc=performance(predicted, "tpr","fpr")
plot(roc)



#decision tree parameters 
rc=rpart.control(minsplit = 20,cp=0.008,minbucket=1)
#performing a decision tree using training dataframe
tree=rpart(satisfaction~.,data=df_train,method="class",control = rc)
#predict satisfactions classes using the decision tree
pred_tree=predict(tree,df_test,type = "class")
#plot the tree
rpart.plot(tree,extra = 106)

##calculate the accuracy and the precision
tab=table(df_test$satisfaction,pred_tree)
accuracy_tree = (tab[1,1]+tab[2,2])/sum(tab)
print(paste("The accuracy = ", accuracy_tree))
precision_tree = tab[2,2]/(tab[1,2]+tab[2,2])
print(paste("The precision = ", precision_tree))
recall_tree = tab_tree[2, 2] / (tab_tree[2, 1] + tab_tree[2, 2])
f1_tree = 2 * ((precision_tree * recall_tree) / (precision_tree + recall_tree))
print(paste("The recall =", recall_tree))
print(paste("The F1-Score= ", f1_tree))

printcp(tree)

#perform the roc plot of the decision tree
pred=predict(tree,df_test,type = "prob")[,2]
predicted=prediction(pred,df_test$satisfaction)
roc=performance(predicted, "tpr","fpr")
plot(roc)


#calculate the best number of variables for the random forst with tuneRF function
tuneRF(df_train[setdiff(colnames(df_train),"satisfaction")],df_train$satisfaction,stepFactor=1.5,mtryStart=2, improve=0.01,ntree=20,trace=F) 

#apply a random forest model with the train dataframe
modele_rf=randomForest(satisfaction~.,data=df_train,ntree=20,mtry=9)
#predict the satisfaction levels with predict function on the test dataframe
pred=predict(modele_rf,df_test,type="class")
#calculate the accuracy and the precision of the model
tab=table(df_test$satisfaction,pred)
accuracy_rf = (tab[1,1]+tab[2,2])/sum(tab)
print(paste("The accuracy = ", accuracy_rf))
precision_rf = tab[2,2]/(tab[1,2]+tab[2,2])
print(paste("The precision = ", precision_rf))
recall_rf = tab[2, 2] / (tab[2, 1] + tab[2, 2])
f1_rf = 2 * ((precision_rf * recall_rf) / (precision_rf + recall_rf))
print(paste("The Recall= ", recall_rf))
print(paste("The F1-Score= ", f1_rf))

#perform the roc plot of the random forest model
pred=predict(modele_rf,df_test,type="prob")[,2]
predicted=prediction(pred,df_test$satisfaction)
roc=performance(predicted, "tpr","fpr")
plot(roc)


# Data frame to store the metrics for each model
metrics_df <- data.frame(
  Model = c("Logistic Regression", "Decision Tree", "Random Forest"),
  Accuracy = c(accuracy_glm, accuracy_tree, accuracy_rf),
  Precision = c(precision_glm, precision_tree, precision_rf),
  Recall = c(recall_glm, recall_tree, recall_rf),
  F1_Score = c(f1_glm, f1_tree, f1_rf)
)

metrics_df

# Melt the data frame for plotting
library(reshape2)
metrics_melted <- melt(metrics_df, id.vars = "Model")

# Plot the performance metrics using an earth tone color palette
ggplot(metrics_melted, aes(x = Model, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Model Performance Comparison", y = "Score", fill = "Metric") +
  theme_minimal() +
  scale_fill_manual(values = c("Accuracy" = "#db6551", "Precision" = "#e6896b", "Recall" = "#e5a186", "F1_Score" = "#f3c3b0"))

