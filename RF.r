#Ex. 15.6 Fit a series of random-forest classifiers to the spam data, to explore the sensitivity to the parameter m. 
#Plot both the oob error as well as the test error against a suitably chosen range of values for m.

#5. Number of Instances: 4601 (1813 Spam = 39.4%)
#6. Number of Attributes: 58 (57 continuous, 1 nominal class label)

library(randomForest)
library(caret)
library(ggplot2)
library(patchwork)
set.seed(1234)  # for reproducibility

#leer spam dataset ( https://web.stanford.edu/~hastie/ElemStatLearn/ )
spam <- read.table('spam.data', head=FALSE)
spam$V58 <- factor(spam$V58) # convert to factor (otherwise randomForest does regression instead of classification)
head(spam)

n <- nrow(spam)
# Number of rows for the training set = 80% of the dataset
n_train <- round(0.80 * n)
train_indices <- sample(1:n, n_train)

spam_train <- spam[train_indices  ,]
spam_test  <- spam[-train_indices ,]

#out vectors
o1 <- c()
o2 <- c()
o3 <- c()
o4 <- c()
pos <- 0
# Train a Random Forest
for(T in seq(from=1,to=301, by =5) ){         # number of trees  
  for(m in c(1, 3, 5, 7, 8, 10, 15, 57 ) ) {  # m
    print(paste0(m,":",T))
    
    spam_model <- randomForest(formula = V58 ~ . , 
                             data = spam_train,  
                             ntree=T,
                             maxnodes=NULL,
                             mtry= m )
                             
    #OOB error matrix
    err <- spam_model$err.rate
    #OOB error
    oob_err <- err[, "OOB"][T] #length is number of trees

    #Now evaluate model performance on a test set

    # Generate predicted classes using the model object
    class_prediction <- predict(object = spam_model,   # model object 
                            newdata = spam_test,  # test dataset
                            type = "class") # return classification labels

    # Calculate the confusion matrix for the test set
    cm <- confusionMatrix(data = unname(class_prediction),       # predicted classes
                      reference = spam_test$V58) 

    # TEST SET ERROR      
    # Compare test set accuracy to OOB accuracy
    test_error <- 1 - cm$overall[1]
    
    pos <- pos+1
    o1[pos] <- oob_err
    o2[pos] <- "OOB Error"
    o3[pos] <- T
    o4[pos] <- m
    pos <- pos+1
    o1[pos] <- test_error
    o2[pos] <- "Test Error"
    o3[pos] <- T
    o4[pos] <- m

  }
}

df <- data.frame(error=o1,error_type=o2, Trees=o3, m=o4)
head(df)

p1 <- ggplot(df, aes(x=Trees, y=error, color=error_type , linetype=error_type) ) +
    geom_line() + facet_wrap(~m, nrow=2) + xlab('Number of Trees') + ylab('Misclassification Error') + labs(color = "",linetype="")

p2 <- ggplot(df, aes(x=Trees, y=error, color=factor(m)) ) +
    geom_line(size=0.4) + facet_wrap(~error_type) + xlab('Number of Trees') + ylab('Misclassification Error') + labs(color = "m (p=57)") + scale_color_brewer(palette="Accent")

p2 / p1

ggsave('Ex.15.6-Madrigal.pdf', height=8, width=9)
