
#Dataset: Red Wine Quality

#libraries needed
library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(corrgram) # Correlograms
library(lattice) #required for nearest neighbors
library(FNN) # nearest neighbors techniques
library(pROC) # to make ROC curve

wine <-("/Users/Aishu/Desktop/winequality-red.csv")

head(wine)
summary(wine$quality)
table(wine$quality)

#Regression Approaches to Quality Score
#linear regression
linear_quality = lm(quality ~ fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+pH+sulphates+alcohol, data=wine)
summary(linear_quality)

#Multiple R-squared:  0.3606,	Adjusted R-squared:  0.3561 F-statistic: 81.35 on 11 and 1587 DF,  p-value: < 2.2e-16

corrgram(wine, lower.panel=panel.shade, upper.panel=panel.ellipse)

#Single-variable linear regression to improve the model

linear_quality_1 = lm(quality ~ alcohol, data = wine)
summary(linear_quality_1)

#Multiple R-squared:  0.2267,	Adjusted R-squared:  0.2263 F-statistic: 468.3 on 1 and 1597 DF,  p-value: < 2.2e-16

#Four variable linear regression to improve the model

linear_quality_4 = lm(quality ~ alcohol + volatile.acidity + citric.acid + sulphates, data = wine)
summary(linear_quality_4)

#visualization

linear_quality.res = resid(linear_quality) # gets residuals
linear_quality_1.res = resid(linear_quality_1)
linear_quality_4.res = resid(linear_quality_4)

plot(wine$alcohol, linear_quality.res) # plot residuals against alcohol variable
points(wine$alcohol, linear_quality_1.res, col="red") # add the residuals for 1-dimension
points(wine$alcohol, linear_quality_4.res, col="blue") # add residuals for 4 dimension

#Logistic Regression: GLM

glm_quality_1 = glm(quality~alcohol, data=wine, family=gaussian(link="identity"))
summary(glm_quality_1)
#Number of Fisher Scoring iterations: 2

glm_quality_2 = glm(quality~alcohol, data=wine, family=gaussian(link="log"))
summary(glm_quality_2)
#AIC: 3451.9,Number of Fisher Scoring iterations: 4

glm_quality_3 = glm(quality~alcohol+sulphates,data=wine,family=poisson(link="identity"))
summary(glm_quality_3)
#AIC: 5877.1,Number of Fisher Scoring iterations: 4

#visualizing the plots

glm_quality_1.res = resid(glm_quality_1) # gets residuals
glm_quality_2.res = resid(glm_quality_2)
glm_quality_3.res = resid(glm_quality_3)

plot(wine$alcohol, glm_quality_1.res) # plot residuals against alcohol variable
points(wine$alcohol, glm_quality_2.res, col="red") # add the residuals for 1-dimension
points(wine$alcohol, glm_quality_3.res, col="blue") # add res

#3. K-nearest neighbours regression

knn10 = knn.reg(train=wine[,1:11], test=wine[,1:11], y=wine$quality, k =10) 
knn20 = knn.reg(train=wine[,1:11],test=wine[,1:11], y = wine$quality, k=20)
plot(wine$alcohol,wine$quality)
points(wine$alcohol,knn10$pred,col="red")
points(wine$alcohol,knn20$pred,col="blue")

#Classification

wine$poor <- wine$quality <= 4
wine$okay <- wine$quality == 5 | wine$quality == 6
wine$good <- wine$quality >= 7
head(wine)
summary(wine)

#logistic Function Fit

log1_good = glm(good~alcohol, data=wine, family=binomial(link="logit"))
log2_good = glm(good~alcohol + volatile.acidity + citric.acid + sulphates,data=wine,family=binomial(link="logit"))

Slog1_good <- pnorm(predict(log1_good))
Slog2_good <- pnorm(predict(log2_good))

roc1 <- plot.roc(wine$good,Slog1_good,main="",percent=TRUE, ci=TRUE, print.auc=TRUE)
roc1.se <- ci.se(roc1,specificities=seq(0,100,5))
plot(roc1.se,type="shape", col="grey")

roc2 <-plot.roc(wine$good,Slog2_good,main="",percent=TRUE, ci=TRUE, print.auc=TRUE)
roc2.se <- ci.se(roc2,specificities=seq(0,100,5))
plot(roc2.se,type="shape", col="blue")

#K-nearest classification

class_knn10 = knn(train=wine[,1:11], test=wine[,1:11], cl=wine$good, k =10) 
class_knn20 = knn(train=wine[,1:11],test=wine[,1:11], cl = wine$good, k=20)
table(wine$good,class_knn10)
table(wine$good,class_knn20)

#decision trees

library(rpart) #for trees
tree1 <- rpart(good ~ alcohol + sulphates, data = wine, method="class")
summary(tree1)

library(rpart.plot) # plotting trees
library(caret)
rpart.plot(tree1)
pred1 <- predict(tree1,newdata=wine,type="class")

tree2 <- rpart(good ~ alcohol + volatile.acidity + citric.acid + sulphates, data = wine, method="class")
summary(tree2)
rpart.plot(tree2)
pred2 <- predict(tree2,newdata=wine,type="class")

table(wine$good,pred1)
table(wine$good,pred2)
