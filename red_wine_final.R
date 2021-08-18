library(gRbase)
library(gRim)
library(bnlearn)
library(caTools)
library(gRain)
library(caret)
library(visNetwork)
library(formattable)

#import the dataset
wine=read.csv("C:/Users/TOSHIBA/Desktop/PM_Project/wine_quality/winequality-red.csv", header = TRUE,sep = ';')
attach(wine)
str(wine)
summary(wine)

#Discretize with Hartemink's Method (except "quality")
dwine = discretize(wine[1:11], method = "hartemink",
                   breaks = 3,ibreaks = 20, idisc = "quantile") #

for (i in names(dwine))
  levels(dwine[, i]) = c("LOW", "AVG", "HIGH")


str(dwine)

#convert quality levels into factors
dwine$quality<-as.factor(wine$quality)
summary(dwine)


###Descriptives#####
######################################################

#quality levels bar chart
ggplot(dwine, aes(x = quality)) +
  geom_bar() +
  xlab("quality") +
  theme(axis.text.x = element_text(hjust = 1))
# contingency charts

#quality levels vs alcohol
### quality tends to increase with alcohol
ggplot(dwine, aes(x = quality)) +
  geom_bar(aes(y = (..count..)/sum(..count..))) +
  xlab("quality") +
  ggtitle("Quality by Alcohol Levels")+
  scale_y_continuous(labels = scales::percent, name = "Proportion")+ 
  facet_grid(~ alcohol) +
  theme(axis.text.x = element_text(hjust = 1),plot.title = element_text(hjust = 0.5))


########## Plotting function#######

plot.network <- function(structure, ht = "1000px"){
  nodes.uniq <- unique(c(structure$arcs[,1], structure$arcs[,2]))
  nodes <- data.frame(id = nodes.uniq,
                      label = nodes.uniq,
                      color = "darkturquoise",
                      shadow = TRUE)
  edges <- data.frame(from = structure$arcs[,1],
                      to = structure$arcs[,2],
                      arrows = "to",
                      smooth = TRUE,
                      shadow = TRUE,
                      color = "black")
  return(visEdges(visNetwork(nodes, edges, height = ht, width = "100%"),physics = FALSE, smooth = FALSE)%>%
    visOptions(highlightNearest = TRUE, nodesIdSelection = TRUE) %>%
    visLayout(randomSeed = 123))
  
}  

#LEARNING THE STRUCTURE ### 

#creating whitelist and blacklist for learning with prior knowledge:
whitelist=matrix(c(
  "alcohol","quality",
  "volatile.acidity","quality",
  "residual.sugar","density",
  "alcohol","residual.sugar",
  "sulphates","quality",
  "pH","sulphates",
  "total.sulfur.dioxide","free.sulfur.dioxide"),,2,byrow = TRUE)
colnames(whitelist)=c("from","to")
whitelist


#Grow-Shrink
learn_gs<-function(data,whitelist=NULL,blacklist=NULL){
  bn <- gs(data, test="mi",whitelist = whitelist,blacklist = blacklist)
  return(bn)}

#Hill-Climbing
learn_hc<-function(data,whitelist=NULL,blacklist=NULL){
  bn <- hc(data,whitelist = whitelist,blacklist = blacklist)
  return(bn)}

###the Hybrid structure algorithms
#MMHC
#(mmhc() is simply rsmax2() with restrict set to mmpc and maximize set to hc.)

learn_mmhc<-function(data,whitelist=NULL,blacklist=NULL){
  bn <- rsmax2(data, restrict = "mmpc",maximize = "hc",whitelist = whitelist,blacklist = blacklist)
  return(bn)}

##structures learned from the data:
gs<-learn_gs(dwine)
hc<-learn_hc(dwine)
mmhc<-learn_mmhc(dwine)

##structures learned with knowledge:
gs_knowledge<-learn_gs(dwine,whitelist = whitelist)
hc_knowledge<-learn_hc(dwine,whitelist = whitelist)
mmhc_knowledge<-learn_mmhc(dwine,whitelist = whitelist)

## structure visualizations ##
plot.network(gs)
plot.network(gs_knowledge)

plot.network(hc)
plot.network(hc_knowledge)

plot.network(mmhc)
plot.network(mmhc_knowledge)

######## PARAMETER ESTIMATION ##################

#splitting the data set
set.seed(123)
split= sample.split(dwine$quality, SplitRatio = 4/5)

train_set= subset(dwine, split== TRUE)
test_set= subset(dwine, split== FALSE)

#model fit, and confusion matrices
#with validation
fit_validate<-function(bn){
  model=bn.fit(bn,data=train_set,method='bayes')
  m_preds=bnlearn:::predict.bn.fit(model,"quality",test_set)
  cf=confusionMatrix(m_preds,test_set$quality, positive = "Yes")
  cf
}

#models estimated from structures learned from the data
#Grow-Shrink
#bn.gs is an undirected graph and must be extended into a DAG with cextend() 
gs_ = cextend(gs)
fit_validate(gs_)
#Hill-Climbing
fit_validate(hc)
#MMHC
fit_validate(mmhc)

#models estimated from structures learned based on prior knowledge
#Grow-Shrink
#bn.gs is an undirected graph and must be extended into a DAG with cextend() 
gs_k = cextend(gs_knowledge)
fit_validate(gs_k)
#Hill-Climbing
fit_validate(hc_knowledge)
#MMHC
fit_validate(mmhc_knowledge)

## Confusion Matrix Statistics ##
acc_scores1=c(fit_validate(gs_)$overall['Accuracy'],fit_validate(hc)$overall['Accuracy'],fit_validate(mmhc)$overall['Accuracy'])
acc_scores2=c(fit_validate(gs_k)$overall['Accuracy'],fit_validate(hc_knowledge)$overall['Accuracy'],fit_validate(mmhc_knowledge)$overall['Accuracy'])
accuracy_scores=data.frame(acc_scores1,acc_scores2)
colnames(accuracy_scores)=c('w/o knowledge',"w/ knowledge")
rownames(accuracy_scores)=c('Grow-Shrink','Hill-Climbing','MMHC')
accuracy_scores
formattable(accuracy_scores,
            list(`Indicator Name` = formatter("span", style = ~ style(color = "grey",font.weight = "bold"))))

#with cross-validation
fit_cv<-function(data,bn,runs){
  model=bn.cv(data=data, bn = bn ,fit = "bayes", runs=runs)
  
}
#models estimated from structures learned from the data
#Grow-Shrink
gs_cv=fit_cv(dwine,gs,10)
#Hill-Climbing
hc_cv=fit_cv(dwine,hc,10)
###the Hybrid structure algorithms
mmhc_cv=fit_cv(dwine,mmhc,10)

#models estimated from structures learned based on prior knowledge
#Grow-Shrink
#first adding directions to arcs
gs_cv_k=fit_cv(dwine,gs_knowledge,10)
#Hill-Climbing
hc_cv_k=fit_cv(dwine,hc_knowledge,10)
###the Hybrid structure algorithms
mmhc_cv_k=fit_cv(dwine,mmhc_knowledge,10)


#plotting losses
plot(gs_cv,hc_cv,mmhc_cv,xlab=c('Grow-Shrink','Hill-Climbing','MMHC'))
plot(gs_cv_k,hc_cv_k,mmhc_cv_k,xlab=c('Grow-Shrink','Hill-Climbing','MMHC'))


losses1=c(mean(loss(gs_cv)),mean(loss(hc_cv)),mean(loss(mmhc_cv)))
losses2=c(mean(loss(gs_cv_k)),mean(loss(hc_cv_k)),mean(loss(mmhc_cv_k)))
losses=data.frame(losses1,losses2)
colnames(losses)=c('w/o knowledge',"w/ knowledge")
rownames(losses)=c('Grow-Shrink','Hill-Climbing','MMHC')
losses

formattable(losses,
            list('Indicator Name' = formatter("span", style = ~ style(color = "grey",font.weight = "bold"))))

## Model Averaging for final model ##


#hc with bic

models_bic <- boot.strength(data = train_set,R=1000,algorithm = 'hc',algorithm.args = list(score='bic',whitelist=whitelist))
plot(models_bic)# inspecting the threshold
best_model_bic <- averaged.network(models_bic,threshold = 0.85)
fitted_model_bic <- bn.fit(best_model_bic,data = train_set,method = 'bayes')
model_y_preds=bnlearn:::predict.bn.fit(fitted_model_bic,"quality",test_set)
Learned_Bayesian_network <- plot.network(best_model_bic)
conf_bic=confusionMatrix(model_y_preds,test_set$quality)
conf_bic
Learned_Bayesian_network
#hc with aic
models_aic <- boot.strength(data = train_set,R=1000,algorithm = 'hc',algorithm.args = list(score='aic',whitelist=whitelist))
best_model_aic <- averaged.network(models_aic,threshold = 0.85)
fitted_model_aic <- bn.fit(best_model_aic,data = train_set,method = 'bayes')
avg_model_y_preds=bnlearn:::predict.bn.fit(fitted_model_bic,"quality",test_set)
Learned_Bayesian_network <- plot.network(best_model_aic)
conf_aic=confusionMatrix(avg_model_y_preds,test_set$quality)
conf_aic

accuracy=c(conf_bic$overall['Accuracy'],conf_aic$overall['Accuracy'])
losses2=c(mean(loss(gs_cv_k)),mean(loss(hc_cv_k)),mean(loss(mmhc_cv_k)))
accuracy=data.frame(accuracy)
colnames(accuracy)=c('Accuracy')
rownames(accuracy)=c('Hc_BIC','Hc_AIC')
formattable(accuracy)

# Confusion Matrix statistics of the final model
F1=conf_aic[["byClass"]][ , "F1"]
Sensitivity=conf_aic[["byClass"]][ , "Sensitivity"]
Presicion=conf_aic[["byClass"]][ , "Precision"]
Recall=conf_aic[["byClass"]][ , "Recall"]
Balanced_acc=conf_aic[["byClass"]][ , "Balanced Accuracy"]
final_stats=data.frame(F1,Sensitivity,Presicion,Recall,Balanced_acc)
formattable(final_stats)

#### INFERENCE ####

fitted = bn.fit(best_model_aic, dwine, method = "bayes")# fitting the model to the whole dataset
fitted_grain=as.grain(fitted ) #jtree

#quality_marginal probabilities
querygrain(fitted_grain,nodes ="quality",type='marginal')

#set the evidence
ev1 = setFinding(fitted_grain, nodes = "alcohol", states = "HIGH")
ev2 = setFinding(fitted_grain, nodes = "alcohol", states = "LOW")
#Conditional probabilities of quality levels given alcohol
querygrain(ev1,nodes = "quality",type = 'conditional')

#Conditional probabilities of quality levels given sulphates
ev3 = setFinding(fitted_grain, nodes = "sulphates", states = "LOW")
ev4 = setFinding(fitted_grain, nodes = "sulphates", states = "HIGH")
querygrain(ev4,nodes = "quality",type = 'conditional')

#Conditional probabilities of the quality levels given volatile acidity
ev5 = setFinding(fitted_grain, nodes = "volatile.acidity", states = "LOW")
ev6 = setFinding(fitted_grain, nodes = "volatile.acidity", states = "HIGH")
querygrain(ev6,nodes = "quality",type = 'conditional')


#Conditional probabilities of residual sugar given alcohol
querygrain(fitted_grain,nodes ="residual.sugar",type='marginal')
querygrain(ev1,nodes = "residual.sugar",type = 'conditional')

#Conditional probabilities of density given alcohol
querygrain(fitted_grain,nodes ="density",type='marginal')
querygrain(ev2,nodes = "density",type = 'conditional')



############################################################################



