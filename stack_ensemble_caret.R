# Ensemble models in caret
#-------------------------
library(dplyr)
library(caret)
library(data.table)
library(mltools)
library(caretEnsemble)

# Working directory
script_name <- 'stack_ensemble_caret.R'
ruta <- gsub(rstudioapi::getActiveDocumentContext()$path,pattern = script_name,replacement = '')
setwd(ruta)

# Loading Data and preprocessing
df<-read.csv("salary.csv") 
df$rk<-as.factor(df$rk)
df.encoded <- one_hot(as.data.table(df)) %>% as.data.frame()

## Train/test
set.seed(1122)
trainIndex <- createDataPartition(df.encoded$sl, p = .8,list=F)
train <- df.encoded[trainIndex,]
test  <- df.encoded[-trainIndex,]

## controls
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,     
                           repeats = 3)

# Grids for hyper parameters
earth_grid=expand.grid(degree = 1:3, nprune = c(2, 6, 10, 12))
cub_grid=expand.grid(committees = c(1,10,50,100), neighbors = c(0,1,5,9))

# Fit models
set.seed(100)
models <- caretList(sl~., 
                    data = train, 
                    trControl=fitControl,
                    tuneList = list(
                      lm=caretModelSpec(method="lm"),
                      earth=caretModelSpec(method="earth",tuneGrid=earth_grid),
                      cubist=caretModelSpec(method="cubist",tuneGrid=cub_grid)
                    )) 
results <- resamples(models)
summary(results)
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)

### Test set error
#------------------
# Linear model
pred_lm<-predict(models$lm,newdata = test)

# MARS model
pred_mr<-predict(models$earth,newdata = test)

# Cubist model
pred_cub<-predict(models$cubist,newdata = test)

# Stacking models
stackControl<-trainControl(method="repeatedcv", 
                           number=10, 
                           repeats=3)
stack.glm <- caretStack(models, 
                        method="glmnet",
                        tuneLength=10,
                        trControl=stackControl)  
pred_ens<-predict(stack.glm,newdata = test)

## Results
df_rs<-data.frame(Models=c("Ensemble","Lm","Mars","Cubist"),
                  RMSE=c(RMSE(pred_ens,test$sl),RMSE(pred_lm,test$sl),RMSE(pred_mr,test$sl),RMSE(pred_cub,test$sl)),
                  MAE=c(MAE(pred_ens,test$sl),MAE(pred_lm,test$sl),MAE(pred_mr,test$sl),MAE(pred_cub,test$sl)),
                  R2=c(R2(pred_ens,test$sl),R2(pred_lm,test$sl),R2(pred_mr,test$sl),R2(pred_cub,test$sl)))
df_rs

