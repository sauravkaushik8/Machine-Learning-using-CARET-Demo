#install.packages("caret")

library("caret")

train<-read.csv("train_u6lujuX_CVtuZ9i.csv",stringsAsFactors = T)

str(train)

'data.frame':	614 obs. of  13 variables:
$ Loan_ID          : Factor w/ 614 levels "LP001002","LP001003",..: 1 2 3 4 5 6 7 8 9 10 ...
$ Gender           : Factor w/ 3 levels "","Female","Male": 3 3 3 3 3 3 3 3 3 3 ...
$ Married          : Factor w/ 3 levels "","No","Yes": 2 3 3 3 2 3 3 3 3 3 ...
$ Dependents       : Factor w/ 5 levels "","0","1","2",..: 2 3 2 2 2 4 2 5 4 3 ...
$ Education        : Factor w/ 2 levels "Graduate","Not Graduate": 1 1 1 2 1 1 2 1 1 1 ...
$ Self_Employed    : Factor w/ 3 levels "","No","Yes": 2 2 3 2 2 3 2 2 2 2 ...
$ ApplicantIncome  : int  5849 4583 3000 2583 6000 5417 2333 3036 4006 12841 ...
$ CoapplicantIncome: num  0 1508 0 2358 0 ...
$ LoanAmount       : int  NA 128 66 120 141 267 95 158 168 349 ...
$ Loan_Amount_Term : int  360 360 360 360 360 360 360 360 360 360 ...
$ Credit_History   : int  1 1 1 1 1 1 1 0 1 1 ...
$ Property_Area    : Factor w/ 3 levels "Rural","Semiurban",..: 3 1 3 3 3 3 3 2 3 2 ...
$ Loan_Status      : Factor w/ 2 levels "N","Y": 2 1 2 2 2 2 2 1 2 1 ...





#Imputing missing values using KNN.Also centering and scaling numerical columns

sum(is.na(train))
#[1] 86
preProcValues <- preProcess(train, method = c("knnImpute","center","scale"))

library('RANN')
train_processed <- predict(preProcValues, train)
sum(is.na(train_processed))
#[1] 0




#Converting every thing to numerical using dummy variables
train_processed$Loan_Status<-ifelse(train_processed$Loan_Status=='N',0,1)
id<-train_processed$Loan_ID
train_processed$Loan_ID<-NULL

str(train_processed)

'data.frame':	614 obs. of  12 variables:
  $ Gender           : Factor w/ 3 levels "","Female","Male": 3 3 3 3 3 3 3 3 3 3 ...
$ Married          : Factor w/ 3 levels "","No","Yes": 2 3 3 3 2 3 3 3 3 3 ...
$ Dependents       : Factor w/ 5 levels "","0","1","2",..: 2 3 2 2 2 4 2 5 4 3 ...
$ Education        : Factor w/ 2 levels "Graduate","Not Graduate": 1 1 1 2 1 1 2 1 1 1 ...
$ Self_Employed    : Factor w/ 3 levels "","No","Yes": 2 2 3 2 2 3 2 2 2 2 ...
$ ApplicantIncome  : num  0.0729 -0.1343 -0.3934 -0.4617 0.0976 ...
$ CoapplicantIncome: num  -0.554 -0.0387 -0.554 0.2518 -0.554 ...
$ LoanAmount       : num  0.0162 -0.2151 -0.9395 -0.3086 -0.0632 ...
$ Loan_Amount_Term : num  0.276 0.276 0.276 0.276 0.276 ...
$ Credit_History   : num  0.432 0.432 0.432 0.432 0.432 ...
$ Property_Area    : Factor w/ 3 levels "Rural","Semiurban",..: 3 1 3 3 3 3 3 2 3 2 ...
$ Loan_Status      : num  1 0 1 1 1 1 1 0 1 0 ...

#Fullrank will create only 3 columns for a catogoriacal column with 4 levels. 
dmy <- dummyVars(" ~ .", data = train_processed,fullRank = T)
train_transformed <- data.frame(predict(dmy, newdata = train_processed))

str(train_transformed)
'data.frame':	614 obs. of  19 variables:
  $ Gender.Female          : num  0 0 0 0 0 0 0 0 0 0 ...
$ Gender.Male            : num  1 1 1 1 1 1 1 1 1 1 ...
$ Married.No             : num  1 0 0 0 1 0 0 0 0 0 ...
$ Married.Yes            : num  0 1 1 1 0 1 1 1 1 1 ...
$ Dependents.0           : num  1 0 1 1 1 0 1 0 0 0 ...
$ Dependents.1           : num  0 1 0 0 0 0 0 0 0 1 ...
$ Dependents.2           : num  0 0 0 0 0 1 0 0 1 0 ...
$ Dependents.3.          : num  0 0 0 0 0 0 0 1 0 0 ...
$ Education.Not.Graduate : num  0 0 0 1 0 0 1 0 0 0 ...
$ Self_Employed.No       : num  1 1 0 1 1 0 1 1 1 1 ...
$ Self_Employed.Yes      : num  0 0 1 0 0 1 0 0 0 0 ...
$ ApplicantIncome        : num  0.0729 -0.1343 -0.3934 -0.4617 0.0976 ...
$ CoapplicantIncome      : num  -0.554 -0.0387 -0.554 0.2518 -0.554 ...
$ LoanAmount             : num  0.0162 -0.2151 -0.9395 -0.3086 -0.0632 ...
$ Loan_Amount_Term       : num  0.276 0.276 0.276 0.276 0.276 ...
$ Credit_History         : num  0.432 0.432 0.432 0.432 0.432 ...
$ Property_Area.Semiurban: num  0 0 0 0 0 0 0 1 0 1 ...
$ Property_Area.Urban    : num  1 0 1 1 1 1 1 0 1 0 ...
$ Loan_Status            : num  1 0 1 1 1 1 1 0 1 0 ...

train_transformed$Loan_Status<-as.factor(train_transformed$Loan_Status)

#Spliting training set into two parts based on outcome: 75% and 25%

index <- createDataPartition(train_transformed$Loan_Status, p=0.75, list=FALSE)
trainSet <- train_transformed[ index,]
testSet <- train_transformed[-index,]

str(trainSet)

'data.frame':	461 obs. of  19 variables:
$ Gender.Female          : num  0 0 0 0 0 0 0 0 0 0 ...
$ Gender.Male            : num  1 1 1 1 1 1 1 1 1 1 ...
$ Married.No             : num  1 0 0 0 1 0 0 0 0 0 ...
$ Married.Yes            : num  0 1 1 1 0 1 1 1 1 1 ...
$ Dependents.0           : num  1 0 1 1 1 0 1 0 0 0 ...
$ Dependents.1           : num  0 1 0 0 0 0 0 0 1 0 ...
$ Dependents.2           : num  0 0 0 0 0 1 0 0 0 1 ...
$ Dependents.3.          : num  0 0 0 0 0 0 0 1 0 0 ...
$ Education.Not.Graduate : num  0 0 0 1 0 0 1 0 0 0 ...
$ Self_Employed.No       : num  1 1 0 1 1 0 1 1 1 1 ...
$ Self_Employed.Yes      : num  0 0 1 0 0 1 0 0 0 0 ...
$ ApplicantIncome        : num  0.0729 -0.1343 -0.3934 -0.4617 0.0976 ...
$ CoapplicantIncome      : num  -0.554 -0.0387 -0.554 0.2518 -0.554 ...
$ LoanAmount             : num  0.0162 -0.2151 -0.9395 -0.3086 -0.0632 ...
$ Loan_Amount_Term       : num  0.276 0.276 0.276 0.276 0.276 ...
$ Credit_History         : num  0.432 0.432 0.432 0.432 0.432 ...
$ Property_Area.Semiurban: num  0 0 0 0 0 0 0 1 1 0 ...
$ Property_Area.Urban    : num  1 0 1 1 1 1 1 0 0 1 ...
$ Loan_Status            : Factor w/ 2 levels "0","1": 2 1 2 2 2 2 2 1 1 2 ...

#Feature selection using rfe in caret


control <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 3,
                   verbose = FALSE)

outcomeName<-'Loan_Status'
predictors<-names(trainSet)[!names(trainSet) %in% outcomeName]

Loan_Pred_Profile <- rfe(trainSet[,predictors], trainSet[,outcomeName],
                      rfeControl = control)

Loan_Pred_Profile

#Recursive feature selection

#Outer resampling method: Cross-Validated (10 fold, repeated 3 times) 

#Resampling performance over subset size:
  
#  Variables Accuracy  Kappa AccuracySD KappaSD Selected
#4   0.7737 0.4127    0.03707 0.09962         
#8   0.7874 0.4317    0.03833 0.11168         
#16   0.7903 0.4527    0.04159 0.11526        *
#18   0.7882 0.4431    0.03615 0.10812         

#The top 5 variables (out of 16):
#  Credit_History, LoanAmount, Loan_Amount_Term, ApplicantIncome, CoapplicantIncome

predictors<-c("Credit_History", "LoanAmount", "Loan_Amount_Term", "ApplicantIncome", "CoapplicantIncome")


#Training modles


#using grid search

modelLookup(model='gbm')

#model         parameter                   label forReg forClass probModel
#1   gbm           n.trees   # Boosting Iterations   TRUE     TRUE      TRUE
#2   gbm interaction.depth          Max Tree Depth   TRUE     TRUE      TRUE
#3   gbm         shrinkage               Shrinkage   TRUE     TRUE      TRUE
#4   gbm    n.minobsinnode Min. Terminal Node Size   TRUE     TRUE      TRUE

grid <- expand.grid(n.trees=c(10,20,50,100,500,1000),shrinkage=c(0.01,0.05,0.1,0.5),n.minobsinnode = c(3,5,10),interaction.depth=c(1,5,10))


fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 5)

# train the model
model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl=fitControl,tuneGrid=grid)
# summarize the model

print(model_gbm)

#Stochastic Gradient Boosting 

#461 samples
#5 predictor
#2 classes: '0', '1' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 5 times) 
#Summary of sample sizes: 368, 370, 369, 369, 368, 369, ... 
#Resampling results across tuning parameters:
  
#  shrinkage  interaction.depth  n.minobsinnode  n.trees  Accuracy   Kappa    
#0.01        1                  3                10     0.6876416  0.0000000
#0.01        1                  3                20     0.6876416  0.0000000
#0.01        1                  3                50     0.7982345  0.4423609
#0.01        1                  3               100     0.7952190  0.4364383
#0.01        1                  3               500     0.7904882  0.4342300
#0.01        1                  3              1000     0.7913627  0.4421230
#0.01        1                  5                10     0.6876416  0.0000000
#0.01        1                  5                20     0.6876416  0.0000000
#0.01        1                  5                50     0.7982345  0.4423609
#0.01        1                  5               100     0.7943635  0.4351912
#0.01        1                  5               500     0.7930783  0.4411348
#0.01        1                  5              1000     0.7913720  0.4417463
#0.01        1                 10                10     0.6876416  0.0000000
#0.01        1                 10                20     0.6876416  0.0000000
#0.01        1                 10                50     0.7982345  0.4423609
#0.01        1                 10               100     0.7943635  0.4351912
#0.01        1                 10               500     0.7939525  0.4426503
#0.01        1                 10              1000     0.7948362  0.4476742
#0.01        5                  3                10     0.6876416  0.0000000
#0.01        5                  3                20     0.6876416  0.0000000
#0.01        5                  3                50     0.7960556  0.4349571
#0.01        5                  3               100     0.7934987  0.4345481
#0.01        5                  3               500     0.7775055  0.4147204
#...
#0.50        5                 10               100     0.7045617  0.2834696
#0.50        5                 10               500     0.6924480  0.2650477
#0.50        5                 10              1000     0.7115234  0.3050953
#0.50       10                  3                10     0.7389117  0.3681917
#0.50       10                  3                20     0.7228519  0.3317001
#0.50       10                  3                50     0.7180833  0.3159445
#0.50       10                  3               100     0.7172417  0.3189655
#0.50       10                  3               500     0.7058472  0.3098146
#0.50       10                  3              1000     0.7001852  0.2967784
#0.50       10                  5                10     0.7266895  0.3378430
#0.50       10                  5                20     0.7154746  0.3197905
#0.50       10                  5                50     0.7063535  0.2984819
#0.50       10                  5               100     0.7151012  0.3141440
#0.50       10                  5               500     0.7108516  0.3146822
#0.50       10                  5              1000     0.7147320  0.3225373
#0.50       10                 10                10     0.7314871  0.3327504
#0.50       10                 10                20     0.7150814  0.3081869
#0.50       10                 10                50     0.6993723  0.2815981
#0.50       10                 10               100     0.6977416  0.2719140
#0.50       10                 10               500     0.7037864  0.2854748
#0.50       10                 10              1000     0.6995610  0.2869718

#Accuracy was used to select the optimal model using  the largest value.
#The final values used for the model were n.trees = 10, interaction.depth = 1, shrinkage =
#  0.05 and n.minobsinnode = 3. 

plot(model_gbm)

#

#using tunelength


model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl=fitControl,tuneLength=5)


print(model_gbm)
#Stochastic Gradient Boosting 

#461 samples
#5 predictor
#2 classes: '0', '1' 

#No pre-processing
#Resampling: Cross-Validated (5 fold, repeated 5 times) 
#Summary of sample sizes: 368, 369, 369, 370, 368, 369, ... 
#Resampling results across tuning parameters:

#  interaction.depth  n.trees  Accuracy   Kappa    
#1                  50      0.7978084  0.4541008
#1                 100      0.7978177  0.4566764
#1                 150      0.7934792  0.4472347
#1                 200      0.7904310  0.4424091
#1                 250      0.7869714  0.4342797
#1                 300      0.7830488  0.4262414
...
#10                 100      0.7575230  0.3860319
#10                 150      0.7479757  0.3719707
#10                 200      0.7397290  0.3566972
#10                 250      0.7397285  0.3561990
#10                 300      0.7362552  0.3513413
#10                 350      0.7340812  0.3453415
#10                 400      0.7336416  0.3453117
#10                 450      0.7306027  0.3415153
#10                 500      0.7253854  0.3295929

#Tuning parameter 'shrinkage' was held constant at a value of 0.1
#Tuning
#parameter 'n.minobsinnode' was held constant at a value of 10
#Accuracy was used to select the optimal model using  the largest value.
#The final values used for the model were n.trees = 50, interaction.depth = 2, shrinkage =
#  0.1 and n.minobsinnode = 10. 


plot(model_gbm)

#


#Model training

names(getModelInfo())

#[1] "ada"                 "AdaBag"              "AdaBoost.M1"         "adaboost"           
#[5] "amdai"               "ANFIS"               "avNNet"              "awnb"               
#[9] "awtan"               "bag"                 "bagEarth"            "bagEarthGCV"        
#[13] "bagFDA"              "bagFDAGCV"           "bam"                 "bartMachine"        
#[17] "bayesglm"            "bdk"                 "binda"               "blackboost"         
#[21] "blasso"              "blassoAveraged"      "Boruta"              "bridge"             
#[25] "brnn"                "BstLm"               "bstSm"               "bstTree"            
#[29] "C5.0"                "C5.0Cost"            "C5.0Rules"           "C5.0Tree"           
#[33] "cforest"             "chaid"               "CSimca"              "ctree"              
#[37] "ctree2"              "cubist"              "dda"                 "deepboost"          
#[41] "DENFIS"              "dnn"                 "dwdLinear"           "dwdPoly"            
#[45] "dwdRadial"           "earth"               "elm"                 "enet"               
#[49] "enpls.fs"            "enpls"               "evtree"              "extraTrees"         
#[53] "fda"                 "FH.GBML"             "FIR.DM"              "foba"               
#[57] "FRBCS.CHI"           "FRBCS.W"             "FS.HGD"              "gam"                
#[61] "gamboost"            "gamLoess"            "gamSpline"           "gaussprLinear"      
#[65] "gaussprPoly"         "gaussprRadial"       "gbm"                 "gcvEarth"           
#[69] "GFS.FR.MOGUL"        "GFS.GCCL"            "GFS.LT.RS"           "GFS.THRIFT"         
#[73] "glm"                 "glmboost"            "glmnet"              "glmStepAIC"         
#[77] "gpls"                "hda"                 "hdda"                "hdrda"              
#[81] "HYFIS"               "icr"                 "J48"                 "JRip"               
#[85] "kernelpls"           "kknn"                "knn"                 "krlsPoly"           
#[89] "krlsRadial"          "lars"                "lars2"               "lasso"              
#[93] "lda"                 "lda2"                "leapBackward"        "leapForward"        
#[97] "leapSeq"             "Linda"               "lm"                  "lmStepAIC"          
#[101] "LMT"                 "loclda"              "logicBag"            "LogitBoost"         
#[105] "logreg"              "lssvmLinear"         "lssvmPoly"           "lssvmRadial"        
#[109] "lvq"                 "M5"                  "M5Rules"             "manb"               
#[113] "mda"                 "Mlda"                "mlp"                 "mlpML"              
#[117] "mlpSGD"              "mlpWeightDecay"      "mlpWeightDecayML"    "multinom"           
#[121] "nb"                  "nbDiscrete"          "nbSearch"            "neuralnet"          
#[125] "nnet"                "nnls"                "nodeHarvest"         "oblique.tree"       
#[129] "OneR"                "ordinalNet"          "ORFlog"              "ORFpls"             
#[133] "ORFridge"            "ORFsvm"              "ownn"                "pam"                
#[137] "parRF"               "PART"                "partDSA"             "pcaNNet"            
#[141] "pcr"                 "pda"                 "pda2"                "penalized"          
#[145] "PenalizedLDA"        "plr"                 "pls"                 "plsRglm"            
#[149] "polr"                "ppr"                 "protoclass"          "pythonKnnReg"       
#[153] "qda"                 "QdaCov"              "qrf"                 "qrnn"               
#[157] "randomGLM"           "ranger"              "rbf"                 "rbfDDA"             
#[161] "Rborist"             "rda"                 "relaxo"              "rf"                 
#[165] "rFerns"              "RFlda"               "rfRules"             "ridge"              
#[169] "rlda"                "rlm"                 "rmda"                "rocc"               
#[173] "rotationForest"      "rotationForestCp"    "rpart"               "rpart1SE"           
#[177] "rpart2"              "rpartCost"           "rpartScore"          "rqlasso"            
#[181] "rqnc"                "RRF"                 "RRFglobal"           "rrlda"              
#[185] "RSimca"              "rvmLinear"           "rvmPoly"             "rvmRadial"          
#[189] "SBC"                 "sda"                 "sddaLDA"             "sddaQDA"            
#[193] "sdwd"                "simpls"              "SLAVE"               "slda"               
#[197] "smda"                "snn"                 "sparseLDA"           "spikeslab"          
#[201] "spls"                "stepLDA"             "stepQDA"             "superpc"            
#[205] "svmBoundrangeString" "svmExpoString"       "svmLinear"           "svmLinear2"         
#[209] "svmLinear3"          "svmLinearWeights"    "svmLinearWeights2"   "svmPoly"            
#[213] "svmRadial"           "svmRadialCost"       "svmRadialSigma"      "svmRadialWeights"   
#[217] "svmSpectrumString"   "tan"                 "tanSearch"           "treebag"            
#[221] "vbmpRadial"          "vglmAdjCat"          "vglmContRatio"       "vglmCumulative"     
#[225] "widekernelpls"       "WM"                  "wsrf"                "xgbLinear"          
#[229] "xgbTree"             "xyf"  

model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf',trControl=fitControl)

model_nnet<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet',trControl=fitControl)

model_glm<-train(trainSet[,predictors],trainSet[,outcomeName],method='glm',trControl=fitControl)

#Variable Importance

varImp(object=model_gbm)
#gbm variable importance

#Overall
#Credit_History    100.000
#LoanAmount         16.633
#ApplicantIncome     7.104
#CoapplicantIncome   6.773
#Loan_Amount_Term    0.000

plot(varImp(object=model_gbm),main="GBM - Variable Importance")


varImp(object=model_rf)
#rf variable importance

#Overall
#Credit_History     100.00
#ApplicantIncome     73.46
#LoanAmount          60.59
#CoapplicantIncome   40.43
#Loan_Amount_Term     0.00

qplot(varImp(object=model_rf),main="RF - Variable Importance")

varImp(object=model_nnet)
#nnet variable importance

#Overall
#ApplicantIncome    100.00
#LoanAmount          82.87
#CoapplicantIncome   56.92
#Credit_History      41.11
#Loan_Amount_Term     0.00

plot(varImp(object=model_nnet),main="NNET - Variable Importance")

varImp(object=model_glm)
#glm variable importance

#Overall
#Credit_History    100.000
#CoapplicantIncome  17.218
#Loan_Amount_Term   12.988
#LoanAmount          5.632
#ApplicantIncome     0.000

plot(varImp(object=model_glm),main="GLM - Variable Importance")

#Predictions

predictions<-predict.train(object=model_gbm,testSet[,predictors],type="raw")

table(predictions)
#predictions
#0   1 
#28 125 

confusionMatrix(predictions,testSet[,outcomeName])

#Confusion Matrix and Statistics

#Reference
#Prediction   0   1
#0  25   3
#1  23 102

#Accuracy : 0.8301         
#95% CI : (0.761, 0.8859)
#No Information Rate : 0.6863         
#P-Value [Acc > NIR] : 4.049e-05      

#Kappa : 0.555          
#Mcnemar's Test P-Value : 0.0001944      

#Sensitivity : 0.5208         
#Specificity : 0.9714         
#Pos Pred Value : 0.8929         
#Neg Pred Value : 0.8160         
#Prevalence : 0.3137         
#Detection Rate : 0.1634         
#Detection Prevalence : 0.1830         
#Balanced Accuracy : 0.7461         

#'Positive' Class : 0   
