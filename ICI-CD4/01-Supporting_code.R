## GI Austin: This file is almost entirely from the original Lozano publication, their comments/reference is below
## the only adjustment under the Austin et al publication is the RLOOCV implementation of the analysis

#This code is originally from a supporting resource for Lozano, Chaudhuri, Nene et al., Nature Medicine 2021
#Scripts for training and validation of composite models, evaluating freedom from severe toxicity, and generating related figures are included.

#   Requirements
#1. Ensure the following R packages are installed: pROC, survminer, ggplot2
#2. Place 'model_input.txt' and this R script in the same directory, or specify the full path to  'model_input.txt' in the INPUT variable below
#3. From the command line, run: "Rscript Supporting_code.R"

suppressMessages(library(survival))
suppressMessages(library(pROC))
suppressMessages(library(survminer))
suppressMessages(library(ggplot2))

options(scipen = 999)

INPUT <- "model_input.txt" #Path to model_input.txt

actCD4mem_ceiling <- 0.1 #All activated CD4 T cell values identified as outliers by the ROUT test in Prism with 10% FDR have frequency >0.1, regardless of training subset in the analyses below (Methods)

x <- read.table(INPUT,sep="\t", header=T, row.names=1)

#================Functions===================

#Build and run composite model on training and validation sets
makeCompositeModel <- function(input, index, actCD4mem_ceiling, LOOCV, 
                               LOOCV_correct = F){
    
    if(!LOOCV){
        ##### replace outliers with max value of non-outliers
        AMCD4 <- input$Activated_CD4_memory_T_CSx
        mx <- max((AMCD4[index])[which(AMCD4[index]<actCD4mem_ceiling)])
        AMCD4[which(AMCD4 > mx)] <- mx
        input$AMCD4 <- AMCD4
        #####

        #Train logistic regression model ('composite model') on activated CD4 memory T cell abundance and TCR Shannon entropy using patient subset defined by 'index'
        rf <- glm(formula=as.factor(Severe_irAE)~AMCD4 + TCR_clonotype_diversity_ShannonEntropy,
                  data=input[index,],
                  family="binomial")
        #Run composite model on all pts
        out <- predict(rf, input,type=c("response"))
        modelPred <- out
        input$CompositeModel <- modelPred #Continuous model results
        
    }else{
        
        #Leave-one-out cross-validatoin (LOOCV)
        input2 <- input
        
        for(i in 1:nrow(input)){
            
            ##### replace outliers with max value of non-outliers
            AMCD4 <- input$Activated_CD4_memory_T_CSx
            mx <- max((AMCD4[-i])[which(AMCD4[-i]<actCD4mem_ceiling)])
            AMCD4[which(AMCD4 > mx)] <- mx
            input2$AMCD4 <- AMCD4
            #####

            #Train logistic regression model ('composite model') on activated CD4 memory T cell abundance and TCR Shannon entropy using patient subset using LOOCV
            if(LOOCV_correct==F){
            
            rf <- glm(formula=as.factor(Severe_irAE)~AMCD4 + TCR_clonotype_diversity_ShannonEntropy,
            data=input2[-i,], ## G. AUSTIN: THEIR ORIGINAL LINE
                      family="binomial"
                      )
            
            }else{
              rf <- glm(formula=as.factor(Severe_irAE) ~ AMCD4 + TCR_clonotype_diversity_ShannonEntropy,
                        data = input2[-c(i, 
                                         which(input2$Severe_irAE != input2$Severe_irAE[i]) %>%
                                         sample(1)), ], ## G. AUSTIN: THIS IS OUR NOVELTY
                        family="binomial"  )
            }
            #Run composite model on all pts
            out <- predict(rf, input2,type=c("response"))
            modelPred <- out
            input$CompositeModelLOOCV[i] <- modelPred[i] #Continuous model results
            
            #Optimize split using Youden's J statistic
            suppressMessages(g <- pROC::roc(as.factor(input2$Severe_irAE[-i]) ~ modelPred[-i], data=input2))
            cutp <- g$thresholds[which.max(g$sensitivities+g$specificities)]
            
            if(modelPred[i] > cutp)  input$BinaryModelLOOCV[i] <- 1
            else input$BinaryModelLOOCV[i] <- 0
        }
    }
    
    input #return input
}

#Calculate AUC of composite model in patient subset 'index'
calculateAUC <- function(input, index, LOOCV){
    
    if(!LOOCV){
        suppressMessages(g <- pROC::roc(as.factor(input$Severe_irAE[index]) ~ CompositeModel[index], data=input))
    }else{
        suppressMessages(g <- pROC::roc(as.factor(input$Severe_irAE[index]) ~ CompositeModelLOOCV[index], data=input))
    }
    g #return AUC model

}

#Optimize split using Youden's J statistic and generate binary composite model
binarizeModel <- function(input, g){

    cutp <- g$thresholds[which.max(g$sensitivities+g$specificities)]
    q <- rep(0,nrow(input))
    q[which(input$CompositeModel>=cutp)] <- 1
    input$BinaryModel <- q
    input #return input
}

#Get hazard ratio of binary composite model
getCox <- function(input, index, LOOCV){
    
    if(!LOOCV){
        mod <- suppressWarnings(summary(coxph(Surv(Time_to_severe_irAE_months, Severe_irAE) ~ BinaryModel, data = input[index,])))
    }else{
        mod <- suppressWarnings(summary(coxph(Surv(Time_to_severe_irAE_months, Severe_irAE) ~ BinaryModelLOOCV, data = input[index,])))
    }
    c(mod$coefficients[1,2],mod$sctest[3])
}

makeKMplot <- function(input, index, label, LOOCV){
    
    pdf(file=paste(label,".pdf", sep=""), height=5, width = 4.5, onefile=FALSE)
    if(!LOOCV){
        fit <- survfit(Surv(Time_to_severe_irAE_months, Severe_irAE) ~ BinaryModel, data = input[index,])
    }else{
        fit <- survfit(Surv(Time_to_severe_irAE_months, Severe_irAE) ~ BinaryModelLOOCV, data = input[index,])
    }
    print(ggsurvplot(fit, data = input[index,], risk.table = TRUE, pval = TRUE, xlim=c(0,4),break.time.by = 1, risk.table.y.text = FALSE, palette=c("#1B76FF","#BD111D"), xlab="Months after treatment initiation", ylab="Fraction free from severe irAE", censor=F))
    q <- dev.off()
    
}

############################################################################

############################################################################

#Extended Data Figure 7a

#Train on all patients using LOOCV
x <- makeCompositeModel(x, "", actCD4mem_ceiling, TRUE)
set.seed(123)
x_nc <-  makeCompositeModel(x, "", actCD4mem_ceiling, TRUE)
x_wc <-  makeCompositeModel(x, "", actCD4mem_ceiling, TRUE, LOOCV_correct = T)
x_nc %>% write.csv('no-correction-results.csv')
x_wc %>% write.csv('with-correction-results.csv')

set.seed(123)
multiple_runs <- list()
for(i in 1:10){
  x_wc_ <- makeCompositeModel(x, "", actCD4mem_ceiling, TRUE, LOOCV_correct = T) %>% 
              mutate(run_num=i)
  multiple_runs[[i]] <- x_wc_
  print(pROC::roc(x_wc$Severe_irAE, x_wc_$CompositeModelLOOCV ))
}

multiple_runs %>% data.table::rbindlist() %>% write.csv('bootstrapped-with-correction-results.csv')


set.seed(123)
multiple_runs <- list()
for(i in 1:10){
  x_wc_ <- makeCompositeModel(x, "", actCD4mem_ceiling, TRUE, LOOCV_correct = F) %>% 
    mutate(run_num=i)
  multiple_runs[[i]] <- x_wc_
  print(pROC::roc(x_wc$Severe_irAE, x_wc_$CompositeModelLOOCV ))
}

multiple_runs %>% data.table::rbindlist() %>% write.csv('bootstrapped-no-correction-results.csv')


