setwd("C:/Users/mohad/OneDrive/Desktop/Humana-Challenge")
library(dplyr)
df_train <- read.csv("X_y_data_stat.csv")
df_test <- read.csv("X_y_test_stat.csv")

#df_train=df_train[c=(cms_tot_partd_payment_amt,est_age,cms_disabled_ind,ccsp_239_ind,crns_rx_risk_score_nbr,cms_low_income_ind,betos_ola_pmpm_ct_betos,MEN_pmpm_sum_MCC,mabh_seg_h_c,cms_partd_ra_factor_amt,total_ambulance_visit_ct_pmpm,submcc_men_depr_pmpm_ct_MCC,bh_sum_ind,cms_tot_ma_payment_amt,rx_overall_pmpm_ct_RX,pmpm_sum_neuro_RX,submcc_res_copd_pmpm_ct_MCC)]
df_train=subset(df_train,select =c('transportation_issues','cms_tot_partd_payment_amt','est_age','cms_disabled_ind','ccsp_239_ind','cms_rx_risk_score_nbr','cms_low_income_ind','betos_o1a_pmpm_ct_betos','MEN_pmpm_sum_MCC','mabh_seg_h_c','cms_partd_ra_factor_amt','total_ambulance_visit_ct_pmpm','submcc_men_depr_pmpm_ct_MCC','bh_sum_ind','cms_tot_ma_payment_amt','rx_overall_pmpm_ct_RX','pmpm_sum_neuro_RX','submcc_res_copd_pmpm_ct_MCC'))
df_test=subset(df_test,select =c('transportation_issues','cms_tot_partd_payment_amt','est_age','cms_disabled_ind','ccsp_239_ind','cms_rx_risk_score_nbr','cms_low_income_ind','betos_o1a_pmpm_ct_betos','MEN_pmpm_sum_MCC','mabh_seg_h_c','cms_partd_ra_factor_amt','total_ambulance_visit_ct_pmpm','submcc_men_depr_pmpm_ct_MCC','bh_sum_ind','cms_tot_ma_payment_amt','rx_overall_pmpm_ct_RX','pmpm_sum_neuro_RX','submcc_res_copd_pmpm_ct_MCC'))
colnames(df_train)
#df_train <- read.csv("Top30_undersampled_train_data.csv")
#df_test <- read.csv("Top30_undersampled_test_data.csv")
#df_train$X=NULL
#df_test$X=NULL
#colnames(df_train)
df_train$transportation_issues<-factor(df_train$transportation_issues)
df_train$cms_disabled_ind<-factor(df_train$cms_disabled_ind)
df_train$ccsp_239_ind<-factor(df_train$ccsp_239_ind)
df_train$cms_low_income_ind<-factor(df_train$cms_low_income_ind)
df_train$mabh_seg_h_c<-factor(df_train$mabh_seg_h_c)
#df_train$betos_m5d_ind_betos<-factor(df_train$betos_m5d_ind_betos)


logit  <- glm(transportation_issues ~ .  , family=binomial (link="logit"), data=df_train)
summary(logit)

logit2prob <- function(logit){
  odds <- exp(logit)
  prob <- odds / (1 + odds)
  return(prob)
}
exp(coef(logit))
logit2prob(coef(logit))

#baseline probit model
probit <- glm(transportation_issues ~ .  , family=binomial (link="probit"), data=df_train)
summary(probit)

df_test$transportation_issues<-factor(df_test$transportation_issues)
df_test$ccsp_239_ind<-factor(df_test$ccsp_239_ind)
#df_test$betos_m5d_ind_betos<-factor(df_test$betos_m5d_ind_betos)
df_test$mabh_seg_h_c<-factor(df_test$mabh_seg_h_c)
df_test$cms_disabled_ind<-factor(df_test$cms_disabled_ind)
df_test$cms_low_income_ind<-factor(df_test$cms_low_income_ind)

tt = subset(df_test, select=-c(transportation_issues))#df_test[-c(df_test$transportation_issues)]
ts = subset(df_train, select=-c(transportation_issues))#df_test[-c(df_test$transportation_issues)]

logit_pred = predict(logit, newdata=tt,type="response")
logit_pred <- ifelse(logit_pred>0.5, 1, 0)
library(caret)
cm <-confusionMatrix(as.factor(logit_pred), reference = df_test$transportation_issues)
cm$byClass['Recall'] #0.90
cm$byClass['F1']# 0.86
cm$byClass['Precision']
cm
#tt = subset(df_test, select=-c(transportation_issues))#df_test[-c(df_test$transportation_issues)]

library(ROCR)
pr <- prediction(logit_pred, df_train$transportation_issues)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)                                                 # ROC plot: TP vs FP

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

ts = subset(df_train, select=-c(transportation_issues))#df_test[-c(df_test$transportation_issues)]
logit_pred = predict(logit, newdata=ts,type="response")
logit_pred <- ifelse(logit_pred>0.5, 1, 0)
library(caret)
cm <-confusionMatrix(as.factor(logit_pred), reference = df_train$transportation_issues)
cm$byClass['Recall'] #0.90
cm$byClass['F1']# 0.86
cm$byClass['Precision']
cm

library(ROCR)
pr <- prediction(logit_pred, df_train$transportation_issues)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)                                                 # ROC plot: TP vs FP

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc