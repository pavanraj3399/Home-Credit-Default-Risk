######## Setting working directory
setwd("~/Desktop/git/Home-Credit-Default-Risk")

######## Importing Libraries ########

library(dplyr)        # Data Manipulation
library(ggplot2)      # Data Visualizationx
library(mltools)      # Data Manipulation
library(data.table)   # Data Manipulation
library(reshape2)     # Data Manipulation
library(VIM)          # Handling Missing values
library(class)        # Functions for Classification
library(validate)     # Handling Outliers
library(ROSE)         # Sampling
library(caret)        # Data Manipulation
library(randomForest) # Machine Learning
library(ROCR)         # Performance Evaluation
library(inTrees)      # Extract rules from random forest

############################################
############ Importing Datasets ############
############################################


# application data
application_data <- read.csv("application_train.csv")

# client's previous credits provided by other financial institutions
bureau_data <- read.csv("bureau.csv")

# previous applications for Home Credit loans of clients who have loans in hc sample.
previous_application_data <- read.csv("previous_application.csv")


############################################
############ Merging datasets ############
############################################


##########  1 - application_data ##########

# Replace "" with NA
application_data <- na_if(application_data, "")

##### Missing values function
#### Argument 1 - Data set
#### Argument 2 - Percent of missing values

missing_data <- function(dataset, set_percent) {
  # missing percent
  missing_values <- data.frame(
    "features" = colnames(dataset),
    "missing_percentage" = round(colSums(is.na(dataset)) /
                                   nrow(dataset) * 100, 3) %>% as.vector()
  )
  
  # plot missing values percentage in dataset
  plot <-
    ggplot(data = missing_values %>% filter(missing_percentage != 0)) +
    geom_bar(aes(y = reorder(features, missing_percentage),
                 x = missing_percentage),
             stat = "identity",
             fill = "lightblue") +
    ggtitle("Missing values percentage") + ylab("features")
  
  
  # extract columns with missing more than set persentage
  missing_values <-
    missing_values %>% filter(missing_percentage > set_percent)
  missing_values <- missing_values$features %>% as.vector()
  
  return(list(plot, missing_values))
}


# Missing values in application data with more than 40% missing values
missing_data(application_data, 40)


# Drop features with more than 40% missing values
application_data_df <-
  application_data %>% select(-missing_data(application_data, 40)[[2]])
head(application_data_df)



# Missed data from applications
missed_data_applications <- application_data %>%
  select(c("SK_ID_CURR", missing_data(application_data, 40)[[2]]))

head(missed_data_applications)

# Replace NA in own car age with 0 - missing values for who dont have a car MNAR
missed_data_applications$OWN_CAR_AGE <-
  ifelse(is.na(application_data$OWN_CAR_AGE),0,
         application_data$OWN_CAR_AGE)

# Merge selected columns with applications data
application_data_df <-
  merge(application_data_df,
        missed_data_applications %>% select(c("SK_ID_CURR", "OWN_CAR_AGE")),
        by = "SK_ID_CURR")


##########  2 - bureau_data ##########

# Replace "" with NA
bureau_data_new <- na_if(bureau_data, "")

# category columns in bureau_data
category_cols <- c("CREDIT_ACTIVE", "CREDIT_CURRENCY", "CREDIT_TYPE")

# convert to factors
bureau_data_new[, category_cols] <-
  lapply(bureau_data_new[, category_cols], as.factor)

# one hot encoding categorical variables
bureau_data_new <- cbind(bureau_data_new %>% select(-all_of(category_cols)),
                         one_hot(as.data.table(
                           bureau_data_new %>% select(all_of(category_cols))
                         )))


# Grouping and summing categorical variables
bureau_data_grouped_cat <-
  bureau_data_new %>% group_by(SK_ID_CURR) %>% summarise(across(colnames(bureau_data_new)[15:37], sum))


# Grouping and summing numerical variables
bureau_data_grouped_num <-
  bureau_data_new %>% group_by(SK_ID_CURR) %>% summarise(PREV_APP_COUNT = length(SK_ID_BUREAU))


# Merge categorical and numerical vars
bureau_data_new <-
  merge(bureau_data_grouped_cat, bureau_data_grouped_num, by = "SK_ID_CURR")



# Replace " " with "_" for feature names in bureau_data_new
names(bureau_data_new) <- gsub(" ", "_", names(bureau_data_new))
names(bureau_data_new) <-
  gsub("(", "", names(bureau_data_new), fixed = TRUE)
names(bureau_data_new) <-
  gsub(")", "", names(bureau_data_new), fixed = TRUE)


# Add prefix "BU" to feature names  of bureau_data_new
colnames(bureau_data_new)[2:length(colnames(bureau_data_new))] <-
  paste("BU", colnames(bureau_data_new)[2:length(colnames(bureau_data_new))], sep = "_")


# Merge bureau data with application data
applications_bureau_merged <-
  merge(application_data_df, bureau_data_new, by = "SK_ID_CURR")

##########  2 - previous_application_data ##########

# Extract previous_application_data with necessary columns
previous_application_data_new <-
  previous_application_data %>% select(c(
    "SK_ID_PREV",
    "SK_ID_CURR",
    "NAME_CONTRACT_STATUS",
    "CODE_REJECT_REASON"
  ))

# Vector for categorical columns in previous_application_data
category_cols_prev <- c("NAME_CONTRACT_STATUS", "CODE_REJECT_REASON")


# convert to factors for categorical columns in previous_application_data
previous_application_data_new[, category_cols_prev] <-
  lapply(previous_application_data_new[, category_cols_prev],
         as.factor)

# one hot encoding categorical variables
previous_application_data_new <-
  cbind(
    previous_application_data_new %>% select(-all_of(category_cols_prev)),
    one_hot(as.data.table(
      previous_application_data_new %>% select(all_of(category_cols_prev))
    ))
  )


# Grouping and summing categorical variables
previous_application_cat <-
  previous_application_data_new %>% group_by(SK_ID_CURR) %>%
  summarise(across(colnames(previous_application_data_new)[3:15], sum))

# Grouping and counting previous applications
previous_application_count <- previous_application_data_new %>%
  group_by(SK_ID_CURR) %>% summarise(PREV_APP_COUNT = length(SK_ID_PREV))


# Merge categorical and count data
previous_application_merged <-
  merge(previous_application_cat, previous_application_count, by = "SK_ID_CURR")


# Replace " " with "_" for features in previous applications merged data
names(previous_application_merged) <-
  gsub(" ", "_", names(previous_application_merged))


# Add prefix "HC" to colnames of previous_application_merged
colnames(previous_application_merged)[2:length(colnames(previous_application_merged))] <-
  paste("HC", colnames(previous_application_merged)[2:length(colnames(previous_application_merged))], sep = "_")

# Merge applications_bureau_merged with previous applications
application_data_final <-
  merge(applications_bureau_merged, previous_application_merged, by = "SK_ID_CURR")


############################################
############ Change data types ############
############################################

# Change case of all column names to lower case
colnames(application_data_final) <- tolower(colnames(application_data_final))

# Replace "-" with "_" in feature names
names(application_data_final) <-
  gsub("-", "_", names(application_data_final))


###### Change data types

# target feature
target_feature <- "target"

# unique identifier
unique_identifier <- "sk_id_curr"

# factor columns
factor_features <- c("name_contract_type","code_gender","flag_own_car","flag_own_realty",
                     "name_type_suite","name_income_type","name_education_type",
                     "name_family_status","name_housing_type","flag_mobil","flag_emp_phone",
                     "flag_work_phone","flag_cont_mobile","flag_phone","flag_email",
                     "occupation_type","region_rating_client","region_rating_client_w_city",
                     "weekday_appr_process_start","hour_appr_process_start","reg_region_not_live_region",
                     "reg_region_not_work_region","live_region_not_work_region","reg_city_not_live_city",
                     "reg_city_not_work_city","live_city_not_work_city","organization_type",
                     "flag_document_2","flag_document_3","flag_document_4","flag_document_5",
                     "flag_document_6","flag_document_7","flag_document_8","flag_document_9",
                     "flag_document_10","flag_document_11","flag_document_12","flag_document_13",
                     "flag_document_14","flag_document_15","flag_document_16","flag_document_17",
                     "flag_document_18","flag_document_19","flag_document_20","flag_document_21" )


# numerical columns
numerical_features <- c("cnt_children","amt_income_total","amt_credit","amt_annuity",
                        "amt_goods_price","region_population_relative","days_birth","days_employed",
                        "days_registration","days_id_publish","cnt_fam_members","ext_source_2",
                        "ext_source_3","obs_30_cnt_social_circle","def_30_cnt_social_circle",
                        "obs_60_cnt_social_circle","def_60_cnt_social_circle","days_last_phone_change",
                        "amt_req_credit_bureau_hour","amt_req_credit_bureau_day","amt_req_credit_bureau_week",
                        "amt_req_credit_bureau_mon","amt_req_credit_bureau_qrt","amt_req_credit_bureau_year",
                        "own_car_age","bu_credit_active_active","bu_credit_active_closed",
                        "bu_credit_active_sold","bu_credit_active_bad_debt","bu_credit_currency_currency_1",
                        "bu_credit_currency_currency_2","bu_credit_currency_currency_3",
                        "bu_credit_currency_currency_4","bu_credit_type_another_type_of_loan",
                        "bu_credit_type_car_loan", "bu_credit_type_cash_loan_non_earmarked",
                        "bu_credit_type_consumer_credit" ,"bu_credit_type_credit_card" ,"bu_credit_type_interbank_credit",
                        "bu_credit_type_loan_for_business_development","bu_credit_type_loan_for_purchase_of_shares_margin_lending",
                        "bu_credit_type_loan_for_the_purchase_of_equipment","bu_credit_type_loan_for_working_capital_replenishment",
                        "bu_credit_type_microloan", "bu_credit_type_mobile_operator_loan", "bu_credit_type_mortgage",
                        "bu_credit_type_real_estate_loan","bu_credit_type_unknown_type_of_loan",
                        "bu_prev_app_count","hc_name_contract_status_approved","hc_name_contract_status_canceled",
                        "hc_name_contract_status_refused","hc_name_contract_status_unused_offer",
                        "hc_code_reject_reason_client" ,"hc_code_reject_reason_hc", "hc_code_reject_reason_limit", 
                        "hc_code_reject_reason_sco" ,"hc_code_reject_reason_scofr" ,
                        "hc_code_reject_reason_system" ,"hc_code_reject_reason_verif" ,
                        "hc_code_reject_reason_xap" ,"hc_code_reject_reason_xna",
                        "hc_prev_app_count")


## Convert selected features to factor types
application_data_final[,c(all_of(c(target_feature,all_of(factor_features))))] <- 
  lapply(application_data_final %>% select(c(target,all_of(factor_features))), factor)


## Convert selected features to numerical types
application_data_final[,numerical_features] <- 
  lapply(application_data_final %>% select(all_of(numerical_features)),as.numeric)



# Structure of data
str(application_data_final)

# Summary of data
summary(application_data_final)



############################################
############ Feature selection ############
############################################

# Unique values in dataset
distict_values_df <- data.frame("feature_name"    = colnames(application_data_final),
                                "distinct_values" = sapply(application_data_final, n_distinct))

rownames(distict_values_df) <- 1:nrow(distict_values_df)



# Features with single values
features_one_uniq <- distict_values_df %>% filter(distinct_values == 1)
head(features_one_uniq)


# Features with equal to 2 unique values
features_two_uniq <- distict_values_df %>% filter(distinct_values == 2) %>% 
  filter(!feature_name %in% c("name_contract_type","flag_own_car","flag_own_realty"))

# subset dataset with two unique values
feature_two_df <- application_data_final %>% select(features_two_uniq$feature_name)

# change datatypes to factor for contingency table 
feature_two_df[,colnames(feature_two_df)] <- lapply(feature_two_df[,colnames(feature_two_df)],
                                                    factor)
contingency_tab <- melt(feature_two_df,id.vars = "target") 
contingency_tab <- round(prop.table(table(contingency_tab$variable,contingency_tab$value),margin = 1)*100,7) %>% 
  as.data.frame.matrix()
colnames(contingency_tab) <- c("target_proportion_0","target_proportion_1")
contingency_tab


# Select features with very less proportion of target variable with less than 0.05%
less_prop_tab <- contingency_tab %>% filter(target_proportion_0 <= 0.05 | target_proportion_1 <= 0.05)
less_prop_tab

################
# Features with three distinct values 
features_three_prop <- rbind(round(table(application_data_final$bu_credit_type_cash_loan_non_earmarked)/nrow(application_data_final)*100,5),
                             round(table(application_data_final$bu_credit_type_loan_for_the_purchase_of_equipment)/nrow(application_data_final)*100,5))

rownames(features_three_prop) <- c("bu_credit_type_cash_loan_non_earmarked",
                                   "bu_credit_type_loan_for_the_purchase_of_equipment")

colnames(features_three_prop) <- c("value_0","value_1","value_2")

# Proportion of features with three distinct values 
features_three_prop

# ##############

# Chi square test for all predictor variables with the target variable

select_features <- c(target_feature,factor_features[!factor_features %in% c(features_one_uniq$feature_name,rownames(less_prop_tab),
                                                                            rownames(features_three_prop))])



chi_sq_test <- lapply((application_data_final %>% select(all_of(c(target_feature,select_features))))[,-1],
                      function(x) chisq.test((application_data_final %>% select(all_of(c(target_feature,select_features))))[,1],x))


# Chi square test table
chi_sq_test <- data.frame(do.call(rbind,chi_sq_test)[,c(1,3)])
chi_sq_test

# Drop features with significance level greater than 0.05
chi_sq_test$features <- rownames(chi_sq_test)
rownames(chi_sq_test) <- NULL
chi_sq_drop <- chi_sq_test %>% filter(p.value > 0.05) %>% 
  select(c(features,statistic,p.value))
chi_sq_drop

# remove weekday to see patterns in combination with hour
chi_sq_drop_features <- chi_sq_drop$features[chi_sq_drop$features != "weekday_appr_process_start"]


# Drop selected features from dataset
drop_features <- c(features_one_uniq$feature_name,rownames(less_prop_tab),
                   rownames(features_three_prop),chi_sq_drop_features)

application_data_final <- application_data_final %>% select(-all_of(drop_features))


# Remove drop features from numerical features vector
numerical_features_update <- numerical_features[!numerical_features %in% drop_features]

# Remove drop features from factor features vector
factor_features_update  <- factor_features[!factor_features %in% drop_features]

application_data_final

#################################################
############ Handling Missing Values ############
#################################################

# Missing values patterns
missing_data(application_data_final,0)
missing_pattern <- application_data_final %>% select(missing_data(application_data_final,0)[[2]])
colnames(missing_pattern) <- c("annuity" ,"goods_price", "suite_type","occupation","ext_src_2",
                               "ext_src_3", "obs_30_sc", "def_30_sc",
                               "obs_60_sc", "def_60_sc")
aggr(missing_pattern,prop = FALSE, numbers = TRUE)


# case-wise deletion MCAR - kabakoff - 427 - 0.02%loss
application_data_missed <- application_data_final %>% filter(!is.na(obs_30_cnt_social_circle))


# Replace NA's in occupation type with "Occupation_unknown" (suite - occupation type)
application_data_missed$occupation_type <- as.character(application_data_missed$occupation_type)
application_data_missed$occupation_type[is.na(application_data_missed$occupation_type)] <- "Occupation_unknown"


# Replace NA's in osuite type with "Suite_unknown"  (suite - occupation type)
application_data_missed$name_type_suite <- as.character(application_data_missed$name_type_suite)
application_data_missed$name_type_suite[is.na(application_data_missed$name_type_suite)] <- "Suite_unknown"

# Replace NA's in gender type with "Gender_unknown"
application_data_missed$code_gender <- as.character(application_data_missed$code_gender)
application_data_missed$code_gender[application_data_missed$code_gender == "XNA"] <- "Gender_unknown"


# Replace NA's in organisation type with "Organisation_unknown"
application_data_missed$organization_type <- as.character(application_data_missed$organization_type)
application_data_missed$organization_type[application_data_missed$organization_type == "XNA"] <- "Organisation_unknown"


# change datatype imputed columns to factor  "occupation_type","name_type_suite"
application_data_missed[,c("occupation_type","name_type_suite","code_gender","organization_type")] <- 
  lapply(application_data_missed %>% select(occupation_type,name_type_suite,code_gender,organization_type), factor)



###

# Function to extract correlation between variables column wise with top correlations
extract_correlation <- function(data_frame,feature_name,type_positive) {
  
  
  # correlation for numerical features - melt the dataframe
  correlation_numericals <- cor(na.omit(data_frame %>%
                                          select(all_of(numerical_features_update)))) %>% melt() %>% as.data.frame()
  # extract datframe with selected feature 
  df <- correlation_numericals %>% filter(Var1 == feature_name | Var2 == feature_name)
  # remove var1 == var2 
  df <- df %>% filter(Var1 != Var2)
  df <- df[!duplicated(df$value),]
  
  # order data frame by positive or negative correlation 
  if (type_positive == TRUE) (
    df <- df %>% arrange(desc(value))
  )
  else if(type_positive == FALSE)(
    df <- df %>% arrange(value)
  )
  return(df)
}

### Replace NA in amt_annuity

# high correlation between amt_goods_price, amt_annuity and amt_credit
head(extract_correlation(application_data_missed,"amt_annuity",type_positive = TRUE))

# linear model to fill NA in annuity 
linear_model_annuity <- lm(amt_annuity ~ amt_goods_price + amt_credit,
                           data = na.omit(application_data_missed))


# summary of linear model
summary(linear_model_annuity)

# subset dataframe where amt annuity is NA
subset_amt_annuity <- application_data_missed %>% 
  filter(is.na(amt_annuity))

# predict missed values in amt_annuity
predicted_annuity <- predict(linear_model_annuity,subset_amt_annuity)

# replace NA with predicted values 
subset_amt_annuity$amt_annuity <- predicted_annuity

# imputed dataframe of amount annuity
application_data_imputed <- application_data_missed %>% filter(!is.na(amt_annuity))
application_data_imputed <- rbind(application_data_imputed,subset_amt_annuity)

### Replace NA in amt_goods

# correlation for amt_goods
head(extract_correlation(application_data_imputed,"amt_goods_price",type_positive = TRUE))

# linear model to fill NA in amt_goods_price
linear_model_goods <- lm(amt_goods_price ~ amt_annuity + amt_credit,
                         data = na.omit(application_data_imputed))

# summary of linear model
summary(linear_model_goods)

# subset dataframe where amt goods is NA 
subset_amt_goods <- application_data_imputed %>% 
  filter(is.na(amt_goods_price))

# predict missed values in amt_goods
predicted_goods <- predict(linear_model_goods,subset_amt_goods)

# replace NA with predicted values 
subset_amt_goods$amt_goods_price <- predicted_goods


# imputed dataframe of amount annuity
application_data_imputed_two  <- application_data_imputed %>% filter(!is.na(amt_goods_price))
application_data_imputed_two  <- rbind(application_data_imputed_two,subset_amt_goods)

head(extract_correlation(application_data_missed,"amt_annuity",type_positive = TRUE))


# Replace NA in ext_source_2 with mean value of the column 0.001% missing
application_data_imputed_two$ext_source_2 <- if_else(is.na(application_data_imputed_two$ext_source_2),
                                                     mean(na.omit(application_data_imputed_two$ext_source_2)),
                                                     application_data_imputed_two$ext_source_2)


# Replace NA in ext_source_3 with KNN 6.73% missing

# many negative correlation with "bureau" vartiables
head(extract_correlation(application_data_imputed_two,"ext_source_3",type_positive = FALSE),10)

# predict knn of ext_source_3 with numerical vars as predictor variables
# by training on randomly on 15000 observations

# select numerical features

knn_vars <- numerical_features_update

# scaling function
min_max <- function(x) {
  tx <- (x - min(x)) / (max(x) - min(x))
  return(tx)
}

# scale data
scaled_data_knn <- na.omit(application_data_imputed_two) %>% select(all_of(knn_vars))
scaled_data_knn <- apply(scaled_data_knn %>% select(-ext_source_3), 2, min_max)
scaled_data_knn <- cbind(na.omit(application_data_imputed_two) %>% 
                           select(ext_source_3),scaled_data_knn)

# randomly sampling for training on 30000 observations 
samp <- sample(nrow(scaled_data_knn),nrow(scaled_data_knn))

# subset train
train_knn <- scaled_data_knn[samp[1:30000],]

# subset test
test_knn  <- scaled_data_knn[samp[!samp %in% samp[1:30000]][1:30000],]



# Training model on train and test split
knn_model <- knn(train = train_knn %>% select(c(all_of(knn_vars))),
                 test = test_knn %>% select(c(all_of(knn_vars))),
                 cl = train_knn$ext_source_3, k = 5)


# Root mean square error
rmse_knn <- round(rmse(as.numeric(as.vector(knn_model)),test_knn$ext_source_3),3)
print(paste("RMSE of the model :  ", rmse_knn))


# Apply model on missing_data of ext_source_3 
subset_ext_source_3 <- application_data_imputed_two %>% filter(is.na(ext_source_3)) 
subset_ext_source_3_scaled <- subset_ext_source_3 %>% select(all_of(knn_vars))
subset_ext_source_3_scaled <- as.data.frame(apply(subset_ext_source_3_scaled %>% select(-ext_source_3), 2, min_max))
subset_ext_source_3_scaled$ext_source_3 <- 0


# Training model on missed data 
knn_model_imp <- knn(train = train_knn %>% select(c(all_of(knn_vars))),
                     test  = subset_ext_source_3_scaled  %>% select(c(all_of(knn_vars))),
                     cl = train_knn$ext_source_3, k = 10)


# Replace the predicted value of ext source to subsetted data frame
subset_ext_source_3$ext_source_3 <- as.numeric(as.vector(knn_model_imp))


# row bind imputed dataframe of ext_source_3 to main dataframe
application_data_imputed_three <- application_data_imputed_two %>% filter(!is.na(ext_source_3))
application_data_imputed_three <- rbind(application_data_imputed_three,subset_ext_source_3)


###################################################################
################ Type conversion grouping categorical #############
###################################################################

# Group categories in organistion types together and create new column


# business types vector
bus_types <- c("Business Entity Type 1","Business Entity Type 2","Business Entity Type 3")

# industry types vector
ind_types <- c("Industry: type 1","Industry: type 2","Industry: type 3","Industry: type 4",
               "Industry: type 5","Industry: type 6","Industry: type 7","Industry: type 8",
               "Industry: type 9","Industry: type 10","Industry: type 11","Industry: type 12",
               "Industry: type 13")

# trade types vector
trd_types <- c("Trade: type 1","Trade: type 2","Trade: type 3","Trade: type 4",
               "Trade: type 5","Trade: type 6","Trade: type 7")

# transport types vector
trp_types <- c("Transport: type 1","Transport: type 2","Transport: type 3","Transport: type 4")

# create a new column grouped 
application_data_imputed_three$organization_type_grouped <- as.character(application_data_imputed_three$organization_type)

# Group column by category vectors
application_data_imputed_three$organization_type_grouped[application_data_imputed_three$organization_type_grouped %in% 
                                                           bus_types] <- "Business"
application_data_imputed_three$organization_type_grouped[application_data_imputed_three$organization_type_grouped %in% 
                                                           ind_types] <- "Industry"
application_data_imputed_three$organization_type_grouped[application_data_imputed_three$organization_type_grouped %in% 
                                                           trd_types] <- "Trade"
application_data_imputed_three$organization_type_grouped[application_data_imputed_three$organization_type_grouped %in% 
                                                           trp_types] <- "Transport"



###########################################
################ Outliers ################
##########################################

# Based on initial data exploration and meta data provided built a few rules that identifies
# potential outliers
# 
application_outlier <- application_data_imputed_three

#
rules <- validator(
  
  # instances where days features that are greater than 0
  Rule1 = (days_birth > 0 | days_employed > 0 | days_registration >0 |
             days_id_publish > 0 | days_last_phone_change > 0),
  
  # own a car but car age is 0
  Rule2 = (flag_own_car == "Y" & own_car_age <= 0),
  
  # children count greater than 15
  Rule3 = (cnt_children > 15),
  
  # family count greater than 16
  Rule4 = (cnt_children > 16),
  
  # income that's greater than 99.995% quantile value
  Rule5 = (amt_income_total) > quantile(application_outlier$amt_income_total, 0.99995),
  
  # annuity that's greater than 99.995% quantile value
  Rule6 = (amt_annuity) > quantile(application_outlier$amt_annuity, 0.99995),
  
  # goods price that's greater than 99.995% quantile value
  Rule7 = (amt_goods_price) > quantile(application_outlier$amt_goods_price, 0.99995),
  
  # credit that's greater than 99.995% quantile value
  Rule8 = (amt_credit) > quantile(application_outlier$amt_credit, 0.99995),
  
  # days employed == 365243 (i.e for ocuppation unknown) & type of ocuupation is not unknown
  Rule9 = (days_employed == 365243 & occupation_type != "Occupation_unknown"),
  
  # observation of client's social surroundings with observable 30 DPD  greater than 50 people
  Rule10 = (obs_30_cnt_social_circle > 50 | obs_60_cnt_social_circle > 50),
  
  # car age greater than 65
  Rule11 = (own_car_age > 65),
  
  # Number of enquiries to Credit Bureau about the client 3 month before application greater than 10
  Rule12 = (amt_req_credit_bureau_qrt > 10),
  
  # gender not in m and f
  Rule13 = (code_gender != "F" & code_gender != "M"),
  
  # income type unemployed
  Rule14 = (name_income_type == "Unemployed")
  
)


# checking results with sk_id_curr as unique identifier
checkResults    <-  confront(application_outlier,rules, key = "sk_id_curr")
checkResults_DF <-  as.data.frame(checkResults)

# Identifying data quality issues
potential_outliers   <-  subset(checkResults_DF, value == "TRUE")
row.names(potential_outliers) <- NULL

# Number of potential outliers by each rule
potential_outliers %>% group_by(name) %>% summarise("count" = n()) %>%
  arrange(desc(count))

# plot percentage in data
barplot(checkResults,
        stack_by = c("passes", "fails", "nNA"),
        main = "Data quality Issues") 


# function to return id's identified by rules
outlier_ids <- function(rule) {
  ids <- potential_outliers %>% 
    filter(name == rule) %>% select(sk_id_curr) 
  ids <- ids$sk_id_curr
  return(ids)
}


# Rule1 -  # instances where days features that are greater than 0
# replace 365243 with 0 
application_outlier$days_employed <- ifelse(application_outlier$days_employed == 365243,
                                            0,application_outlier$days_employed)



# Rule2 - own a car but car age is 0
# New one haven't passed a year - replace value with specific value 0.5 that belongs to below one year 
# 0 represents NA value in data that belongs to subset of data that doest own a car
application_outlier %>% filter(sk_id_curr %in% outlier_ids("Rule2")) %>% select(own_car_age,flag_own_car)
application_outlier[application_outlier$sk_id_curr %in% outlier_ids("Rule2"),"own_car_age"] <- 0.5


# Rule3 & Rule4 - children count greater than 15 (implausible) , days_birth , cnt_fam_members > 20
# remove the two instances
application_outlier %>% filter(sk_id_curr %in% outlier_ids("Rule3")) %>% 
  select(days_birth,cnt_children,cnt_fam_members)
application_outlier <- application_outlier %>% filter(cnt_children < 15)


# Rule5 - # income that's greater than 99.995% quantile value
# Plot box plot for income variables
#%>% filter(!sk_id_curr %in% outlier_ids("Rule5")) 
inc_vars_bp <- application_outlier %>% select(amt_income_total,amt_credit, amt_annuity,amt_goods_price) %>% melt()

plot1 <- ggplot(data = inc_vars_bp) + geom_boxplot(aes(x = variable,y = value),
                                                   fill="slateblue",alpha = .4)
plot1 + facet_wrap(~variable,scales = "free",nrow = 1,ncol = 4) + 
  ggtitle("Amount variables box plot",subtitle = "Plot 1") + 
  theme(plot.title = element_text(hjust = 0.5)) 


# Remove instances which seems implausible [Laborers, Police, Drivers ]
application_outlier %>% filter(sk_id_curr %in% outlier_ids("Rule5")) %>% 
  arrange(desc(amt_income_total)) %>% select(amt_income_total,amt_annuity,amt_credit,
                                             amt_goods_price,occupation_type,organization_type,sk_id_curr)

# Remove instances 
remove_inc_ids <- c(114967,252084,317748,399467,445335)
application_outlier <- application_outlier %>% filter(!sk_id_curr %in% remove_inc_ids)



# Rule6 - annuity that's greater than 99.995% quantile value
application_outlier %>% filter(sk_id_curr %in% outlier_ids("Rule6")) %>% 
  arrange(desc(amt_annuity)) %>% select(amt_income_total,amt_annuity,amt_credit,
                                        amt_goods_price,occupation_type,organization_type,sk_id_curr)


# Rule7 - goods price that's greater than 99.995% quantile value
application_outlier %>% filter(sk_id_curr %in% outlier_ids("Rule7"))%>% 
  arrange(desc(amt_goods_price)) %>% select(amt_income_total,amt_annuity,amt_credit,
                                            amt_goods_price,occupation_type,organization_type,sk_id_curr)

# Rule8 - credit that's greater than 99.995% quantile value
application_outlier %>% filter(sk_id_curr %in% outlier_ids("Rule8"))%>% 
  arrange(desc(amt_goods_price)) %>% select(amt_income_total,amt_annuity,amt_credit,
                                            amt_goods_price,occupation_type,organization_type,sk_id_curr)


# Rule9 - days employed == 365243 (i.e for ocuppation unknown) & type of ocuupation is not unknown
# Replace value with occupation unknown 
application_outlier %>% filter(sk_id_curr %in% outlier_ids("Rule9"))
application_outlier[application_outlier$sk_id_curr %in%  outlier_ids("Rule9"),"occupation_type"] <- "Occupation_unknown"



# Rule10 - observation of client's social surroundings with observable 30 DPD  greater than 50 people
table(application_outlier$obs_30_cnt_social_circle)
table(application_outlier$def_30_cnt_social_circle)

#
application_outlier %>% filter(sk_id_curr %in% outlier_ids("Rule10")) %>% 
  select(organization_type,obs_30_cnt_social_circle,def_30_cnt_social_circle,
         obs_60_cnt_social_circle,def_60_cnt_social_circle)

# Remove the instance 
application_outlier <- application_outlier %>% filter(!sk_id_curr %in% outlier_ids("Rule10"))


# Rule11 - car age greater than 65
# replace car age with median value of occupation type where car age is 91
table(application_outlier$own_car_age)
application_outlier %>% filter(sk_id_curr %in% outlier_ids("Rule11")) %>% select(amt_income_total,amt_credit,
                                                                                 own_car_age,occupation_type)

application_outlier[application_outlier$own_car_age == 91,"own_car_age"] <-
  median(application_outlier[application_outlier$occupation_type == "Drivers" & 
                               application_outlier$flag_own_car == "Y","own_car_age"])


# Rule12 - Number of enquiries to Credit Bureau about the client 3 month before application greater than 10
# Replace with how many enquires made last month
table(application_outlier$amt_req_credit_bureau_qrt)
application_outlier %>% filter(sk_id_curr %in% outlier_ids("Rule12"))
application_outlier[application_outlier$sk_id_curr %in% outlier_ids("Rule12"),"amt_req_credit_bureau_qrt"] <- 
  application_outlier[application_outlier$sk_id_curr %in% outlier_ids("Rule12"),"amt_req_credit_bureau_mon"]


# Rule13 - gender not in m and f
# drop instances
application_outlier %>% filter(sk_id_curr %in% outlier_ids("Rule13"))
application_outlier <- application_outlier %>% filter(!code_gender == "Gender_unknown")



# Rule14 - income type unemployed
# remove one instance 
table(application_outlier$name_income_type)
application_outlier <- application_outlier %>% filter(!name_income_type == "Unemployed")

# drop levels of deleted factor levels
application_outlier <- droplevels(application_outlier)


write.csv(application_outlier,"application_data_prepared.csv",row.names = FALSE)



#################################################################
################ Exploratory Data Analysis - EDA ################
##################################################################

application_prepared <- application_outlier 

# create new feature for target name
application_prepared$target_name <- ifelse(application_prepared$target == 1,
                                           "Defaulters","Non-Defaulters")


############  plot2 - Target Proportion ######### 

# target proportion Non - Defaulters - 92.15% ; Defaulters - 7.85%
target_prop <- data.frame(round(table(application_prepared$target_name)/nrow(application_prepared)*100,2))
colnames(target_prop) <- c("Target","Percentage")


# plot target proportion 
plot2 <- ggplot(data = target_prop) + geom_bar(aes(x = Percentage,y = Target,
                                                   fill = Target),
                                               stat = "identity",width = 0.45)
plot2 + ggtitle("Target Variable proportion",subtitle = "Plot 2") +
  theme(legend.position = "none",plot.title = element_text(hjust = 0.5)) 



############  plot3  - Distribution of Amount Variables ######### 

# Distribution of Amount Variables
amt_vars_grp <- application_prepared %>% 
  select(target_name,amt_annuity,amt_credit,amt_goods_price,amt_income_total) %>% 
  melt(id = "target_name")

# plot features
plot3 <- ggplot(data = amt_vars_grp) + geom_density(aes(x = value,
                                                        group = target_name,
                                                        fill = target_name),
                                                    adjust=1.5, alpha=.4)
plot3 + facet_wrap(~variable, scales = "free") + 
  ggtitle("Distribution of Amount Features",subtitle = "Plot 3") +
  theme(legend.position = "bottom",
        plot.title = element_text(hjust = 0.5,size = 11,)) +
  xlab("") 


# function to plot pie charts

plot_pie <- function(feature,title) {
  
  prop_def <- application_prepared  %>% filter(target == 1)  %>%
    select(target_name,feature)
  prop_def <- data.frame(round(table(prop_def[,2])/nrow(prop_def)*100,1))
  prop_def$taget <- "Defaulters"
  
  prop_non_def <- application_prepared  %>% filter(target == 0)  %>%
    select(target_name,feature)
  prop_non_def <- data.frame(round(table(prop_non_def[,2])/nrow(prop_non_def)*100,1))
  prop_non_def$taget <- "Non-Defaulters"
  
  prop_fin <- rbind(prop_def,prop_non_def)
  colnames(prop_fin) <- c("feature","percentage","target")
  
  plotted <- ggplot(prop_fin,aes(x = "",y = percentage,fill = feature)) +
    geom_bar(width = 1,stat = "identity",color = "white") +
    coord_polar(theta = "y",start = 0) +
    geom_text(aes(label = paste(percentage, "%"),
                  y = percentage/1.3, 
    ),color = "black",alpha = .8) +
    facet_wrap(~target) +
    theme(legend.position = "bottom",legend.title = element_blank(),
          plot.title = element_text(hjust = 0.5,size = 11),
          panel.grid  = element_blank(),
          axis.text = element_blank(),
          axis.ticks = element_blank()) + 
    ggtitle(title) +xlab("") + ylab("") 
  return(plotted)
  
  
}


############  plot4  - Contract status by Target type ######### 
plot_pie("name_contract_type","Contract status by Target type")

############  plot5  - Gender by Target type ######### 
plot_pie("code_gender","Gender by Target type")

############  plot6  - Car ownership by Target type ######### 
plot_pie("flag_own_car","Car ownership by Target type")


############  Function for dodged bar chart ######### 
bar_dodge <- function(feature,title) {
  
  df <- data.frame(table(application_prepared[,feature],application_prepared$target_name))
  colnames(df) <- c("feature","target","count")
  
  plot_bar_dg <- ggplot(data = df) + geom_bar(aes(x = count,y = reorder(feature,count),fill = target),
                                              stat = "identity",position = "Dodge",width = .9,
                                              color = "white")
  
  plot_bar_dg <- plot_bar_dg + ylab(feature) + 
    ggtitle(title) + 
    theme(legend.position = "bottom",legend.title = element_blank(),
          plot.title = element_text(hjust = 0.5,size = 11))
  
  return(plot_bar_dg)
}

############  plot7  - Frequency count of occupation type ######### 
bar_dodge("occupation_type","Frequency count of occupation type")

############  plot8 - Frequency count of income type ######### 
bar_dodge("name_income_type","Frequency count of income type")

############  plot9  - Distribution of Amount Variables ######### 
bar_dodge("name_family_status","Frequency count of family status")

############  plot10  - Distribution of Amount Variables ######### 
bar_dodge("name_education_type","Frequency count of education type")

############  plot11  - Distribution of Amount Variables ######### 
bar_dodge("organization_type_grouped","Frequency count of organization type")


############  plot12  - Document submission by target feature ######### 
fl_dc <- application_prepared %>% select(starts_with("flag_document"),target_name) %>%
  melt(id.var = "target_name")

fl_dc$value <- if_else(fl_dc$value == 1, "Yes","No")

ggplot(data = fl_dc) + geom_bar(aes(x = value,fill = target_name),stat = "count",
                                color = "white",width = .95,position = "Dodge") + 
  facet_wrap(~variable)  + xlab("") + ggtitle("Document submission by target feature") + 
  theme(plot.title = element_text(hjust = 0.5,size = 11))


############  plot13  - Defaulters frequency by week and hour of the day ######### 
def_wk_hr <- application_prepared %>% filter(target == 0)
def_wk_hr <- data.frame(table(def_wk_hr$weekday_appr_process_start,
                              def_wk_hr$hour_appr_process_start))

ggplot(def_wk_hr,aes(x = Var2,y = Var1,fill = Freq))  + 
  scale_fill_distiller(palette = "RdYlGn") +
  geom_tile() + xlab("Hour of the day") +
  ylab("Week of the day") + 
  ggtitle("Defaulters frequency by week and hour of the day")+ 
  theme(plot.title = element_text(hjust = 0.5,size = 11))


############  plot14  - Distribution of External source Features ######### 

ext_vars_grp <- application_prepared %>% 
  select(target_name,ext_source_2,ext_source_3) %>% 
  melt(id = "target_name")

plot14 <- ggplot(data = ext_vars_grp) + geom_density(aes(x = value,
                                                         group = target_name,
                                                         fill = target_name),
                                                     adjust=1.5, alpha=.4)
plot14 + facet_wrap(~variable, scales = "free",nrow = 1,ncol = 2) + 
  ggtitle("Distribution of External source Features") +
  theme(legend.position = "bottom",
        plot.title = element_text(hjust = 0.5,size = 11)) +
  xlab("") 


############  plot15  - Distribution of External source Features ######### 
days_vars_grp <- application_prepared %>% 
  select(target_name,days_birth,days_employed,days_registration,days_id_publish,days_last_phone_change) %>% 
  melt(id = "target_name")

days_vars_grp

plot15 <- ggplot(data = days_vars_grp) + geom_boxplot(aes(y = value,
                                                          x = variable,
                                                          fill = target_name),
                                                      alpha=.4)

plot15 + facet_wrap(~variable, scales = "free",nrow = 1,ncol = 5) + 
  ggtitle("Box plot for days Features") +
  theme(legend.position = "bottom",
        plot.title = element_text(hjust = 0.5,size = 11)) +
  xlab("") 



######################################
############### Sampling ############
######################################

# Remove unique id feature, organization_type(replaced with organistion_type_grouped), target_name (replaced with target)
model_final <- application_prepared %>% select(-c(target_name,sk_id_curr,organization_type))

# covert target to factor
model_final$target<- as.factor(model_final$target)

# split training and test data with 70/30
set.seed(1)
sampling_rows <- sample(nrow(model_final),nrow(model_final)*0.7)

#training and test data
training_data <- model_final[sampling_rows,]
testing_data <- model_final[-sampling_rows,]

# proportion of target in training and test datasets
prop.table(table(training_data$target)) * 100
prop.table(table(testing_data$target)) * 100

# under sampling training data with downsampling majority class
under_sampled <- ovun.sample(target~.,
                             data = training_data,
                             method = "under",
                             N = training_data %>% filter(target == "1") %>% nrow()/0.5,
                             seed = 1)
under_sampled_data <- under_sampled$data


#  proportion of target in under sampled data
prop.table(table(under_sampled_data$target))*100

############################################################
############### Model building - Random Forest ############
############################################################

#######################################
############# Baseline model ###########
#######################################

# formula with all variables
formula_model <- reformulate(termlabels = colnames(model_final)[!colnames(model_final) %in% "target"],
                             response = "target")


# Baseline model with all features 
# Number of trees: 50
# No. of variables tried at each split: 
set.seed(1)
random_forest_baseline <- randomForest(formula = formula_model,
                                       data = under_sampled_data,
                                       ntree = 50,
                                       replace = FALSE)





# get a sample tree
plot.getTree(random_forest_baseline,k=1,labelVar = TRUE)


# plot out of bag errors for baseline model
plot(random_forest_baseline,col = c("black","red","blue"))
legend(lty=c(1,2,3),"topright",
       legend = c("OOB","0 - Non Defaulters", "1 - Defaulters"),
       col = c("black","red","blue"))


random_forest_baseline

# Feature importance 
varImpPlot(random_forest_baseline)

getTree(random_forest_baseline,1,labelVar = TRUE)


# prediction of baseline model on testing data set
predicted_baseline <- predict(random_forest_baseline,
                              newdata = testing_data %>% select(-target),
                              type = "response")

# predicted probabilities on baseline
predicted_prob_baseline <- predict(random_forest_baseline,
                                   newdata = testing_data %>% select(-target),
                                   type = "prob") %>% as.data.frame()
colnames(predicted_prob_baseline) <- c("zero_prob","one_prob")
predicted_prob_baseline_one <- predicted_prob_baseline$one_prob


# Classification report baseline model
report_baseline <- confusionMatrix(predicted_baseline,
                                   testing_data$target,
                                   positive = "1")

report_baseline

#######################################################################
############# Model - Parameter Tuning and feature selection ###########
########################################################################

# select features that lie above  80% quantile range of Mean Decrease Gini
feature_importances <- random_forest_baseline$importance %>% as.data.frame()
feature_importances$features <- rownames(feature_importances)
rownames(feature_importances) <- NULL
feature_importances <- feature_importances %>% arrange(desc(MeanDecreaseGini))
feature_importances
feature_importances_select <- feature_importances %>% 
  filter(MeanDecreaseGini > quantile(feature_importances$MeanDecreaseGini,0.7)) 
feature_importances_select <- feature_importances_select$features

getTree(random_forest_tuned,k = 1,labelVar = TRUE)

# formula on selected features
formula_model_update <- reformulate(termlabels = feature_importances_select,
                                    response = "target")




# Tune model with selected features
# mtry - minimum variables used at each split
# find optimal value of mtry with lowest oob error
# Select trees where oob stabilises
set.seed(1)
tune_model <- tuneRF(under_sampled_data %>% 
                       select(all_of(feature_importances_select)),
                     under_sampled_data$target,
                     stepFactor = 1.5,
                     improve = 0.01,
                     trace = TRUE,
                     plot = TRUE,
                     ntreeTry = 1000)


# select mtry where oob is minimum
minimum_mtry <- tune_model[tune_model[,2] == min(tune_model[,2]),1]


# model on selected features and minimum mtry with oob value
set.seed(1)
random_forest_tuned <- randomForest(formula = formula_model_update,
                                    data = under_sampled_data %>% 
                                      select(c("target",all_of(feature_importances_select))),
                                    mtry = minimum_mtry,
                                    ntree = 700,
                                    importance = TRUE)

# rules from 700 trees
tree_list <- RF2List(random_forest_tuned)

# extract rules from 700 trees
extract_rules <- extractRules(tree_list,
                              under_sampled_data %>% select(all_of(feature_importances_select)))

# measure rules - (len = no of vars, pred = out come class, freq = percentage of data)
rule_metric <- getRuleMetric(extract_rules,under_sampled_data %>% select(feature_importances_select),
                             under_sampled_data$target)

# readable rules
readable_rules <- presentRules(rule_metric,feature_importances_select) %>% as.data.frame()


readable_rules %>% arrange(desc(freq)) %>% head(10)



# plot oob error by trees 
plot(random_forest_tuned,col = c("black","white","white"))
legend(lty=c(1),"topright",
       legend = c("OOB"))
legend(lty=c(1,2,3),"topright",
       legend = c("OOB","0 - Non Defaulters", "1 - Defaulters"),
       col = c("black","white","white"))


# prediction of feature selected model on testing data set
predicted_tuned <- predict(random_forest_tuned,
                           newdata = testing_data %>% select(all_of(feature_importances_select)),
                           type = "response")


# predicted probabilities onfeature selected model
predicted_prob_tuned <- predict(random_forest_tuned,
                                newdata = testing_data %>% select(all_of(feature_importances_select)),
                                type = "prob") %>% as.data.frame()

colnames(predicted_prob_tuned) <- c("zero_prob","one_prob")
predicted_prob_tuned_one <- predicted_prob_tuned$one_prob


# Classification report feature selected model
report_tuned <- confusionMatrix(predicted_tuned,
                                testing_data$target,
                                positive = "1")

report_tuned
report_baseline

###############################################
############# Performance evaluation ##########
###############################################

# data frame for probabilities
df_prob_models <- data.frame(Baseline = predicted_prob_baseline_one,
                             Tuned_Model = predicted_prob_tuned_one)



# data frame for actuals
df_actual <- data.frame(Baseline = testing_data$target,
                        Tuned_Model = testing_data$target)



# baseline roc
baseline_roc_pred <- prediction(df_prob_models,df_actual)
baseline_roc_perf <- performance(baseline_roc_pred,"tpr","fpr")


# Area under curve 
df_auc <- data.frame(Model = names(df_prob_models),
                     AUC   = c(performance(baseline_roc_pred, measure = "auc")@y.values[[1]][1],
                               performance(baseline_roc_pred, measure = "auc")@y.values[[2]][1]))

df_auc$AUC <- round(df_auc$AUC*100,2)

# plot ROC
opar <- par(cex = .7)
par(pty = 's')
plot(baseline_roc_perf,
     col = as.list(c("red", "blue","brown")),
     linestyle = "dotted")

abline(a = 0, b = 1, lty = 2, col = 'black')
grid()
legend("bottomright", 
       paste(names(df_prob_models),
             c(" - AUC - "), 
             c(df_auc$AUC),
             c("%")),
       col = c("red", "blue"),lty = 1,
       bty = 'n')
title("ROCR Curve")


# compare metrics
metrics_compare <- data.frame("Baseline_model" = report_baseline$byClass,
                              "Tuned_model" = report_tuned$byClass)

metrics_compare <- metrics_compare %>% mutate("Metric" = rownames(metrics_compare))

metrics_compare <- metrics_compare %>% filter(Metric %in% c("Sensitivity","Specificity",
                                                            "Precision","Recall","Balanced Accuracy"))

metrics_compare
rownames(metrics_compare) <- NULL

metrics_compare_pt <- metrics_compare %>% melt()
ggplot(data = metrics_compare_pt) + geom_bar(aes(y = Metric,x = value,fill = variable),
                                             stat = "identity",position = "Dodge",
                                             width = .5) +
  ggtitle("Model comparision") +
  theme(legend.position = "bottom",legend.title = element_blank(),
        plot.title = element_text(hjust = 0.5,size = 11))

metrics_compare

