---
title: "Homelessness"
output: 
  html_document: 
    keep_md: true
editor_options: 
  chunk_output_type: inline
---
## Introduction

The 2020 [point-in-time count](https://www.kingcounty.gov/elected/executive/constantine/news/release/2020/July/01-homeless-count.aspx) of people experiencing homelessness for Seattle/King County was 11,751. This represents a 5% increase over the 2019 count and reflects similar trend across many counties in the western U.S. A step towards addressing homelessness is improving our understanding of the relationship between local housing market factors and homelessness. 

The U.S. Department of Housing and Urban Development (HUD) produced a report in 2019 [Market Predictors of Homelessness](https://www.huduser.gov/portal/sites/default/files/pdf/Market-Predictors-of-Homelessness.pdf) that describes a model-based approach to understanding of the relationship between local housing market factors, policies, demographics, climate and homelessness. Our project is motivated by the goals of the HUD study:

"To continue progressing toward the goals of ending and preventing homelessness, we must further our knowledge of the basic community-level determinants of homelessness. The primary objectives of this study are to (1) identify market factors that have established effects on homelessness, (2) construct and evaluate empirical models of community-level homelessness.."

We will investigate whether there are alternative modeling approaches that outperform the models described in the HUD report. We will be testing various different approaches to find the best indicators of homelessness by comparing the root mean square error(RMSE) of the models. These include: multiple linear regression, lasso regression, ridge regression, and XGBoost. As well we will be doing some best subset selection with BIC, adjusted R-squared, CP, and CV.  



## Data Collection

The data for this project are described in HUD's report [Market Predictors of Homelessness](https://www.huduser.gov/portal/sites/default/files/pdf/Market-Predictors-of-Homelessness.pdf) in the section titled DATA.

I will refer you to this section of the HUD report for a detailed description of the sources of the data and how they were processed.


### Load necessary packages


```r
#skimr provides a nice summary of a data set
library(skimr)
#leaps will be used for model selection
library(leaps)
#tidyverse contains packages we will use for processing and plotting data
library(tidyverse)
```

```
## ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.0 ──
```

```
## ✓ ggplot2 3.3.3     ✓ purrr   0.3.4
## ✓ tibble  3.1.1     ✓ dplyr   1.0.5
## ✓ tidyr   1.1.3     ✓ stringr 1.4.0
## ✓ readr   1.4.0     ✓ forcats 0.5.1
```

```
## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
## x dplyr::filter() masks stats::filter()
## x dplyr::lag()    masks stats::lag()
```

```r
#readxl lets us read Excel files
library(readxl)
#GGally has a nice pairs plotting function
library(GGally)
```

```
## Registered S3 method overwritten by 'GGally':
##   method from   
##   +.gg   ggplot2
```

```r
#corrplot has nice plots for correlation matrices
library(corrplot)
```

```
## corrplot 0.84 loaded
```

```r
#gridExtra
library(gridExtra)
```

```
## 
## Attaching package: 'gridExtra'
```

```
## The following object is masked from 'package:dplyr':
## 
##     combine
```

```r
#glmnet is used to fit glm's. It will be used for lasso and ridge regression models.
library(glmnet)
```

```
## Loading required package: Matrix
```

```
## 
## Attaching package: 'Matrix'
```

```
## The following objects are masked from 'package:tidyr':
## 
##     expand, pack, unpack
```

```
## Loaded glmnet 4.1-1
```

```r
#tidymodels has a nice workflow for many models. We will use it for XGBoost
library(tidymodels)
```

```
## ── Attaching packages ────────────────────────────────────── tidymodels 0.1.3 ──
```

```
## ✓ broom        0.7.6      ✓ rsample      0.0.9 
## ✓ dials        0.0.9      ✓ tune         0.1.4 
## ✓ infer        0.5.4      ✓ workflows    0.2.2 
## ✓ modeldata    0.1.0      ✓ workflowsets 0.0.2 
## ✓ parsnip      0.1.5      ✓ yardstick    0.0.8 
## ✓ recipes      0.1.16
```

```
## ── Conflicts ───────────────────────────────────────── tidymodels_conflicts() ──
## x gridExtra::combine() masks dplyr::combine()
## x scales::discard()    masks purrr::discard()
## x Matrix::expand()     masks tidyr::expand()
## x dplyr::filter()      masks stats::filter()
## x recipes::fixed()     masks stringr::fixed()
## x dplyr::lag()         masks stats::lag()
## x Matrix::pack()       masks tidyr::pack()
## x yardstick::spec()    masks readr::spec()
## x recipes::step()      masks stats::step()
## x Matrix::unpack()     masks tidyr::unpack()
## x recipes::update()    masks Matrix::update(), stats::update()
## ● Use tidymodels_prefer() to resolve common conflicts.
```

```r
#xgboost lets us fit XGBoost models
library(xgboost)
```

```
## 
## Attaching package: 'xgboost'
```

```
## The following object is masked from 'package:dplyr':
## 
##     slice
```

```r
#vip is used to visualize the importance of predicts in XGBoost models
library(vip)
```

```
## 
## Attaching package: 'vip'
```

```
## The following object is masked from 'package:utils':
## 
##     vi
```

```r
library(broom)
library(sjPlot)
```

```
## Install package "strengejacke" from GitHub (`devtools::install_github("strengejacke/strengejacke")`) to load all sj-packages at once!
```

```r
#Set the plotting theme
theme_set(theme_gray())


#Set seed for consistent results
set.seed(1)
```


### Examine the data dictionary

*The data dictionary `HUD TO3 - 05b Analysis File - Data Dictionary.xlsx` contains descriptions of all variables in the data set.*

**$\rightarrow$ Load the data dictionary (call it `dictionary`) and view its contents using the function `View`.**


```r
dictionary <- read_xlsx("HUD TO3 - 05b Analysis File - Data Dictionary.xlsx")
```


## Data Preparation

### Load the data 

*The HUD data set is contained in the file `05b_analysis_file_update.csv`.*

**$\rightarrow$ Load the data set contained in the file `05b_analysis_file_update.csv` and name the data frame `df`.**


```r
df <- read_csv("05b_analysis_file_update.csv")
```

```
## 
## ── Column specification ────────────────────────────────────────────────────────
## cols(
##   .default = col_double(),
##   cocnumber = col_character(),
##   state_abr = col_character(),
##   dem_health_ins_acs5yr_2012 = col_logical()
## )
## ℹ Use `spec()` for the full column specifications.
```

Refer to the data dictionary and the [HUD report]((https://www.huduser.gov/portal/sites/default/files/pdf/Market-Predictors-of-Homelessness.pdf)) to understand what variables are present.


**$\rightarrow$ Add variables to the list `variable_names` that we will keep for the analysis.**


```r
#Search through data dictionary to find other variables to include

variable_names <- c("year", "cocnumber",
  
"pit_tot_hless_pit_hud", "pit_tot_shelt_pit_hud", "pit_tot_unshelt_pit_hud","dem_pop_pop_census",
  
"fhfa_hpi_2009", "ln_hou_mkt_medrent_xt", "hou_mkt_utility_xt", "hou_mkt_burden_own_acs5yr_2017", "hou_mkt_burden_sev_rent_acs_2017", "hou_mkt_rentshare_acs5yr_2017", "hou_mkt_rentvacancy_xt", "hou_mkt_density_dummy", "hou_mkt_evict_count", "hou_mkt_ovrcrowd_acs5yr_2017", "major_city", "suburban",
           
"econ_labor_unemp_rate_BLS", "econ_labor_incineq_acs5yr_2017", "econ_labor_pov_pop_census_share",
           
"hou_pol_hudunit_psh_hud_share", "hou_pol_occhudunit_psh_hud", "hou_mkt_homeage1940_xt",
           
  "dem_soc_black_census", "dem_soc_hispanic_census", "dem_soc_asian_census", "dem_soc_pacific_census", "dem_pop_child_census", "dem_pop_senior_census", "dem_pop_female_census", "dem_pop_mig_census", "d_dem_pop_mig_census_share", "dem_soc_singadult_xt", "dem_soc_singparent_xt", "dem_soc_vet_xt", "dem_soc_ed_lessbach_xt", "dem_health_cost_dart", "dem_health_excesdrink_chr",
           
"env_wea_avgtemp_noaa","env_wea_avgtemp_summer_noaa", "env_wea_precip_noaa", "env_wea_precip_annual_noaa"
)
```


**$\rightarrow$ Select this subset of variables from the full data set. Call the new data frame `df_small`.**


```r
df_small <- df %>% dplyr::select(all_of(variable_names))
```


**$\rightarrow$ Create a new dictionary for this subset of variables.**


```r
dictionary_small <- dictionary %>% filter(Variable %in% variable_names)
```

### Data cleaning

#### Rename variables

I used the data dictionary to create more readable names for the minimal set of variables. You should add in new names for the additional variables you included in the data set.

The data frame with renamed columns is called `df_hud`.


```r
#Add your new names to this list

df_hud <- df_small %>% 
  rename(coc_number = dictionary_small$Variable[2],
    total_sheltered = dictionary_small$Variable[3],
total_unsheltered = dictionary_small$Variable[4],
total_homeless = dictionary_small$Variable[5],
total_population = dictionary_small$Variable[6],
total_female_population = dictionary_small$Variable[7],
total_population_0_19 = dictionary_small$Variable[8],
total_population_65_plus = dictionary_small$Variable[9],
total_black = dictionary_small$Variable[10],
total_asian = dictionary_small$Variable[11],
total_pacific_islander = dictionary_small$Variable[12],
total_latino_hispanic = dictionary_small$Variable[13],
house_price_index_2009 = dictionary_small$Variable[14],
rate_unemployment = dictionary_small$Variable[15],
net_migration = dictionary_small$Variable[16],
HUD_unit_occupancy_rate = dictionary_small$Variable[17],
number_eviction = dictionary_small$Variable[18],
percentage_excessive_drinking = dictionary_small$Variable[19],
medicare_reimbursements_per_enrollee = dictionary_small$Variable[20],
average_summer_temperature = dictionary_small$Variable[21],
total_annual_precipitation = dictionary_small$Variable[22],
average_Jan_temperature = dictionary_small$Variable[23],
total_Jan_precipitation = dictionary_small$Variable[24],
gini_coefficient_2016 = dictionary_small$Variable[25],
poverty_rate = dictionary_small$Variable[26],
share_renters_2016 = dictionary_small$Variable[27],
share_overcrowded_units_2016 = dictionary_small$Variable[28],
percentage_owners_cost_burden_2016 = dictionary_small$Variable[29],
percentage_renters_severe_cost_burden_2016 = dictionary_small$Variable[30],
share_HUD_units = dictionary_small$Variable[31],
high_housing_density = dictionary_small$Variable[32],
share_built_before_1940  = dictionary_small$Variable[33],
utility_costs  = dictionary_small$Variable[34],
rental_vacancy_rate = dictionary_small$Variable[35],
proportion_one_person_households  = dictionary_small$Variable[36],
share_under_18_with_single_parent = dictionary_small$Variable[37],
share_veteran_status = dictionary_small$Variable[38],
log_median_rent = dictionary_small$Variable[39],
migration_4_year_change  = dictionary_small$Variable[40],
share_no_bachelors = dictionary_small$Variable[41],
city_or_urban = dictionary_small$Variable[42],
suburban = dictionary_small$Variable[43]
)
```

### Identify and deal with missing values

**$\rightarrow$ Create a data frame with data only from 2017 and remove the rows where `total_homeless` is `NA`.**

We only want to look at observations that has `total_homeless` values. Some years have missing data, so we will be using 2017 since it is complete.


```r
df_2017 <- df_hud %>% 
  filter(is.na(total_homeless) == FALSE) %>% 
  filter(year == 2017) 
```

**$\rightarrow$ Use the `mutate` function to create new rate_homeless variables in the data frame `df_2017` that are the counts per 10,000 people in the population.**


```r
df_2017 <- df_2017 %>%
  mutate(
    rate_sheltered = (total_homeless/total_population)*10000,
    rate_unsheltered = (total_sheltered/total_population)*10000,
    rate_homeless = (total_unsheltered/total_population)*10000,
    percent_black = total_black / total_population,
    percent_latino_hispanic = total_latino_hispanic / total_population,
    percent_asian = total_asian / total_population,
    percent_pacific_islander = total_pacific_islander / total_population,
    percent_population_0_19 = total_population_0_19 / total_population,
    percent_population_65_plus = total_population_65_plus / total_population,
    percent_female_population = total_female_population / total_population
  )
```


## Model


**We will multiple approaches to construct models that predict `rate_homeless`:**

>1. Use statistical significance to create a multiple linear regression model.
2. Best subset selection for a multiple linear regression model.
3. Lasso
4. Ridge regression
5. XGBoost

To compare the different approaches, we will use a training and testing split of the data set.

### Set up the data set for training and testing


#### Remove some variables

There are several variables that we do not want to include as predictors. We want to remove demographic totals, the year, the CoC number, and the other homeless rates that we are not predicting. You may have additional variables to remove to create the data set that contains only the response variable and the predictors that you want to use.



```r
variable_remove = c("total_homeless", "total_sheltered", "total_unsheltered", "total_black", "total_latino_hispanic", "total_asian", "total_pacific_islander", "total_population_0_19", "total_population_65_plus", "total_female_population", "year", "coc_number", "total_population","rate_sheltered","rate_unsheltered")

df_model <- df_2017 %>% 
  dplyr::select(-all_of(variable_remove))
```

#### Get train and test splits


**We will split the data into training and testing sets, with 80% of the data kept for training.**


```r
#Do the split. Keep 80% for training. Use stratified sampling based on rate_homeless
split <- initial_split(df_model, prop = 0.8, strata = rate_homeless)

#Extract the training and testing splits
df_train <- training(split)
df_test <- testing(split)
```


### Full regression model


#### Fit the model on the training data

**$\rightarrow$ Use the training data to fit a multiple linear regression model to predict `rate_homeless` using all possible predictor variables.**



```r
fit <- lm(rate_homeless~., df_train)
s <- summary(fit)

s$coefficients[s$coefficients[,4] < 0.05,1] %>% 
  round(2) %>%
  data.frame() 
```

```
##                                      .
## suburban                         -3.92
## rate_unemployment                 1.44
## poverty_rate                     -0.84
## migration_4_year_change           2.43
## proportion_one_person_households  1.34
## total_Jan_precipitation           0.92
```

**$\rightarrow$ Plot the residuals to look for systematic patterns of residuals that might suggest a new model.**


```r
plot(fit,1)
```

![](Clean-Homelessness-Project_files/figure-html/unnamed-chunk-13-1.png)<!-- -->

There is a systematic pattern in the residual plot. 


#### Assess the model on the testing data

**$\rightarrow$ Use the model to predict the homeless rate in the testing data.**


```r
lm_pred <- predict(fit, df_test)
```

**$\rightarrow$ Make a scatter plot to compare the actual value and the predicted value of `rate_homeless`.**


```r
par(pty="s")
plot(df_test$rate_homeless, lm_pred, xlab = "Measured homeless rate", ylab = "Predicted homeless rate", main = "Full linear model", pch = 20, asp =1, ylim = c(0,100), xlim=c(0,100), panel.first = grid(9, lty = 3, lwd = 1))
abline(0,1, col = "red", lty = 2) 
```

![](Clean-Homelessness-Project_files/figure-html/unnamed-chunk-15-1.png)<!-- -->
The model seems to predict homelessness relatively better when there are lower rates of homelessness. For the measured rates of homelessness over 40%, the model underpredicts the homelessness rate.


**$\rightarrow$ Compute the RMSE**


```r
lm_rmse <- sqrt(mean((df_test$rate_homeless- lm_pred)^2))
```

The RMSE is 6.9492825

### Subset selection

*The full model contains too many predictors, so we will use subset selection on the training data to find a smaller number of predictors.*


**$\rightarrow$ Do best subset selection on the training data. Set the maximum number of variables to equal the number of predictor variables in the data set.**


```r
num_var <- ncol(df_train) - 1 # number of variables in the data set that will be used as predictors, this will be used to set the subset size.
regfit_full <- regsubsets(rate_homeless ~ . , data = df_train, nvmax = num_var)
```

**$\rightarrow$ Get the summary. We will use this after performing cross validation to determine the best model.**


```r
reg_summary <- summary(regfit_full)
```


#### Cross validation

**$\rightarrow$ Use 10-fold cross validation on the training data to determine the best subset of predictors in the model.**


```r
n <- nrow(df_train)
k <- 10
set.seed(1)
folds <- sample(1:k,n,replace = TRUE)
cv_errors <- matrix(NA,k,num_var,dimnames = list(NULL,paste(1:num_var)))
for (j in 1:k){
  reg_fit_best <- regsubsets(rate_homeless ~ ., data = df_train[folds !=j,], nvmax = num_var)
    test_mat <- model.matrix(rate_homeless ~ .,data = df_train[folds == j,])
    for (i in 1:num_var){
    coefi <- coef(reg_fit_best,id = i)
    pred <- test_mat[,names(coefi)] %*% coefi
    cv_errors[j,i] <- mean((df_train$rate_homeless[folds == j] - pred)^2)
  }
}

cv_errors_mean <- apply(cv_errors,2,mean)
```


**$\rightarrow$ Plot the assessment measures vs. the number of predictors**


```r
par(mfrow = c(2,2))

ind_cp = which.min(reg_summary$cp)
plot(reg_summary$cp,type = "b",xlab = "Number of variables",ylab = "Cp", main = toString(ind_cp))
points(ind_cp, reg_summary$cp[ind_cp],col = "red",pch = 20)

ind_bic = which.min(reg_summary$bic)
plot(reg_summary$bic,type = "b",xlab = "Number of variables",ylab = "BIC", main = toString(ind_bic))
points(ind_bic, reg_summary$bic[ind_bic],col = "red",pch = 20)

ind_adjr2 = which.max(reg_summary$adjr2)
plot(reg_summary$adjr2,type = "b",xlab = "Number of variables",ylab = "Adjusted R2", main = toString(ind_adjr2))
points(ind_adjr2, reg_summary$adjr2[ind_adjr2],col = "red",pch = 20)

ind_cv <- which.min(cv_errors_mean)
plot(cv_errors_mean,type = "b",xlab = "Number of variables",ylab = "Cross validation", main = toString(ind_cv))
points(ind_cv, cv_errors_mean[ind_cv],col = "red",pch = 20)
```

![](Clean-Homelessness-Project_files/figure-html/unnamed-chunk-20-1.png)<!-- -->

- Adjusted R2 suggests that a model with 23 variables is the best.
- CP suggests that a model with 15 variables is the best.
- Both BIC and Cross Validation suggest that a model with 8 variables is the best.

#### Assess the performance of the model on the testing data

*Use the model to predict the homeless rate in the testing data.*

**$\rightarrow$ Select the variables that are indicated by the best subset selection criterion that you choose (Cp, BIC, Adjusted R-squared, or CV) and produce a new, smaller data frame.**


```r
df_bic <- df_train %>% 
  dplyr::select(all_of(names(coef(regfit_full,ind_bic))[2:length(coef(regfit_full,ind_bic))]), "rate_homeless")

df_adjr2 <- df_train %>% 
  dplyr::select(all_of(names(coef(regfit_full,ind_adjr2))[2:length(coef(regfit_full,ind_adjr2))]), "rate_homeless")

df_cp <- df_train %>% 
  dplyr::select(all_of(names(coef(regfit_full,ind_cp))[2:length(coef(regfit_full,ind_cp))]), "rate_homeless")

df_cv <- df_train %>% 
  dplyr::select(all_of(names(coef(regfit_full,ind_cv))[2:length(coef(regfit_full,ind_cv))]), "rate_homeless")
```

**$\rightarrow$ Fit the model using this subset. The `lm` object will help us to make predictions easily.**


```r
fit_bic <- lm(rate_homeless ~., data = df_bic)
fit_adjr2 <- lm(rate_homeless ~., data = df_adjr2)
fit_cp <- lm(rate_homeless ~., data = df_cp)
fit_cv <- lm(rate_homeless ~., data = df_cv)
```

**$\rightarrow$ Generate predictions of homelessness in the testing data.**


```r
bic_pred <- predict(fit_bic, df_test)
adjr2_pred <- predict(fit_adjr2, df_test)
cp_pred <- predict(fit_cp, df_test)
cv_pred <- predict(fit_cv, df_test)
```

**$\rightarrow$ Compute the RMSE. How does the RMSE compare to the error for the full model?**


```r
bic_rmse <- sqrt(mean((df_test$rate_homeless- bic_pred)^2))
adjr2_rmse <- sqrt(mean((df_test$rate_homeless- adjr2_pred)^2))
cp_rmse <- sqrt(mean((df_test$rate_homeless- cp_pred)^2))
cv_rmse <- sqrt(mean((df_test$rate_homeless- cv_pred)^2))
```

The RMSE for the BIC model is 7.0645304

The RMSE for the adjusted r-squared model is 6.9757962

The RMSE for the cv model is 6.2590016

The RMSE for the cp model is 7.138212


### Lasso

*The lasso is another approach to producing a linear regression model with a subset of the available predictors. The lasso works by finding coefficients $\boldsymbol{\beta}$ that minimize the cost function:*

$$C(\boldsymbol{\beta}) = \sum_{i=1}^n(y_i - \beta_{0} - \sum_{j=1}^p\beta_{j}x_{ij})^2 + \lambda \sum_{j=1}^p|\beta_{j}| = \text{RSS} + \lambda \sum_{j=1}^p|\beta_{j}|$$

*where $\lambda \geq 0$ is a tuning parameter (or hyperparameter).*

**$\rightarrow$ Prepare the data by creating a model matrix `x_train` from the training data. Create the model matrix using `model.matrix` so that it includes the training data for all predictors, but does not include a column for the intercept. Also create the training response data `y_train`.**


```r
x_train <- model.matrix(rate_homeless ~ ., df_train)[,-1] #Remove the intercept
y_train <- df_train$rate_homeless
```


**$\rightarrow$ Use cross-validation to find the best hyperparameter $\lambda$**


```r
#alpha = 1 says that we will use the lasso
cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1)
```


**$\rightarrow$ Show error as a function of the hyperparameter $\lambda$ and its best value.**


```r
plot(cv_lasso)
```

![](Clean-Homelessness-Project_files/figure-html/unnamed-chunk-27-1.png)<!-- -->


```r
best_lam <- cv_lasso$lambda.min
```

The best lambda (hyperparameter) is 0.487267.


**$\rightarrow$ Fit the lasso with the best $\lambda$ using the function `glmnet`.**


```r
lasso_mod <- glmnet(x_train, y_train, alpha = 1, lambda = best_lam)
```



#### Look at prediction error


**$\rightarrow$ Use the model to predict the homeless rate in the testing data.**


```r
x_test <- model.matrix(rate_homeless ~ .,df_test)[,-1] #Remove the intercept
lasso_pred <- predict(lasso_mod, s = best_lam, newx = x_test)
```



**$\rightarrow$ Make a scatter plot to compare the actual value and the predicted value of `rate_homeless`.**


```r
par(pty="s")
plot(df_test$rate_homeless, lasso_pred, xlab = "Measured homeless rate", ylab = "Predicted homeless rate", pch = 20, ylim = c(0,100), xlim=c(0,100), panel.first = grid(9, lty = 3, lwd = 1))
abline(0,1, col = "red", lty = 2)
```

![](Clean-Homelessness-Project_files/figure-html/unnamed-chunk-31-1.png)<!-- -->

The lasso model also seems to underpredict homelessness for areas with a higher actual homelessness rate.

**$\rightarrow$ Compute the RMSE. How does it compare to the other models?**


```r
lasso_rmse <- sqrt(mean((df_test$rate_homeless - lasso_pred)^2))
```

The Lasso RMSE is 6.3247289.

### Ridge regression

*Ridge regression is another approach to model building.* 

*Ridge regression works by finding coefficients $\boldsymbol{\beta}$ that minimize the cost function:*

$$C(\boldsymbol{\beta}) = \sum_{i=1}^n(y_i - \beta_{0} - \sum_{j=1}^p\beta_{j}x_{ij})^2 + \lambda \sum_{j=1}^p\beta_{j}^2 = \text{RSS} + \lambda \sum_{j=1}^p\beta_{j}^2$$

*In contrast to the lasso, ridge regression will not reduce the number of non-zero coefficients in the model. Ridge regression will shrink the coefficients, which helps to prevent overfitting of the training data.*


*The fitting procedure for the ridge regression model mirrors the lasso approach, only changing the parameter $\alpha$ to 0 in the `cv.glmnet` and `glmnet` functions.*


**$\rightarrow$ Fit and assess a ridge regression model. How does it compare to the other models?**


```r
cv_ridge <- cv.glmnet(x_train,y_train,alpha = 0)
best_lam <- cv_ridge$lambda.min
ridge_mod <- glmnet(x_train, y_train, alpha = 0, lambda = best_lam)
ridge_pred <- predict(ridge_mod, s = best_lam, newx = x_test)
ridge_rmse <- sqrt(mean((df_test$rate_homeless - ridge_pred)^2))
```


The rmse is 6.5725872.

### XGBoost

*XGBoost is short for eXtreme Gradient Boosting.*

*We are going to use the `tidymodels` package to fit the XGBoost model.*

#### Set up the model

**The model will be a boosted tree model, so we start by specifying the features of a `boost_tree` model. The`boost_tree` creates a specification of a model, but does not fit the model.**


```r
xgb_spec <- boost_tree(
  mode = "regression",  #We are solving a regression problem
  trees = 1000, 
  tree_depth = tune(),  # tune() says that we will specify this parameter later
  min_n = tune(), 
  loss_reduction = tune(),                     
  sample_size = tune(), 
  mtry = tune(),         
  learn_rate = tune(),                         
  ) %>% 
  set_engine("xgboost", objective = "reg:squarederror") ## We will use xgboost to fit the model and try to minimize the squared error

xgb_spec
```

```
## Boosted Tree Model Specification (regression)
## 
## Main Arguments:
##   mtry = tune()
##   trees = 1000
##   min_n = tune()
##   tree_depth = tune()
##   learn_rate = tune()
##   loss_reduction = tune()
##   sample_size = tune()
## 
## Engine-Specific Arguments:
##   objective = reg:squarederror
## 
## Computational engine: xgboost
```

**Create a workflow that specifies the model formula and the model type. We are still setting up the model; this does not fit the model.**


```r
xgb_wf <- workflow() %>%
  add_formula(rate_homeless ~ .) %>%
  add_model(xgb_spec)

xgb_wf
```

```
## ══ Workflow ════════════════════════════════════════════════════════════════════
## Preprocessor: Formula
## Model: boost_tree()
## 
## ── Preprocessor ────────────────────────────────────────────────────────────────
## rate_homeless ~ .
## 
## ── Model ───────────────────────────────────────────────────────────────────────
## Boosted Tree Model Specification (regression)
## 
## Main Arguments:
##   mtry = tune()
##   trees = 1000
##   min_n = tune()
##   tree_depth = tune()
##   learn_rate = tune()
##   loss_reduction = tune()
##   sample_size = tune()
## 
## Engine-Specific Arguments:
##   objective = reg:squarederror
## 
## Computational engine: xgboost
```


#### Fit the model

**We need to fit all of the parameters that we specified as `tune()`. We will specify the parameter grid using the functions `grid_latin_hypercube`:**


```r
xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), df_train),
  learn_rate(),
  size = 30  #Create 30 sets of the 6 parameters
)
```


**Create folds for cross-validation. **


```r
folds <- vfold_cv(df_train)
```


**Do the parameter fitting. This will take some time.**


```r
xgb_res <- tune_grid(
  xgb_wf,              #The workflow
  resamples = folds,   #The training data split into folds
  grid = xgb_grid,     #The grid of parameters to fit
  control = control_grid(save_pred = TRUE)
)
```

```
## ! Fold01: internal: A correlation computation is required, but `estimate` is const...
```

```
## ! Fold02: internal: A correlation computation is required, but `estimate` is const...
```

```
## ! Fold03: internal: A correlation computation is required, but `estimate` is const...
```

```
## ! Fold04: internal: A correlation computation is required, but `estimate` is const...
```

```
## ! Fold05: internal: A correlation computation is required, but `estimate` is const...
```

```
## ! Fold06: internal: A correlation computation is required, but `estimate` is const...
```

```
## ! Fold07: internal: A correlation computation is required, but `estimate` is const...
```

```
## ! Fold08: internal: A correlation computation is required, but `estimate` is const...
```

```
## ! Fold09: internal: A correlation computation is required, but `estimate` is const...
```

```
## ! Fold10: internal: A correlation computation is required, but `estimate` is const...
```

**Set up the final workflow with the best model parameters.**


```r
#Get the best model, according to RMSE
best_rmse <- select_best(xgb_res, "rmse")

#Update the workflow with the best parameters
final_xgb <- finalize_workflow(
  xgb_wf,
  best_rmse
)
```


#### Prediction


**Fit the final model to the training data and predict the test data.**


```r
final_res <- last_fit(final_xgb, split)
```


**Show the RMSE. Compare the result to the test RMSE for the other models.**


```r
xgb_rmse <- collect_metrics(final_res)$.estimate[1]
```

The RMSE for XGBoost is 6.9243758.

**Plot the homelessness rate and the prediction**


```r
par(pty="s")
plot(final_res$.predictions[[1]]$rate_homeless,final_res$.predictions[[1]]$.pred, pch = 20, title = "XGBoost", xlab = "Homeless rate", ylab = "Predicted homeless rate", ylim = c(0,100), xlim=c(0,100), panel.first = grid(9, lty = 3, lwd = 1))
```

```
## Warning in plot.window(...): "title" is not a graphical parameter
```

```
## Warning in plot.xy(xy, type, ...): "title" is not a graphical parameter
```

```
## Warning in axis(side = side, at = at, labels = labels, ...): "title" is not a
## graphical parameter

## Warning in axis(side = side, at = at, labels = labels, ...): "title" is not a
## graphical parameter
```

```
## Warning in box(...): "title" is not a graphical parameter
```

```
## Warning in title(...): "title" is not a graphical parameter
```

```r
abline(0,1,col = "red", lty = 3)
```

![](Clean-Homelessness-Project_files/figure-html/unnamed-chunk-42-1.png)<!-- -->

XGBoost seems to underpredict homelessness rates in areas with actual homelessness rates over 30%.

#### Relative importance of predictors

**Look at which predictors are most important in the model**


```r
final_xgb %>%
  fit(data = df_train) %>%
  pull_workflow_fit() %>%
  vip(geom = "col")
```

![](Clean-Homelessness-Project_files/figure-html/unnamed-chunk-43-1.png)<!-- -->

### Compare models

*You used several methods to construct a model*

>1. Use statistical significance to create a multiple linear regression model.
2. Best subset selection for a multiple linear regression model.
3. Lasso
4. Ridge regression
5. XGBoost

*Compare the performance of the models. *

|  | Model | RMSE|
|:-- |:-- |:-- |
|1. | Multiple Linear Regression (Statistical Significance) | 6.9492825|
|2. | Multiple Linear Regression (Best Subset Selection, BIC) | 7.0645304|
|2. | Multiple Linear Regression (Best Subset Selection, Adjusted r-squared) | 6.9757962|
|2. | Multiple Linear Regression (Best Subset Selection, CP) | 7.138212|
|2. | Multiple Linear Regression (Best Subset Selection, CV) | 6.2590016|
|3. | Lasso | 6.3247289|
|4. | Ridge Regression | 6.5725872|
|5. | XGBoost | 6.9243758 |

## Conclusion

In this analysis, we used several methods of modeling predictions of homelessness: multiple linear regression, best subset selection(BIC, adjusted r-squared, cv, and cp), lasso, ridge regression, and XGBoost. 

To compare these models, we looked at the RMSE. The model with the lowest RMSE is the standard multiple linear regression model. However this may not be the best model since it does not take overfitting into account. 

Just because we've produced models doesn't mean that we have an answer to our original problem of predicting homelessness. Developing these models and examining the data has given us more insight into which factors have a greater effect on homelessness, but we cannot say that we are able to predict homelessness rates based off of a few socio-economic factors.
