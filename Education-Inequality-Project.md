---
title: "Education inequality"
output: 
  html_document: 
    keep_md: true
---


# Introduction

This project addresses education inequality in U.S. high schools. The quality of a high school education can be measured in multiple ways, but here we will focus on average student performance on the ACT or SAT exams that students take as part of the college application process.

We expect a range of school performance on these exams, but is school performance predicted by socioeconomic factors? 

$\rightarrow$ Complete the introduction with a description of the problem you are addressing. The introduction should include at least two paragraphs that describe (1) the general problem you are addressing and (2) the specific questions you are asking in this project.

It is known that wealthier areas are known for having "better" schools. In this project, we are looking at an array of socioeconomic factors that could potentially be criterea for predicting quality of high school education in terms of ACT and SAT scores. There is no proof that these tests are indication of the quality of the schools, but it is a standardized and widely used assessment, which allows us to better analyze individual high school's performances and socioeconomic factors.

We will be looking which factors may be able to predict a school's test scores. Before making any conclusions, we will make sure to clean or prepare the data for analysis by removing any outliers and null values, or any other values that could skew our data. Then, we will take a look at all of our factors to verify the factors we would like to examine further. After this, we will fit models and pick the best predictor(s) of the average ACT/SAT score for a school.

# Data Collection

This project utilizes two data sets. The primary data set is the EdGap data set from [EdGap.org](https://www.edgap.org/#5/37.875/-96.987). This data set from 2016 includes information about average ACT or SAT scores for schools and several socioeconomic characteristics of the school district. The secondary data set is basic information about each school from the [National Center for Education Statistics](https://nces.ed.gov/ccd/pubschuniv.asp).

## EdGap data

All socioeconomic data (household income, unemployment, adult educational attainment, and family structure) is from the Census Bureau's American Community Survey. 

[EdGap.org](https://www.edgap.org/#5/37.875/-96.987) report that ACT and SAT score data is from each state's department of education or some other public data release. The nature of the other public data release is not known.

The quality of the census data and the department of education data can be assumed to be reasonably high. 

[EdGap.org](https://www.edgap.org/#5/37.875/-96.987) do not indicate that they processed the data in any way. The data were assembled by the [EdGap.org](https://www.edgap.org/#5/37.875/-96.987) team, so there is always the possibility for human error. Given the public nature of the data, we would be able to consult the original data sources to check the quality of the data if we had any questions.


## School information data

The school information data is from the [National Center for Education Statistics](https://nces.ed.gov/ccd/pubschuniv.asp). This data set consists of basic identifying information about schools can be assumed to be reasonably high. As for the EdGap.org data, the school information data is public, so we would be able to consult the original data sources to check the quality of the data if we had any questions.


# Data Preparation

## Load necessary packages


```r
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
#skimr provides a nice summary of a data set
library(skimr)
#leaps will be used for model selection
library(leaps)
#kableExtra will be used to make tables in the html document
library(kableExtra)
```

```
## 
## Attaching package: 'kableExtra'
```

```
## The following object is masked from 'package:dplyr':
## 
##     group_rows
```

```r
#latex2exp lets us use LaTex in ggplot
library(latex2exp)
library(dplyr)
```

## Loading the data 

### Load the EdGap set

$\rightarrow$ Load the data set contained in the files `EdGap_data.xlsx` and `ccd_sch_029_1617_w_1a_11212017.csv`,  name the data frames `edgap` and `school_info`.


```r
edgap <- read_xlsx("EdGap_data.xlsx")
school_info <- read_csv("ccd_sch_029_1617_w_1a_11212017.csv")
```

```
## 
## ── Column specification ────────────────────────────────────────────────────────
## cols(
##   .default = col_character(),
##   UNION = col_logical(),
##   MSTREET2 = col_logical(),
##   MSTREET3 = col_logical(),
##   MZIP = col_double(),
##   MZIP4 = col_logical(),
##   LSTREET2 = col_logical(),
##   LSTREET3 = col_logical(),
##   LZIP = col_double(),
##   LZIP4 = col_logical(),
##   SY_STATUS = col_double(),
##   UPDATED_STATUS = col_double(),
##   SCH_TYPE = col_double(),
##   CHARTAUTH1 = col_logical(),
##   CHARTAUTHN1 = col_logical(),
##   CHARTAUTH2 = col_logical(),
##   CHARTAUTHN2 = col_logical()
## )
## ℹ Use `spec()` for the full column specifications.
```

```
## Warning: 138139 parsing failures.
##  row   col           expected actual                                 file
## 1549 MZIP4 1/0/T/F/TRUE/FALSE   8999 'ccd_sch_029_1617_w_1a_11212017.csv'
## 1550 MZIP4 1/0/T/F/TRUE/FALSE   0800 'ccd_sch_029_1617_w_1a_11212017.csv'
## 1551 MZIP4 1/0/T/F/TRUE/FALSE   0700 'ccd_sch_029_1617_w_1a_11212017.csv'
## 1552 MZIP4 1/0/T/F/TRUE/FALSE   0050 'ccd_sch_029_1617_w_1a_11212017.csv'
## 1553 MZIP4 1/0/T/F/TRUE/FALSE   0019 'ccd_sch_029_1617_w_1a_11212017.csv'
## .... ..... .................. ...... ....................................
## See problems(...) for more details.
```

### Select a subset of columns

$\rightarrow$ Use the `select` function to select these columns from the data frame. 


```r
school_info <- school_info %>% 
  select(NCESSCH, MSTATE, MZIP, SCH_TYPE_TEXT, LEVEL)
```

Select columns we want to keep from `school_info` so that the data frame is smaller and easier to work with.

### Data cleaning

#### Rename variables

In any analysis, we might rename the variables in the data frame to make them easier to work with. We have seen that the variable names in the `edgap` data frame allow us to understand them, but they can be improved. In contrast, many of the variables in the `school_info` data frame have confusing names.

#### EdGap data

$\rightarrow$ View the column names for the EdGap data


```r
names(edgap)
```

```
## [1] "NCESSCH School ID"                              
## [2] "CT Unemployment Rate"                           
## [3] "CT Pct Adults with College Degree"              
## [4] "CT Pct Childre In Married Couple Family"        
## [5] "CT Median Household Income"                     
## [6] "School ACT average (or equivalent if SAT score)"
## [7] "School Pct Free and Reduced Lunch"
```

We will use the `rename` function from the `dplyr` package to rename the columns.


```r
#The new name for the column is on the left of the =

edgap <- edgap %>% 
  rename(id = "NCESSCH School ID",
         rate_unemployment = "CT Unemployment Rate",
         percent_college = "CT Pct Adults with College Degree",
         percent_married = "CT Pct Childre In Married Couple Family",
         median_income = "CT Median Household Income",
         average_act = "School ACT average (or equivalent if SAT score)",
         percent_lunch = "School Pct Free and Reduced Lunch"
        )
```

Renames all variables so that they have better names that make sense. 

### School information data

$\rightarrow$ View the column names for the school information data


```r
names(school_info)
```

```
## [1] "NCESSCH"       "MSTATE"        "MZIP"          "SCH_TYPE_TEXT"
## [5] "LEVEL"
```

The names can be improved for readability. We also have the constraint that we rename `ncessch` to `id` to be consistent with the `edgap` data.

Rename the columns of the school information data frame.


```r
school_info <- school_info %>% 
  rename(id = "NCESSCH",
         state = "MSTATE",
         zip_code = "MZIP",
         school_type = "SCH_TYPE_TEXT",
         school_level = "LEVEL"
         )

#Print the names to see the change
names(school_info)
```

```
## [1] "id"           "state"        "zip_code"     "school_type"  "school_level"
```

Renames all variables so that they follow proper naming principles and NCESSCH id is named id so that both columns have the same names in both data frames.

### Joining the data frames

We will join the `edgap` and `school_info` data frames based on the school ID. We should first note that the `id` is coded differently in the two data frames:


```r
typeof(edgap$id)
```

```
## [1] "double"
```

```r
typeof(school_info$id)
```

```
## [1] "character"
```

While `id` is a number, it is a categorical variable and should be represented as a character variable in R. 


Convert `id` in `edgap` id to a character variable:

We will use the `mutate` function from the `dplyr` package to rename the columns.

We will now also join the data frames. We want to perform a left join based on the school ID `id` so that we incorporate all of the school information into the `edgap` data frame.


```r
edgap <- edgap %>% 
  mutate(id = as.character(id))  %>% 
  left_join(school_info, by = "id") 
```

### Identify and deal with missing values

$\rightarrow$ How many missing values are there in each column? Give the number of missing values and the percent of values in each column that are missing.

number of missing values:

```r
edgap %>% 
  map_df(~sum(is.na(.)))
```

```
## # A tibble: 1 x 11
##      id rate_unemployment percent_college percent_married median_income
##   <int>             <int>           <int>           <int>         <int>
## 1     0                14              13              25            20
## # … with 6 more variables: average_act <int>, percent_lunch <int>, state <int>,
## #   zip_code <int>, school_type <int>, school_level <int>
```

percentage of missing values:

```r
edgap %>% 
  map_df(~sum(is.na(.)))/nrow(edgap)*100 
```

```
##   id rate_unemployment percent_college percent_married median_income
## 1  0         0.1753068       0.1627849       0.3130478     0.2504383
##   average_act percent_lunch    state zip_code school_type school_level
## 1           0             0 1.101928 1.101928    1.101928     1.101928
```


Recall that missing values are coded in R with `NA`, or they may be empty. We want to convert empty cells to `NA`, so that we can use the function `is.na` to find all missing values. The function `read_excel` that we used to read in the data does this automatically, so we do not need to take further action to deal with empty cells.


$\rightarrow$ Find the rows where the missing value occurs in each column.


```r
apply(is.na(edgap),2,which)
```

```
## $id
## integer(0)
## 
## $rate_unemployment
##  [1]  142  143  205  364 1056 1074 2067 2176 2717 2758 2931 3973 4150 6103
## 
## $percent_college
##  [1]  142  205  364 1056 1074 2067 2176 2717 2758 2931 3973 4150 6103
## 
## $percent_married
##  [1]    7  142  143  205  364  465 1056 1074 2067 2176 2313 2419 2593 2717 2758
## [16] 2774 2931 3905 3973 4150 4772 5879 6103 6626 7248
## 
## $median_income
##  [1]  142  143  205  364  465 1056 1074 2067 2176 2419 2593 2717 2758 2931 3973
## [16] 4150 5879 6103 6626 7248
## 
## $average_act
## integer(0)
## 
## $percent_lunch
## integer(0)
## 
## $state
##  [1]  482  483  485  486  487  488  490  507 1088 1428 1444 1458 1468 1575 1649
## [16] 2028 2029 2042 2046 2047 2105 2107 2575 2580 2589 2592 2617 2635 2688 2767
## [31] 2771 2772 2773 2774 2775 2776 2777 2778 2779 2780 2781 2782 2783 2784 2785
## [46] 2786 2787 2788 2932 3001 3127 3487 3544 3830 4397 4660 4662 4665 4737 4766
## [61] 4776 5096 5421 5453 5454 5496 5568 5764 6006 6007 6025 6027 6029 6032 6034
## [76] 6043 6044 6350 6354 6356 6357 6359 6379 6834 6901 7564 7568 7961
## 
## $zip_code
##  [1]  482  483  485  486  487  488  490  507 1088 1428 1444 1458 1468 1575 1649
## [16] 2028 2029 2042 2046 2047 2105 2107 2575 2580 2589 2592 2617 2635 2688 2767
## [31] 2771 2772 2773 2774 2775 2776 2777 2778 2779 2780 2781 2782 2783 2784 2785
## [46] 2786 2787 2788 2932 3001 3127 3487 3544 3830 4397 4660 4662 4665 4737 4766
## [61] 4776 5096 5421 5453 5454 5496 5568 5764 6006 6007 6025 6027 6029 6032 6034
## [76] 6043 6044 6350 6354 6356 6357 6359 6379 6834 6901 7564 7568 7961
## 
## $school_type
##  [1]  482  483  485  486  487  488  490  507 1088 1428 1444 1458 1468 1575 1649
## [16] 2028 2029 2042 2046 2047 2105 2107 2575 2580 2589 2592 2617 2635 2688 2767
## [31] 2771 2772 2773 2774 2775 2776 2777 2778 2779 2780 2781 2782 2783 2784 2785
## [46] 2786 2787 2788 2932 3001 3127 3487 3544 3830 4397 4660 4662 4665 4737 4766
## [61] 4776 5096 5421 5453 5454 5496 5568 5764 6006 6007 6025 6027 6029 6032 6034
## [76] 6043 6044 6350 6354 6356 6357 6359 6379 6834 6901 7564 7568 7961
## 
## $school_level
##  [1]  482  483  485  486  487  488  490  507 1088 1428 1444 1458 1468 1575 1649
## [16] 2028 2029 2042 2046 2047 2105 2107 2575 2580 2589 2592 2617 2635 2688 2767
## [31] 2771 2772 2773 2774 2775 2776 2777 2778 2779 2780 2781 2782 2783 2784 2785
## [46] 2786 2787 2788 2932 3001 3127 3487 3544 3830 4397 4660 4662 4665 4737 4766
## [61] 4776 5096 5421 5453 5454 5496 5568 5764 6006 6007 6025 6027 6029 6032 6034
## [76] 6043 6044 6350 6354 6356 6357 6359 6379 6834 6901 7564 7568 7961
```

There are some schools that are missing all four of the socioeconomic variables, e.g. at rows 142 and 205. However, many of the schools are missing only a subset of the variables. If we drop rows that have `NA`s, then we will negatively affect our analysis using the variables where data were present. So, we will not drop the rows in this data set that are missing the socioeconomic variables. We have so few missing values from each value that we will not worry about replacing `NA`s with some other value. We will selectively omit the `NA`s when working with those columns.

There are, however, 88 schools where we do not have the school information. This raises the possibility that the information is unreliable. Because we are not able to check from the source, we will omit these rows from the data set.

$\rightarrow$ Use the `filter` function to drop only those rows where the state information is missing.


```r
edgap <- edgap %>% 
  filter(is.na(state) == FALSE) 
```


### Are there data points that look like errors?

We will do some quality control for the data set.

We can check the range of each variable to see that the values fall in the expected ranges.


```r
summary(edgap)
```

```
##       id            rate_unemployment percent_college   percent_married 
##  Length:7898        Min.   :0.00000   Min.   :0.09149   Min.   :0.0000  
##  Class :character   1st Qu.:0.05854   1st Qu.:0.45120   1st Qu.:0.5261  
##  Mode  :character   Median :0.08523   Median :0.55572   Median :0.6686  
##                     Mean   :0.09806   Mean   :0.56938   Mean   :0.6354  
##                     3rd Qu.:0.12272   3rd Qu.:0.67719   3rd Qu.:0.7778  
##                     Max.   :0.59028   Max.   :1.00000   Max.   :1.0000  
##                     NA's   :14        NA's   :13        NA's   :24      
##  median_income     average_act     percent_lunch         state          
##  Min.   :  3589   Min.   :-3.071   Min.   :-0.05455   Length:7898       
##  1st Qu.: 36787   1st Qu.:18.639   1st Qu.: 0.23772   Class :character  
##  Median : 47000   Median :20.400   Median : 0.37904   Mode  :character  
##  Mean   : 52167   Mean   :20.211   Mean   : 0.41824                     
##  3rd Qu.: 61517   3rd Qu.:21.935   3rd Qu.: 0.57060                     
##  Max.   :226181   Max.   :32.363   Max.   : 0.99873                     
##  NA's   :20                                                             
##     zip_code     school_type        school_level      
##  Min.   : 1001   Length:7898        Length:7898       
##  1st Qu.:28429   Class :character   Class :character  
##  Median :45243   Mode  :character   Mode  :character  
##  Mean   :44851                                        
##  3rd Qu.:62350                                        
##  Max.   :99403                                        
## 
```

There are a few suspicious values. The minimum `average_act` is -3.071, but ACT scores must be non-negative. Similarly, the minimum `percent_lunch` is -0.05455, but a percent must be non-negative. We do not have access to information about how these particular data points were generated, so we will remove them from the data set by converting them to `NA`.


```r
#Number of NA ACT scores before conversion
sum(is.na(edgap$average_act))
```

```
## [1] 0
```

```r
#Convert negative scores to NA
edgap[edgap$average_act < 0,"average_act"] = NA

#Number of NA ACT scores after conversion
sum(is.na(edgap$average_act))
```

```
## [1] 3
```

There were 3 schools with negative ACT scores, where the values were omitted from the data set.


```r
#Number of NA percent_lunch values before conversion
sum(is.na(edgap$percent_lunch))
```

```
## [1] 0
```

```r
#Convert negative values to NA
edgap[edgap$percent_lunch < 0,"percent_lunch"] = NA

#Number of NA percent_lunch values after conversion
sum(is.na(edgap$percent_lunch))
```

```
## [1] 20
```

There were 20 schools with negative percent free or reduced lunch, where the values were omitted from the data set.

# Model

## Model selection

We do not have many input variables, so we can examine the full model.


```r
fit_full <- lm(average_act ~ percent_lunch + median_income + rate_unemployment + percent_college + percent_married, data = edgap)
```

Examine the summary of the fit


```r
summary(fit_full)
```

```
## 
## Call:
## lm(formula = average_act ~ percent_lunch + median_income + rate_unemployment + 
##     percent_college + percent_married, data = edgap)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -8.5200 -0.9542 -0.0908  0.8780 10.8438 
## 
## Coefficients:
##                     Estimate Std. Error t value Pr(>|t|)    
## (Intercept)        2.272e+01  1.321e-01 172.003  < 2e-16 ***
## percent_lunch     -7.590e+00  9.230e-02 -82.230  < 2e-16 ***
## median_income     -3.392e-07  1.185e-06  -0.286    0.775    
## rate_unemployment -2.200e+00  3.870e-01  -5.685 1.35e-08 ***
## percent_college    1.669e+00  1.520e-01  10.986  < 2e-16 ***
## percent_married   -6.412e-02  1.273e-01  -0.504    0.615    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 1.551 on 7845 degrees of freedom
##   (47 observations deleted due to missingness)
## Multiple R-squared:  0.6242,	Adjusted R-squared:  0.6239 
## F-statistic:  2606 on 5 and 7845 DF,  p-value: < 2.2e-16
```

The coefficients for `median_income` and `percent_married` are not statistically significant. Additionally, the sign of the coefficients for `median_income` and `percent_married` do not make sense. These results support removing `median_income` and `percent_married` from the model. 


#### Do best subset selection

Use the `regsubsets` function from the `leaps` package to perform best subset selection in order to choose the best model to predict `average_act` from the socioeconomic predictors. 


```r
#perform best subset selection
regfit_full <- regsubsets(average_act ~ percent_lunch + median_income + rate_unemployment + percent_college + percent_married, data = edgap)
```

Get the summary of the best subset selection analysis


```r
reg_summary <- summary(regfit_full)
```


What is the best model obtained according to Cp, BIC, and adjusted $R^2$? Make a plot of Cp, BIC, and adjusted $R^2$ vs. the number of variables in the model.


```r
#Set up a three panel plot
par(mfrow = c(1,3))

#Plot Cp
plot(reg_summary$cp,type = "b",xlab = "Number of variables",ylab = "Cp")
#Identify the minimum Cp
ind_cp = which.min(reg_summary$cp)
points(ind_cp, reg_summary$cp[ind_cp],col = "red",pch = 20)

#Plot BIC
plot(reg_summary$bic,type = "b",xlab = "Number of variables",ylab = "BIC")
#Identify the minimum BIC
ind_bic = which.min(reg_summary$bic)
points(ind_bic, reg_summary$bic[ind_bic],col = "red",pch = 20)

#Plot adjusted R^2
plot(reg_summary$adjr2,type = "b",xlab = "Number of variables",ylab = TeX('Adjusted $R^2$'),ylim = c(0,1))

#Identify the maximum adjusted R^2
ind_adjr2 = which.max(reg_summary$adjr2)
points(ind_adjr2, reg_summary$adjr2[ind_adjr2],col = "red",pch = 20)
```

![](Clean-Education-Inequality-Project_files/figure-html/unnamed-chunk-21-1.png)<!-- -->

The three measures agree that the best model has three variables.


Show the best model for each possible number of variables. Focus on the three variable model.


```r
reg_summary$outmat
```

```
##          percent_lunch median_income rate_unemployment percent_college
## 1  ( 1 ) "*"           " "           " "               " "            
## 2  ( 1 ) "*"           " "           " "               "*"            
## 3  ( 1 ) "*"           " "           "*"               "*"            
## 4  ( 1 ) "*"           " "           "*"               "*"            
## 5  ( 1 ) "*"           "*"           "*"               "*"            
##          percent_married
## 1  ( 1 ) " "            
## 2  ( 1 ) " "            
## 3  ( 1 ) " "            
## 4  ( 1 ) "*"            
## 5  ( 1 ) "*"
```

The best model uses the predictors `percent_lunch`, `rate_unemployment` and `percent_college`.

Fit the best model and examine the results


```r
fit_best <- lm(average_act ~ percent_lunch + rate_unemployment + percent_college, data = edgap)
```


```r
summary(fit_best)
```

```
## 
## Call:
## lm(formula = average_act ~ percent_lunch + rate_unemployment + 
##     percent_college, data = edgap)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -8.4792 -0.9550 -0.0916  0.8758 10.8014 
## 
## Coefficients:
##                   Estimate Std. Error t value Pr(>|t|)    
## (Intercept)       22.64338    0.09800 231.057  < 2e-16 ***
## percent_lunch     -7.58151    0.08801 -86.144  < 2e-16 ***
## rate_unemployment -1.91829    0.35214  -5.447 5.26e-08 ***
## percent_college    1.65519    0.12187  13.581  < 2e-16 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 1.552 on 7857 degrees of freedom
##   (37 observations deleted due to missingness)
## Multiple R-squared:  0.6235,	Adjusted R-squared:  0.6234 
## F-statistic:  4338 on 3 and 7857 DF,  p-value: < 2.2e-16
```


#### Relative importance of predictors

To compare the magnitude of the coefficients, we should first normalize the predictors. Each of the predictors `percent_lunch`, `rate_unemployment` and `percent_college` is limited to the interval (0,1), but they occupy different parts of the interval. We can normalize each variable through a z-score transformation:


```r
scale_z <- function(x, na.rm = TRUE) (x - mean(x, na.rm = na.rm)) / sd(x, na.rm)

edgap_z <- edgap %>% 
  mutate_at(c("percent_lunch","rate_unemployment","percent_college"),scale_z) 
```

$\rightarrow$ Fit the model using the transformed variables and examine the coefficients


```r
fit_full <- lm(average_act ~ percent_lunch + median_income + rate_unemployment + percent_college + percent_married, data = edgap)
```


```r
df <- subset(edgap, select = c("rate_unemployment", "percent_married", "median_income", "average_act", "percent_lunch", "percent_college"))
df <- na.omit(df)
n <- nrow(df)
num_var <- ncol(df) - 1

# folds
k <- 10
set.seed(1)
folds <- sample(1:k,n,replace = TRUE)

cv_errors <- matrix(NA,k,num_var,dimnames = list(NULL,paste(1:num_var)))


for (j in 1:k){
  reg_fit_best <- regsubsets(average_act ~ ., data = df[folds !=j,])
  test_mat <- model.matrix(average_act ~ .,data = df[folds == j,])
    for (i in 1:num_var){
    coefi <- coef(reg_fit_best,id = i)
    pred <- test_mat[,names(coefi)] %*% coefi
    cv_errors[j,i] <- mean((df$average_act[folds == j] - pred)^2)
  }
}

cv_errors_mean <- apply(cv_errors,2,mean)
ind_cv <- which.min(cv_errors_mean)

# best subset selection on data set using the optimal number of variables
reg_best <- regsubsets(average_act ~ ., data = df)

coef(reg_best,ind_cv)
```

```
##       (Intercept) rate_unemployment     percent_lunch   percent_college 
##         22.669613         -2.094955         -7.569927          1.629588
```

# Conclusion

From this analysis, we can see that there is in fact a correlation between economic contributors and average ACT scores. Now, this doesn't fully address the initial question of overall education quality. Predicting standardized test scores based on economic factors makes sense as there are many resources available to the wealth for test preparation which can increase scores. Wealthy families and parents who also went to college are more likely to encourage their children to do well so that they can go to college. If a student does not want to go to college, they likely won't put in the effort to do well on the test since there is no point. In addition, poorer families usually can't afford to send their children to college and/or need them to start working. 

I also believe that we can't just judge quality of education on solely one factor. Standardized tests suit some students' strengths more than others, so I don't think we should consider the scores in isolation. However, there are not many other data that is consistent throughout the country in all high schools. This is a very difficult question to answer statistically as I think there is more to the quality of education other than numbers.
