/****************************************************************/
* DESCRIPTION: Macros for logistic regression in SAS demonstrated on a public dataset
*
* AUTHOR: 
* CREATION DATE: 2023-05-18
* LAST CHANGE DATE: 2023-05-18 
* REVIEWWER:
* REVIEW DATE:
* 
* INPUT: For a binary dependent variable, the "Adult" dataset from the 
*	UCI Machine Learning Repository is used. This dataset contains a 
*	binary income variable (more than $50K per year or less) as well as 
*	various independent variables such as age, workclass, education, marital status, etc.
*
* Download link: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
*
* More information: https://archive.ics.uci.edu/ml/datasets/Adult
* This is not a CSV file by default, but has been converted into a CSV by adding headers 
* and changing the delimiter from a space to a semicolon.
*
* SUMMARY : The SAS code provided describes a project for creating a logistic regression model 
* to predict whether a person's income is more than $50,000 a year. The script is divided into 
* several sections: data import and preparation, model development, model scoring, and model performance testing.
*
* Please find a detailed summary at the end of the script.
/**** DETAILED SUMMARY ****

**Data Import and Preparation**
The dataset used is the publicly available "Adult" dataset from the UCI Machine Learning Repository. The script imports this dataset and prepares it by creating a binary dependent variable, high_income.

**Model Development**
The dataset is stratified based on the high_income variable. A macro, `stratify_data`, is used to randomly split the data into a training set (70%) and a test set (30%). The logistic regression model is built on the training dataset with the dependent variable being high_income and independent variables including age, workclass, education, marital status, and more. Automatic stepwise selection is performed to choose the best subset of predictors. The model estimates are stored for future use.

**Model Scoring**
The model is then scored on both the training and test datasets. The scored datasets contain the predicted probability of high_income, which is renamed to PP. The observations are then grouped into four 'Grades' based on the quantiles of PP.

**Model Performance Testing**
Several statistical techniques are used to test the performance of the model on both the training and test data:

1. Descriptive Analysis: A summary of the data is provided including the mean, median, min, max, and standard deviation.

2. Discriminatory Power: The ability of the model to differentiate between different outcomes is assessed by calculating the Gini coefficient and creating a confusion matrix. The Gini coefficient is derived from the area under the ROC curve and can range from 0 to 1, with a higher value indicating a better model. The F-score, which is a measure of test accuracy, is also calculated.

3. Accuracy: The accuracy of the model is evaluated using a paired t-test comparing high_income and PP. This is done overall and also separately for each 'Grade'. A binomial test is also performed to compare the number of "successes" (high_income = 1) to the average predicted probability (PP).

The final logistic regression model achieved a Gini coefficient of around 80%, which is considered good as a higher Gini coefficient indicates better model performance.

/****************************************************************;

/****************************************************************/
**** Data import and prep ****;
/****************************************************************/
libname datlib "D:\E\Wissensbasis\Projekte\SAS_Macro_collection\SAS_Macro_Regression_examples";

/* Macro: %import_csv
Parameters:
    csv_path: is the full path to your .csv file.
    out_dataset: is the name of the output dataset you want to create. It should be in the format libname.dataset.
    
Options:	
	DELIMITER: specifies the character that separates columns in your .csv file.
    GETNAMES: tells SAS to use the first row of the .csv file as variable names.
    DATAROW: specifies the row number where the actual data starts.
*/
%macro import_csv(csv_path, out_dataset);
    PROC IMPORT 
        DATAFILE= &csv_path
        OUT= &out_dataset 
        DBMS=CSV 
        REPLACE;
        DELIMITER=';';
        GETNAMES=YES;
        DATAROW=2; 
    RUN;
%mend;

%import_csv("D:\E\Wissensbasis\Projekte\SAS_Macro_collection\SAS_Macro_Regression_examples\adult.csv", datlib.adult);


/* Create dependent binary variable high_income, being either 0 or 1 */
DATA datlib.adult; 
    SET datlib.adult;
    IF income = ">50K" THEN high_income = 1;
    ELSE high_income = 0;
RUN;


/****************************************************************/
**** Model development ****;
/****************************************************************/

/* Sort the dataset by the stratification variable */
PROC SORT DATA=datlib.adult;
    BY high_income;
RUN;

/* Macro: stratify_data
Parameters:
    - data: full path to your original dataset
    - out_split: name of the output dataset to create containing new column "selected" (0 or 1)
    - test_percent: proportion of the data to use for testing (decimal between 0 and 1)
    - seed: seed for the random number generator for reproducibility
*/
%macro stratify_data(data, out_split, test_percent, seed);
    /* PROC SURVEYSELECT is used to create a stratified sample of the data
    The STRATA statement specifies the variable by which to stratify the dataset (high_income) */
    PROC SURVEYSELECT DATA=&data
                      OUTALL
                      OUT=&out_split 
                      METHOD=SRS
                      SAMPRATE=&test_percent 
                      SEED=&seed;
                      
        STRATA high_income;
    RUN;
%mend;

%stratify_data(datlib.adult, datlib.out_split, 0.7, 12345);


/* Macro: %logit_model_k_best
Parameters:
    - data: name of the dataset to use (format: libname.dataset)
    - out_est: name of the output dataset for the parameter estimates (format: libname.dataset)
    - out_data: name of the output dataset with predicted probabilities (format: libname.dataset)
*/
%macro logit_model_stepwise(data, out_est, out_data);
    /* Subset the data for model training */
    DATA train;
        SET &data;
        IF selected = 1;
    RUN;

    /* Fit the logistic regression model with automatic feature selection */
    /* OUTEST option provides the parameter estimates */
    PROC LOGISTIC DATA=train DESCENDING OUTEST=&out_est OUTMODEL=datlib.model;
	    CLASS workclass education marital_status occupation;
        MODEL high_income = age workclass fnlwgt education marital_status occupation 
					        capital_gain capital_loss hours_per_week / SELECTION=STEPWISE;
        output out=&out_data p=phat lower=lcl upper=ucl predprobs=(individual crossvalidate);
    RUN;

    /* Score the full dataset to get predicted probabilities 
    PROC LOGISTIC INMODEL=model;
        SCORE DATA=&data OUT=&out_data PREDPROBS;
    RUN;
    */
%mend;

/* Use the macro to fit a logistic regression model with automatic feature selection */
%logit_model_stepwise(datlib.out_split, datlib.logit_est, datlib.out_score);


**** Score model on test dataset ****;
DATA datlib.test;
    SET datlib.out_split;
    IF selected = 0;
RUN;

PROC LOGISTIC INMODEL=datlib.model;
    SCORE DATA=datlib.test OUT=datlib.out_score_test;
RUN;

**** rename the predicted probability PP ****;
DATA datlib.out_score;
	SET datlib.out_score;
	PP = phat;
RUN;

DATA datlib.out_score_test;
	SET datlib.out_score_test;
	PP = P_1;
RUN;


**** Create Grades based on quantiles of PP ****;
PROC UNIVARIATE DATA=datlib.out_score noprint;
   VAR PP;
   OUTPUT OUT=PP_Quantiles pctlpts=25 50 75 pctlpre=PP_;
RUN;

/* Adding 'Grade' to training data */
DATA datlib.out_score;
   SET datlib.out_score;
   IF PP <= 0.0196523716 THEN Grade = '1';
   ELSE IF PP <= 0.1016713671 THEN Grade = '2';
   ELSE IF PP <= 0.3869616567 THEN Grade = '3';
   ELSE Grade = '4';
RUN;

/* Adding 'Grade' to test data */
DATA datlib.out_score_test;
   SET datlib.out_score_test;
   IF PP <= 0.0196523716 THEN Grade = '1';
   ELSE IF PP <= 0.1016713671 THEN Grade = '2';
   ELSE IF PP <= 0.3869616567 THEN Grade = '3';
   ELSE Grade = '4';
RUN;


/****************************************************************/
**** Model performance testing ****;
/****************************************************************/

**** Model Discriminatory power ****;

/* Macro 1: Descriptive analysis */
%macro DescriptiveAnalysis(data=, out=);
    /* Use PROC MEANS to get descriptive statistics */
    proc means data=&data mean median min max stddev;
        var PP high_income;
        output out=&out;
    run;
%mend;

/* Call macros for both training and testing data */
%DescriptiveAnalysis(data=datlib.out_score, out=datlib.descriptive_analysis_train);
%DescriptiveAnalysis(data=datlib.out_score_test, out=datlib.descriptive_analysis_test);


/* Macro 2: Discriminatory power - Gini */
%macro Gini(data=, out=, plot=0);
    /* Use PROC LOGISTIC to get ROC and other stats */
    proc logistic data=&data        
		    /* Optionally plot ROC */
	        %if &plot = 1 %then %do;
	            plot = roc
	        %end;;
        model high_income(event='1')= PP / outroc=roc_data;
        roc 'roc' PP;
    run;

    /* Calculate Gini coefficient from ROC data */
    data roc_gini;
        set roc_data end=eof;
        if _n_=1 then do;
            retain _s 0;
            _a=_1mspec_ * _sensit_;
        end;
        else do;
            _s=coalesce(_1mspec_ - lag(_1mspec_), 0);
            _a=(_sensit_ + lag(_sensit_)) / 2 * _s;
        end;
    run;
    
    data &out;
	    set roc_gini end=eof;
	    retain auc;
	    auc + _a;
        if eof then do;
	        gini = 2*auc-1;
	        output;
        end;
	run;    
%mend;

%Gini(data=datlib.out_score, out=datlib.auc_gini_train, plot=1);
%Gini(data=datlib.out_score_test, out=datlib.auc_gini_test, plot=1);


/* Macro 3: Discriminatory power - Confusion table */
%macro ConfusionTable(data=, out=);
    /* Binarize predictions */
    data temp;
        set &data;
        predicted = ifc(PP > 0.5, 1, 0);
    run;

    /* Create confusion matrix */
    proc freq data=temp;
        tables high_income*predicted / nocum nopercent out=&out;
    run;
%mend;

%ConfusionTable(data=datlib.out_score, out=datlib.conf_matrix_train);
%ConfusionTable(data=datlib.out_score_test, out=datlib.conf_matrix_test);


/* Macro 4: Discriminatory power - F-score */
%macro FScore(data=, out=);
    /* Binarize predictions */
    data temp;
        set &data;
        predicted = ifc(PP > 0.5, 1, 0);
    run;

    /* Create confusion matrix */
    proc freq data=temp noprint;
        tables high_income*predicted / out=confusion;
    run;

    /* Calculate precision, recall, and F-score */
    data &out;
        set confusion;
        if high_income = 1 and predicted = 1 then precision = count / (count + sum(count));
        if high_income = 1 and predicted = 1 then recall = count / (count + sum(count));
        if high_income = 1 and predicted = 1 then F_score = 2 * (precision * recall) / (precision + recall);
        keep precision recall F_score;
    run;
%mend;

/* Call macros for both training and testing data */
%FScore(data=datlib.out_score, out=datlib.F_score_train);
%FScore(data=datlib.out_score_test, out=datlib.F_score_test);



/**** Accuracy ****/

/****************************************************************/
/* Macro 5: Accuracy overall - t-test                           */
/* This macro performs a paired t-test comparing the variables  */
/* high_income and PP. This comparison is used to assess the    */
/* accuracy of the model. The results of the t-test are output  */
/* to a specified data set for future reference.                */
/****************************************************************/
%macro TTest(data=, out=);
    /* Paired T-test */
    proc ttest data=&data;
        paired PP*high_income;
        ods output ttests=&out;
    run;
%mend;

/* Call macros for both training and testing data */
%TTest(data=datlib.out_score, out=datlib.t_test_train);
%TTest(data=datlib.out_score_test, out=datlib.t_test_test);


/****************************************************************/
/* Macro 5b: Accuracy with observations grouped by Grade - t-test*/
/* This macro performs a paired t-test comparing the variables  */
/* high_income and PP, separately for each grade. This comparison is used */
/* to assess the accuracy of the model within each grade. The results */
/* of the t-test are output to a specified data set for future reference. */
/****************************************************************/
%macro TTestGrade(data=, out=);
    /* Identify the unique levels of Grade */
    proc sql noprint;
        select distinct Grade 
        into :grades separated by ' ' 
        from &data;
    quit;

    /* Initialize the output dataset */
    data &out;
        length Grade $10 TestStatistic PValue 8;
        stop;
    run;

    /* Perform the paired T-test for each grade separately */
    %let n_grades = %sysfunc(countw(&grades));
    %do i = 1 %to &n_grades;
        %let grade = %scan(&grades, &i);
        data temp;
            set &data;
            where Grade = "&grade";
        run;

        proc ttest data=temp;
            paired PP*high_income;
            ods output ttests=temp_results;
        run;

        data temp_results;
            set temp_results;
            Grade = "&grade";
        run;

        data &out;
            set &out temp_results;
        run;
    %end;

    proc datasets lib=work nolist;
        delete temp temp_results;
    quit;
%mend;

/* Call macros for both training and testing data */
%TTestGrade(data=datlib.out_score, out=datlib.t_test_train_grade);
%TTestGrade(data=datlib.out_score_test, out=datlib.t_test_test_grade);


/****************************************************************/
/* Macro 6: Accuracy overall - binomial test */
/* This macro performs a binomial test comparing the number of "successes" */
/* (where high_income is 1) to the average predicted probability (PP). */
/* This is used to assess the accuracy of the model's predictions. */
/****************************************************************/
%macro BinomialTest(data=, out=);
    /* Calculate the average PP */
    proc means data=&data noprint;
        var PP;
        output out=temp mean=mean_PP;
    run;
    
    data _null_;
        set temp;
        call symput('avg_PP', mean_PP);
    run;

    /* Perform the binomial test */
    proc freq data=&data;
        tables high_income / binomial(p=&avg_PP);
        ods output BinomialProportion=&out;
    run;
%mend;

/* Call macros for both training and testing data */
%BinomialTest(data=datlib.out_score, out=datlib.binomial_test_train);
%BinomialTest(data=datlib.out_score_test, out=datlib.binomial_test_test);


/****************************************************************/
/* Macro 6b: Accuracy with observations grouped by Grade - binomial test */
/* This macro performs a binomial test comparing the number of "successes" */
/* (where high_income is 1) to the average predicted probability (PP), */
/* separately for each grade. This is used to assess the accuracy */
/* of the model within each grade. */
/****************************************************************/
%macro BinomialTestGrade(data=, out=);
    /* Identify the unique levels of Grade */
    proc sql noprint;
        select distinct Grade 
        into :grades separated by ' ' 
        from &data;
    quit;

    /* Initialize the output dataset */
    data &out;
        length Grade $10 Proportion LowerCL UpperCL 8;
        stop;
    run;

    /* Perform the binomial test for each grade separately */
    %let n_grades = %sysfunc(countw(&grades));
    %do i = 1 %to &n_grades;
        %let grade = %scan(&grades, &i);
        /* Calculate the average PP for the current grade */
        proc means data=&data(where=(Grade="&grade")) noprint;
            var PP;
            output out=temp mean=mean_PP;
        run;
        
        data _null_;
            set temp;
            call symput('avg_PP', mean_PP);
        run;
        
        /* Perform the binomial test */
        proc freq data=&data(where=(Grade="&grade"));
            tables high_income / binomial(p=&avg_PP);
             *ods output BinomialProportion=temp_results;
        run;
		/*
        data temp_results;
            set temp_results;
            Grade = "&grade";
        run;

        data &out;
            set &out temp_results;
        run;
        */
    %end;
%mend;

%BinomialTestGrade(data=datlib.out_score, out=datlib.binomial_test_train_grade);
%BinomialTestGrade(data=datlib.out_score_test, out=datlib.binomial_test_test_grade);


/**** Model Extension ****
To extend the logistic regression model in your SAS project by investigating the potential of an additional variable for enhancing model performance within the existing grades, the following approach can be taken:

    Identify Potential Predictors: Start by examining the existing variables in your dataset. Look for those not included in the current model which might have predictive power. This could involve variables that were excluded during the stepwise selection process or new variables derived from existing ones.

    Statistical Methods for Variable Selection: You can employ several statistical methods to identify the best predictors. These include:
        Chi-Square Test for Categorical Variables: If the potential predictor is categorical, use a chi-square test to determine if there is a significant relationship between this variable and the grades.
        ANOVA for Continuous Variables: If the potential predictor is continuous, use ANOVA to test if there are significant differences in this variable across different grades.
        Correlation Analysis: Analyze the correlation of each potential predictor with the outcome variable (high_income).

    Model Comparison: Once a new potential predictor is identified, fit a new logistic regression model that includes this variable. Then compare the performance of this new model with the existing model using metrics like AIC (Akaike Information Criterion), BIC (Bayesian Information Criterion), or changes in the Gini coefficient.

Here are the SAS macros to perform these tasks:

*-------------------------------------------------------;
* Macro for Chi-Square Test or ANOVA based on Variable Type
*-------------------------------------------------------;
%macro testPredictor(data, var);
    %if %sysfunc(vartype(&data, &var)) = N %then %do;
        /* ANOVA for Continuous Variable */
        proc anova data=&data;
            class Grade;
            model &var = Grade;
        run;
    %end;
    %else %do;
        /* Chi-Square Test for Categorical Variable */
        proc freq data=&data;
            tables Grade*&var / chisq;
        run;
    %end;
%mend testPredictor;

*-------------------------------------------------------;
* Macro for Correlation Analysis
*-------------------------------------------------------;
%macro correlationAnalysis(data, var);
    proc corr data=&data;
        var &var high_income;
    run;
%mend correlationAnalysis;

*-------------------------------------------------------;
* Macro for Model Comparison
*-------------------------------------------------------;
%macro compareModels(data, oldModel, newVar);
    /* Fit the old model */
    proc logistic data=&data descending;
        model high_income = &oldModel;
        output out=old_pred p=old_pp;
    run;

    /* Fit the new model with additional variable */
    proc logistic data=&data descending;
        model high_income = &oldModel &newVar;
        output out=new_pred p=new_pp;
    run;

    /* Model comparison */
    proc compare data=old_pred compare=new_pred;
    run;
%mend compareModels;

* How to Use These Macros 
Replace YourVariable with the name of the variable you want to test, and OldModelVars with the variables in your existing model.
These macros will provide a comprehensive analysis of the potential new variable's relationship with the grades and its impact on the model's performance.
;

* To test a potential predictor, use ;
%testPredictor(data=datlib.out_score, var=YourVariable);

* For correlation analysis, use ;
%correlationAnalysis(data=datlib.out_score, var=YourVariable);

* To compare models, use ;
%compareModels(data=datlib.out_split, oldModel=OldModelVars, newVar=NewVariable);

