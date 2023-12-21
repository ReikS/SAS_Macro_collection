/****************************************************************
* PROGRAM NAME : SAS_Macro_collection_v01.sas
* DESCRIPTION : Collection of useful SAS macros, code snippets, etc.
*
* AUTHOR : ReikS
* CREATION DATE : 2023-19-22
* LAST CHANGE DATE : 2023-19-22
* REVIEWWER : <name of the reviewer>
* REVIEW DATE : <date of the review yyyy-mm-dd>
*	
*	
* SUMMARY : See below a list of the macros and code snippets contained
*           in this collection : 
*               000. Coding standards
*               001. Program header template
*				002. Macro docstring template
* 				110. Linear regression
*				120. Logistic regression
*
*
* REVIEW SUMMARY : <reviewer's notes>
* 
*
* INPUT : none
* 
* 
* OUTPUT : none
*
*
****************************************************************
* CHANGE TRACKER
* DATE			AUTHOR				DESCRIPTION
* <yyyy-mm-dd>	<name of author>	<short description>
*
****************************************************************/


****************************************************************
* 000. Coding standards
****************************************************************/
/**
* Adhere to the following coding standards.
* 1. The solution should consist of a single or a small number of scripts.
* 2. Each script has a program header.
* 3. Coding style to be used is functional programming. That means that custom functions are desinged for parts of the solution and then used for the overall solution.
* 4. Each function is commented with a doc string as well as inline comments. The doc string contains a detailed description on the function's arguments and the returned objects. Type hints shall be used.
* 5. PIP-8 applies.

* Use the following templape for the program header:
/**/

****************************************************************
* 001. Program header template
****************************************************************/

/****************************************************************
* PROGRAM NAME : <fill in name of the file>
* DESCRIPTION : <fill in short description>
*
* AUTHOR : <name of the author>
* CREATION DATE : <initial creation of the file in formal yyyy-mm-dd>
* LAST CHANGE DATE : <last change of the file in yyyy-mm-dd>
* REVIEWWER : <name of the reviewer>
* REVIEW DATE : <date of the review yyyy-mm-dd>
*	
*	
* SUMMARY : <detailed summary of this program>
* 
*
* REVIEW SUMMARY : <reviewer's notes>
* 
*
* INPUT : <description of input data, files, data sources, links, etc.>
* 
* 
* OUTPUT : <description of permanent output datasets, files, tables, etc.>
*
*
****************************************************************
* CHANGE TRACKER
* DATE			AUTHOR				DESCRIPTION
* <yyyy-mm-dd>	<name of author>	<short description>
*
****************************************************************/

****************************************************************
* 002. Macro docstring template
****************************************************************/ 
/*
This template provides a structured way to document important aspects of a SAS macro, 
including its name, purpose, parameters, usage example, authorship, and any special notes. 
Adjust the fields as needed for each specific macro.
*/

/*
Macro Name:
    MacroName

Description:
    [Brief description of what the macro does.]

Parameters:
    param1 (type): [Description of param1]
    param2 (type): [Description of param2]
    ...
    paramN (type): [Description of paramN]

Returns:
    [Description of what the macro returns or outputs, if applicable.]

Example:
    [Example of how to use the macro.]

Author:
    [Your Name]

Date:
    [Date of creation or last modification]

Notes:
    [Any additional notes or considerations]

*/

/* Macro code here */
* %macro MacroName(parameters);
* %mend MacroName;



****************************************************************
* 110. Linear regression
****************************************************************/ 

/* 
Program Header
---------------
Program: Boston Housing Dataset Analysis
Purpose: To develop a linear regression model on the Boston Housing dataset with MEDV as the dependent variable.
Methods: Importing data, descriptive statistics, checking for missing values, train-test split, automatic variable selection, linear regression, model scoring, model validation, performance testing, and interpretation.
Summary: This script begins by importing the Boston Housing dataset from a CSV file. It then conducts basic analysis, including printing the first few observations, calculating descriptive statistics, and checking for missing values. The data is then split into a training set and a test set. A linear regression model is developed using automatic variable selection (forward selection method) on the training data. The fitted model is then scored on both the training and test data. The performance of the model is evaluated using mean squared error. Finally, the results and interpretation of the analysis are presented.
*/


**** Data import and prep ****;

/*
Macro Name:
    importData

Description:
    Imports CSV data into a SAS dataset using PROC IMPORT.

Parameters:
    pathfile (string): The full path and filename of the CSV file to import.
    outlib (string): The libname where the dataset should be saved.
    outfile (string): The name of the output SAS dataset.
    delimiter (char): The delimiter used in the CSV file (e.g., ',', ';').

Returns:
    A SAS dataset is created in the specified library with the given outfile name.

Example:
    libname datlib "D:\E\Wissensbasis\Projekte\SAS_Macro_collection\SAS_Macro_Regression_examples";
    %importData('D:\E\Wissensbasis\Projekte\SAS_Macro_collection\SAS_Macro_Regression_examples\housing.csv', datlib, HOUSING, ';');

Author:
    [Your Name]

Date:
    [Date of creation or last modification]

Notes:
    The macro assumes that the first row of the CSV file contains column names.

*/
%macro importData(pathfile, outlib, outfile, delimiter);
    FILENAME REFFILE "&pathfile.";
    PROC IMPORT DATAFILE=REFFILE
        DBMS=CSV
        OUT=&outlib..&outfile
        REPLACE;
        GETNAMES=YES;
        DATAROW=2;
        DELIMITER=&delimiter.;
    RUN;
%mend importData;

/* Example call */
libname datlib "D:\E\Wissensbasis\Projekte\SAS_Macro_collection\SAS_Macro_Regression_examples";
%importData('D:\E\Wissensbasis\Projekte\SAS_Macro_collection\SAS_Macro_Regression_examples\housing.csv', datlib, HOUSING, ';');

%macro basicAnalysis(dataset);
    PROC PRINT DATA=&dataset (OBS=5); RUN;
    PROC MEANS DATA=&dataset; RUN;
    PROC MI DATA=&dataset; RUN;
%mend basicAnalysis;

%basicAnalysis(datlib.HOUSING);



**** Train/Test Split ****;

/* Perform stratified sampling */
PROC SORT DATA=datlib.HOUSING;
	BY MEDV;
RUN;

PROC SURVEYSELECT DATA=datlib.HOUSING 
				  OUT=datlib.HOUSING_SPIT 
				  method=srs 
                  outall 
                  seed=12345 
                  samprate=0.7;
    STRATA MEDV;
RUN;

/* Create training dataset */
DATA datlib.TRAIN;
    SET datlib.HOUSING_SPIT;
    WHERE selected = 1;
RUN;

/* Create testing dataset */
DATA datlib.TEST;
    SET datlib.HOUSING_SPIT;
    WHERE selected = 0;
RUN;


**** Model Fitting and Scoring ****;

%macro linearRegression(dataset, dependent, independent);
    PROC REG DATA=&dataset;
        MODEL &dependent = &independent / SELECTION=FORWARD;
        OUTPUT OUT=WORK.regOut P=PREDICTED;
    RUN;
%mend linearRegression;

%linearRegression(datlib.TRAIN, MEDV, CRIM INDUS CHAS NOX RM AGE DIS RAD TAX PTRATIO LSTAT);

%macro modelScoring(dataset, outlib, outfile);
    DATA &outlib..&outfile;
        SET &dataset;
        IF _N_ = 1 THEN SET WORK.regOut(KEEP=PREDICTED RENAME=(PREDICTED=PRED));
        ELSE SET WORK.regOut(KEEP=PREDICTED RENAME=(PREDICTED=PRED)) POINT=_N_;
    RUN;
%mend modelScoring;

%modelScoring(datlib.TRAIN, datlib, TRAIN_SCORED);
%modelScoring(datlib.TEST, datlib, TEST_SCORED);


**** Model Performance Testing ****;

%macro modelPerformance(dataset, dependent, predicted);
    /*
    Macro Header
    ------------
    Macro Name: modelPerformance
    Purpose: To evaluate the performance of a linear regression model.
    Parameters: 
        dataset - The name of the dataset to be analysed.
        dependent - The dependent variable.
        predicted - The predicted values from the model.
    */
    
    PROC MEANS DATA=&dataset NOPRINT;
        VAR &dependent &predicted;
        OUTPUT OUT=WORK.results 
               MEAN=&dependent &predicted
               STD=&dependent &predicted;
    RUN;
    
    DATA _NULL_;
        SET WORK.results;
        MAE = ABS(&dependent - &predicted);
        RMSE = SQRT((&dependent - &predicted)**2);
        PUT 'Mean Absolute Error (MAE) = ' MAE;
        PUT 'Root Mean Squared Error (RMSE) = ' RMSE;
    RUN;
    
    PROC REG DATA=&dataset;
        MODEL &dependent = &predicted;
        ODS OUTPUT FitStatistics=WORK.r2;
    RUN;
    
    DATA _NULL_;
        SET WORK.r2;
        IF Label1 = "R-Square" THEN PUT 'R-squared (R2) = ' Value1;
        IF Label1 = "Adj R-Sq" THEN PUT 'Adjusted R-squared (Adj R2) = ' Value1;
    RUN;
%mend modelPerformance;

%modelPerformance(datlib.TRAIN_SCORED, MEDV, PRED);
%modelPerformance(datlib.TEST_SCORED, MEDV, PRED);



****************************************************************
* 120. Logistic regression
****************************************************************/ 

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




****************************************************************
* 900. Transfer
****************************************************************/ 


****************************************************************
* 901. Pricing
****************************************************************/ 


**** Pricing in general **** ;
/**

Pricing strategies for retail products are diverse and complex, often tailored to the specific industry, customer base, market conditions, and product characteristics. Below is an overview of some commonly used pricing strategies in the retail sector, along with sources where you can find more detailed information.

Here's an extended overview of the pricing strategies for retail products, including conditions, assumptions, advantages, disadvantages, and relevant formulas where applicable:

### 1. Cost-Plus Pricing : This is one of the most straightforward pricing methods. Retailers add a mark-up to the cost of the product to determine its selling price. This method ensures that all costs are covered and a profit margin is achieved. For more details, you can refer to "Cost-plus Pricing" on [Wikipedia](https://en.wikipedia.org/wiki/Cost-plus_pricing).

**Conditions & Assumptions**:
   - Knowledge of the product's production cost.
   - Stable cost structure.
   
**Advantages**:
   - Simple to calculate and implement.
   - Ensures that costs are covered and a profit margin is achieved.

**Disadvantages**:
   - Ignores market conditions and customer value perception.
   - May lead to overpricing or underpricing in competitive markets.

**Formula**:
   \[ Price = Cost \times (1 + Markup) \]
   Where `Cost` is the production cost, and `Markup` is the desired profit margin as a percentage.


### 2. Value-Based Pricing : This strategy involves setting prices based on the perceived value of the product to the customer, rather than on the cost of production. It requires an understanding of the customer’s needs and the value they attach to the product. The article "Value-Based Pricing" on [Harvard Business Review](https://hbr.org/2016/08/a-quick-guide-to-value-based-pricing) provides a comprehensive overview.

**Conditions & Assumptions**:
   - Understanding of customers' perceived value of the product.
   - Flexibility in pricing according to customer segments.

**Advantages**:
   - Aligns price with customer value perception.
   - Potential for higher profit margins.

**Disadvantages**:
   - Difficult to accurately gauge customer perceived value.
   - Requires continuous market research.

**Formula**: No standard formula, as it relies on qualitative assessments of value.



### 3. Competitive Pricing : Retailers set their prices based on what their competitors are charging. This strategy is common in markets with many competitors selling similar products. The "Competition-based Pricing" page on [Wikipedia](https://en.wikipedia.org/wiki/Competition-based_pricing) offers more insights.

**Conditions & Assumptions**:
   - Knowledge of competitors' pricing.
   - Similar product offerings in the market.

**Advantages**:
   - Helps to stay competitive.
   - Reduces risk of pricing too high or too low.

**Disadvantages**:
   - May lead to price wars.
   - Undermines unique value proposition.

**Formula**: No standard formula, pricing is set relative to competitors.


### 4. Dynamic Pricing : Also known as surge pricing or demand pricing, this strategy involves changing prices in real-time based on demand, competition, and other market factors. It is often used by online retailers. A detailed explanation can be found in the article "What You Need to Know About Dynamic Pricing" on [Investopedia](https://www.investopedia.com/terms/d/dynamic-pricing.asp).

**Conditions & Assumptions**:
   - Ability to monitor market demand and competitors in real-time.
   - Flexible pricing infrastructure.

**Advantages**:
   - Maximizes profits by adapting to market conditions.
   - Can quickly respond to changes in demand.

**Disadvantages**:
   - Requires sophisticated technology and data analysis.
   - May frustrate customers if prices fluctuate frequently.

**Formula**: No fixed formula, prices are adjusted based on algorithms analyzing real-time data.


### 5. Psychological Pricing : This strategy leverages psychological factors to encourage purchasing. A common example is setting prices slightly lower than a round number (e.g., $9.99 instead of $10). The "Psychological Pricing" entry on [Wikipedia](https://en.wikipedia.org/wiki/Psychological_pricing) provides more information.

**Conditions & Assumptions**:
   - Customer sensitivity to pricing.
   - Psychological triggers that influence buying behavior.

**Advantages**:
   - Can increase sales through perceived bargains.
   - Simple to implement.

**Disadvantages**:
   - Overuse may diminish effectiveness.
   - May appear gimmicky to some customers.

**Formula**: Commonly, pricing is set just below a round number (e.g., $9.99 instead of $10).


### 6. Premium Pricing : Retailers set the prices of products significantly higher than competitors to create a perception of superior quality and exclusivity. This is often seen in luxury goods. For more on this, see "Premium Pricing" on [Investopedia](https://www.investopedia.com/terms/p/premium-pricing.asp).

**Conditions & Assumptions**:
   - High-quality or unique product offerings.
   - Target market willing to pay a premium.

**Advantages**:
   - Higher profit margins.
   - Enhances brand perception as high-end or exclusive.

**Disadvantages**:
   - Limited market reach.
   - Risk of being outpriced by competitors.

**Formula**: No standard formula, prices are set significantly higher than competitors.


### 7. Promotional Pricing : This short-term strategy involves temporarily reducing prices to attract customers and increase sales volume. It is often used during sales events or for product launches. The "Sales Promotion" page on [Wikipedia](https://en.wikipedia.org/wiki/Sales_promotion) discusses this strategy in more detail.

**Conditions & Assumptions**:
   - Ability to absorb temporary reduction in margins.
   - Attractive promotional offer.

**Advantages**:
   - Boosts sales volume.
   - Attracts new customers.

**Disadvantages**:
   - May erode profit margins if overused.
   - Customers may wait for promotions to make purchases.

**Formula**: Typically involves a temporary percentage discount (e.g., 20% off).

### 8. Bundle Pricing : This involves selling multiple products together at a price lower than if they were purchased individually. This can increase the perceived value and encourage customers to buy more. The "Price Bundling" article on [Wikipedia](https://en.wikipedia.org/wiki/Price_bundling) elaborates on this approach.

**Conditions & Assumptions**:
   - Complementary products available.
   - Customer interest in purchasing multiple items.

**Advantages**:
   - Encourages customers to buy more.
   - Increases perceived value.

**Disadvantages**:
   - Reduced revenue per individual item.
   - Bundled items may not always align with customer preferences.

**Formula**:
   \[ Bundle Price = \sum Individual Prices \times (1 - Discount) \]
   Where `Individual Prices` are the prices of each product in the bundle, and `Discount` is the percentage discount applied to the bundle.

Each pricing strategy should be carefully chosen based on the specific context, market conditions, and business objectives. Combining different strategies might also be effective in addressing various market segments and achieving diverse business goals.


**** Unsecured retail loans **** ;

Unsecured retail loans' pricing typically involves accounting for the cost of funds, expected losses (due to default), operating costs, and a margin for profit. 

Sources : 
https://www.experian.com/assets/decision-analytics/white-papers/regionalization-of-price-optimization-white-paper.pdf

Given your specific context, here's an overview of pricing model types and methodologies for unsecured retail loans:

### 1. **Cost-Plus Pricing**

**Methodology**: This is the simplest method. The price (or interest rate) is set by summing up the cost of funds, the expected credit loss, the operating expenses, and the required profit margin.

**Complexity**: Low

**Challenges**: 
- Does not account for competition or market dynamics.
- May overprice or underprice loans.

**Benefits**: 
Easy to understand and implement.
Transparent for regulatory concerns.

**Further Detail**:
- **Cost of Funds**: Reflects the bank's cost to acquire the money it lends. This can include interest paid on deposits or the cost of borrowing from other institutions.
- **Expected Credit Loss (ECL)**: Estimated loss over the loan's lifetime, considering the probability of default and loss given default.
- **Operating Expenses**: Costs related to processing and managing the loan.
- **Profit Margin**: The markup added to cover the bank's profit objectives.

**Sources/References**:
- "Bank Management and Financial Services" by Peter S. Rose and Sylvia C. Hudgins provides an overview of cost-plus pricing in banking.

### 2. **Risk-Based Pricing**

**Methodology**: Interest rates are set based on the estimated probability of default (PD) of the borrower. Borrowers with higher PDs are charged higher interest rates and vice versa. 

**Complexity**: Medium

**Challenges**: 
- Requires a robust risk assessment model.
- Might be perceived as discriminatory by some customers.

**Benefits**: 
- Aligns the price with risk.
- Allows for pricing optimization.

**Further Detail**:
- **Risk Assessment**: Evaluating the borrower's probability of default (PD) using credit scores or other risk factors.
- **Pricing Adjustment**: Setting interest rates based on the assessed risk level.

**Sources/References**:
- "Credit Risk Pricing Models: Theory and Practice" by Bernd Schmid provides a comprehensive look into risk-based pricing models.
- "Pricing and Risk Management of Synthetic CDOs" by Norbert Jobst and Stavros A. Zenios (Source: Operations Research).

### 3. **Competitor-Based Pricing**

**Methodology**: Pricing is set based on competitor rates and market dynamics. The bank might set its interest rates a little below, at par, or above the competition, depending on its value proposition.

**Complexity**: Medium

**Challenges**: 
- Requires consistent monitoring of competitors.
- Reactivity might lead to a race-to-the-bottom or away from strategic objectives.

**Benefits**: 
- Stays competitive in the market.
- Can attract customers if priced correctly.

**Further Detail**:
- **Market Analysis**: Regularly reviewing competitors’ rates and adjusting prices accordingly.
- **Strategic Positioning**: Deciding whether to price loans lower, at par, or higher than competitors, based on the bank's value proposition.

**Sources/References**:
- "The Strategy and Tactics of Pricing: A Guide to Growing More Profitably" by Thomas T. Nagle and Georg Müller provides insights into competitor-based pricing strategies.


### 4. **Yield Curve Based Pricing**

**Methodology**: This method takes into account the yield curve (term structure of interest rates). Longer-term loans are often exposed to greater interest rate risk, requiring a different pricing mechanism.

**Complexity**: High

**Challenges**: 
- Need to anticipate shifts in the yield curve.
- Complex to implement.

**Benefits**: 
- Better alignment of loan pricing with market conditions.
- Accounts for maturity mismatches in bank's assets and liabilities.

**Further Detail**:
- **Interest Rate Risk**: Accounting for the risk associated with changes in interest rates over different loan maturities.
- **Yield Curve Analysis**: Using the term structure of interest rates to price loans.

**Sources/References**:
- "Interest Rate Markets: A Practical Approach to Fixed Income" by Siddhartha Jha discusses yield curve-based pricing in financial markets.


### 5. **Behavioral Pricing**

**Methodology**: Prices are set based on customer behavior insights, using advanced analytics and segmentation.

**Complexity**: High

**Challenges**: 
- Requires deep customer data analysis.
- Might be perceived as manipulative or as a privacy concern.

**Benefits**: 
- Can optimize for customer retention and acquisition.
- Enables personalization of offers.

**Further Detail**:
- **Data Analysis**: Leveraging data on customer preferences, spending habits, and interactions.
- **Segmentation**: Offering different rates based on customer segments identified through behavior analysis.

**Sources/References**:
- "Customer-Centric Pricing: The Surprising Secret for Profitability" by Utpal M. Dholakia and Itamar Simonson provides insights into behavioral pricing strategies.


### 6. **Elasticity Based Pricing**

**Methodology**: This involves adjusting prices based on demand elasticity. Prices might be adjusted based on how sensitive customers are to price changes.

**Complexity**: High

**Challenges**: 
- Requires robust demand forecasting models.
- Complex to adjust in real-time.

**Benefits**: 
- Optimizes revenue.
- Can be more responsive to market changes.

**Further Detail**:
- **Demand Forecasting**: Predicting customer reactions to price changes.
- **Dynamic Pricing**: Adjusting rates in real-time or periodically based on demand elasticity.

**Sources/References**:
- "Pricing Strategies: A Marketing Approach" by Robert M. Schindler offers an in-depth look into elasticity-based pricing.
---

Given the constraints of the bank:

- For loans with terms over 60 months and amounts exceeding 30,000 EUR, risk-based pricing combined with the yield curve-based approach would be highly effective. This ensures that the risk and term structure of interest rates are adequately accounted for.
  
- Given the DSS environment by Schufa, a simpler approach such as Cost-Plus or Risk-Based might be more feasible for actual production. However, the model development, especially the risk estimation, can be performed using sophisticated tools like SAS, Python, or R.

Finally, while simplified structures are favored, it's essential to strike a balance between simplicity and accuracy. It's crucial to align the pricing strategy with the bank's overall business objectives and risk appetite.


**** Retail loan pricing formulas **** ;

Certainly, let's proceed step-by-step.

### Step 1: Risk-Based Pricing Model

The goal of a risk-based pricing model is to set an interest rate for a loan that appropriately compensates the lender for the risks taken. The rate should cover:

1. **Risk Costs (Expected Credit Loss)**: This represents the anticipated loss from the loan due to borrower defaults. It's a function of loan amount, probability of default, loss given default, and loan term.
   
2. **Margin**: This is the profit component, ensuring the lender is adequately compensated for its services and achieves its return objectives.

3. **Funding Costs**: Representing the costs the lender incurs to secure the funds that are lent out. For a bank, this might be the interest it pays on deposits or on interbank loans.

4. **Operational Costs**: Costs associated with the origination, maintenance, and servicing of the loan throughout its term.

Given these components, the formula for the risk-based pricing rate \( R \) is:

\[ R = Risk Costs + Margin + Funding Costs + Operational Costs \]

Given our previous discussion:

1. **Risk Costs (Expected Credit Loss)**:
\[ ECL = PD \times LGD \times \left( \frac{L \times T - L \times \frac{T(T+1)}{2T}}{T} \right) \]

2. **Margin**:
To be derived later using RARORAC.

3. **Funding Costs**: 
Let's denote this as \( C \) (constant per annum for the entire loan term due to hedging).

4. **Operational Costs**: 
Can be denoted as \( OC \). This can be a fixed value or a percentage of the loan amount.

Combining these, the rate \( R \) is:

\[ R = ECL + Margin + C + OC \]


Now, let's implement this in Python:

Note: The `Margin` parameter in this function is a placeholder, and its calculation using RARORAC will be detailed in the next steps.

```python

def risk_based_pricing_rate(PD: float, LGD: float, L: float, T: int, C: float, OC: float, Margin: float) -> float:
    """
    Calculate the risk-based pricing rate for a loan.
    
    Arguments:
    PD (float)      : Probability of Default for the borrower.
    LGD (float)     : Loss Given Default, represents the portion of the loan that will be lost if a default occurs.
    L (float)       : Loan amount.
    T (int)         : Loan term in months.
    C (float)       : Funding cost, the cost the bank incurs to secure the funds lent out (per annum).
    OC (float)      : Operational cost associated with the origination, maintenance, and servicing of the loan (per annum).
    Margin (float)  : Profit component to ensure the lender is compensated for its services.
    
    Returns:
    R (float)       : The interest rate set for the loan to cover risk, funding and operational costs, as well as the desired margin.
    """
    
    ECL = PD * LGD * (L * T - L * (T*(T+1))/(2*T)) / T
    R = ECL + Margin + C + OC
    return R

# Testing Facility:
if __name__ == "__main__":
    # Sample test values
    PD = 0.03   # 3% probability of default
    LGD = 0.5  # 50% loss given default
    L = 10000  # Loan amount of 10,000 EUR
    T = 60     # Loan term of 60 months (5 years)
    C = 0.02   # 2% funding cost per annum
    OC = 0.01  # 1% operational cost per annum
    Margin = 0.03  # 3% margin for profit
    
    # Calculate and print the risk-based pricing rate
    R = risk_based_pricing_rate(PD, LGD, L, T, C, OC, Margin)
    print(f"The risk-based pricing rate for the loan is: {R:.2%}")

```

The Capital Adequacy Ratio (CAR) is a measure used by banks to determine the adequacy of their capital keeping in view their risk exposures. It represents the proportion of a bank's capital to its risk-weighted assets. This ratio ensures that banks have enough capital on reserve to absorb a reasonable amount of loss before becoming insolvent.
Regulatory bodies, such as the Basel Committee on Banking Supervision, set minimum CAR standards to ensure that banks can absorb a reasonable amount of loss. For instance, under the Basel III standards, the minimum CAR is set at 8%, though individual countries' banking regulators may require higher levels.
However, while there is a regulatory minimum, individual banks often set a higher internal CAR based on their risk appetite, strategic objectives, and market conditions. This internal CAR is indeed a strategic choice by the bank. Banks often maintain a CAR above the regulatory minimum to ensure a buffer against unforeseen losses and to convey financial robustness to investors and customers.
