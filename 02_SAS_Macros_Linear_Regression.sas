/* 
Program Header
---------------
Program: Boston Housing Dataset Analysis
Purpose: To develop a linear regression model on the Boston Housing dataset with MEDV as the dependent variable.
Methods: Importing data, descriptive statistics, checking for missing values, train-test split, automatic variable selection, linear regression, model scoring, model validation, performance testing, and interpretation.
Summary: This script begins by importing the Boston Housing dataset from a CSV file. It then conducts basic analysis, including printing the first few observations, calculating descriptive statistics, and checking for missing values. The data is then split into a training set and a test set. A linear regression model is developed using automatic variable selection (forward selection method) on the training data. The fitted model is then scored on both the training and test data. The performance of the model is evaluated using mean squared error. Finally, the results and interpretation of the analysis are presented.
*/

/****************************************************************/
**** Data import and prep ****;
/****************************************************************/
libname datlib "D:\E\Wissensbasis\Projekte\SAS_Macro_collection";

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

%importData('D:\E\Wissensbasis\Projekte\SAS_Macro_collection\housing.csv', datlib, HOUSING, ';');

%macro basicAnalysis(dataset);
    PROC PRINT DATA=&dataset (OBS=5); RUN;
    PROC MEANS DATA=&dataset; RUN;
    PROC MI DATA=&dataset; RUN;
%mend basicAnalysis;

%basicAnalysis(datlib.HOUSING);


/****************************************************************/
**** Train/Test Split ****;
/****************************************************************/
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


/****************************************************************/
**** Model Fitting and Scoring ****;
/****************************************************************/
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


/****************************************************************/
**** Model Performance Testing ****;
/****************************************************************/

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
