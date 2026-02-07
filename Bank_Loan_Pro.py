# -*- coding: utf-8 -*-
"""
Bank Loan Approval Prediction System
=====================================
A PySpark-based machine learning pipeline for predicting bank loan approvals.
This project implements multiple classification models to analyze loan applications
and predict approval status.

Models Implemented:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Support Vector Machine (Linear SVC)
    - Gradient Boosted Trees (GBT)

Author: Bank Loan Prediction Team
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Data settings
    "dataset_path": "loan_data.csv",
    
    # Train-test split
    "train_ratio": 0.8,
    "test_ratio": 0.2,
    "random_seed": 42,
    
    # Model hyperparameters
    "rf_num_trees": 100,
    "svm_max_iter": 100,
    "gbt_max_iter": 20,
    "lr_max_iter": 100,
    
    # Feature columns
    "numeric_features": ["LoanAmount", "Loan_Amount_Term", "Credit_History"],
    "categorical_features": ["Gender", "Married", "Dependents", "Self_Employed"],
    "target_column": "Loan_Status",
    
    # Columns to drop
    "drop_columns": ["Unnamed: 0", "Loan_ID"],
}

# =============================================================================
# IMPORTS
# =============================================================================

import sys
import logging
from typing import Dict, List, Tuple, Any

# PySpark imports
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, mean, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, 
    OneHotEncoder, 
    VectorAssembler, 
    StandardScaler
)
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    RandomForestClassifier,
    LinearSVC,
    LogisticRegression,
    GBTClassifier
)
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)
from pyspark.mllib.evaluation import MulticlassMetrics

# Data science imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_spark_session(app_name: str = "BankLoanApproval") -> SparkSession:
    """
    Create and return a SparkSession.
    
    Args:
        app_name: Name of the Spark application
        
    Returns:
        SparkSession instance
    """
    logger.info(f"Creating SparkSession: {app_name}")
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
    
    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_and_validate_data(spark: SparkSession, path: str) -> DataFrame:
    """
    Load CSV data and validate its structure.
    
    Args:
        spark: SparkSession instance
        path: Path to the CSV file
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If the dataset file doesn't exist
        ValueError: If required columns are missing
    """
    logger.info(f"Loading data from: {path}")
    
    try:
        df = spark.read.csv(path, header=True, inferSchema=True)
        logger.info(f"Loaded {df.count()} rows with {len(df.columns)} columns")
        
        # Validate required columns exist
        required_cols = ["LoanAmount", "Loan_Status", "ApplicantIncome"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def handle_missing_values(df: DataFrame, config: Dict) -> DataFrame:
    """
    Handle missing values in the dataset.
    
    - Numeric columns: Fill with mean
    - Categorical columns: Fill with mode (most frequent value)
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        
    Returns:
        DataFrame with missing values handled
    """
    logger.info("Handling missing values...")
    
    # Handle numeric features - fill with mean
    for feature in config["numeric_features"]:
        if feature in df.columns:
            avg_value = df.select(mean(col(feature))).collect()[0][0]
            if avg_value is not None:
                df = df.fillna({feature: avg_value})
                logger.info(f"  {feature}: filled with mean = {avg_value:.2f}")
    
    # Handle categorical features - fill with mode
    for feature in config["categorical_features"]:
        if feature in df.columns:
            mode_row = df.groupBy(feature).count() \
                .orderBy("count", ascending=False) \
                .first()
            if mode_row and mode_row[0] is not None:
                df = df.fillna({feature: mode_row[0]})
                logger.info(f"  {feature}: filled with mode = {mode_row[0]}")
    
    return df


def create_feature_engineering_pipeline(
    categorical_cols: List[str],
    numeric_cols: List[str]
) -> Tuple[List, List[str]]:
    """
    Create preprocessing stages for the ML pipeline.
    
    Args:
        categorical_cols: List of categorical column names
        numeric_cols: List of numeric column names
        
    Returns:
        Tuple of (pipeline stages, feature column names)
    """
    stages = []
    
    # String indexers for categorical columns + target
    indexed_cols = []
    for col_name in categorical_cols:
        indexed_name = f"{col_name}_idx"
        indexer = StringIndexer(
            inputCol=col_name, 
            outputCol=indexed_name,
            handleInvalid="keep"
        )
        stages.append(indexer)
        indexed_cols.append(indexed_name)
    
    # Target column indexer
    target_indexer = StringIndexer(
        inputCol="Loan_Status",
        outputCol="Loan_Status_idx",
        handleInvalid="keep"
    )
    stages.append(target_indexer)
    
    # One-hot encoders
    encoded_cols = []
    for col_name in categorical_cols:
        encoded_name = f"{col_name}_vec"
        encoder = OneHotEncoder(
            inputCol=f"{col_name}_idx",
            outputCol=encoded_name
        )
        stages.append(encoder)
        encoded_cols.append(encoded_name)
    
    # Feature columns for assembler
    feature_cols = numeric_cols + encoded_cols
    
    # Vector assembler
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="raw_features",
        handleInvalid="keep"
    )
    stages.append(assembler)
    
    # Standard scaler
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="scaled_features",
        withStd=True,
        withMean=False
    )
    stages.append(scaler)
    
    return stages, feature_cols


def engineer_features(df: DataFrame) -> DataFrame:
    """
    Create derived features for better model performance.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with new engineered features
    """
    logger.info("Engineering features...")
    
    # Combined income
    df = df.withColumn(
        "Combined_Income", 
        col("ApplicantIncome") + col("CoapplicantIncome")
    )
    
    # Income to loan ratio
    df = df.withColumn(
        "Income_Loan_Ratio",
        when(col("LoanAmount") > 0, col("Combined_Income") / col("LoanAmount"))
        .otherwise(0)
    )
    
    logger.info("  Created: Combined_Income, Income_Loan_Ratio")
    return df


# =============================================================================
# MODEL TRAINING FUNCTIONS
# =============================================================================

def get_model_configs(config: Dict) -> Dict[str, Any]:
    """
    Get configuration for all models to train.
    
    Args:
        config: Main configuration dictionary
        
    Returns:
        Dictionary of model name to model instance
    """
    return {
        "Logistic Regression": LogisticRegression(
            labelCol="Loan_Status_idx",
            featuresCol="scaled_features",
            maxIter=config["lr_max_iter"]
        ),
        "Decision Tree": DecisionTreeClassifier(
            labelCol="Loan_Status_idx",
            featuresCol="scaled_features"
        ),
        "Random Forest": RandomForestClassifier(
            labelCol="Loan_Status_idx",
            featuresCol="scaled_features",
            numTrees=config["rf_num_trees"]
        ),
        "SVM (LinearSVC)": LinearSVC(
            labelCol="Loan_Status_idx",
            featuresCol="scaled_features",
            maxIter=config["svm_max_iter"]
        ),
        "Gradient Boosted Trees": GBTClassifier(
            labelCol="Loan_Status_idx",
            featuresCol="scaled_features",
            maxIter=config["gbt_max_iter"]
        )
    }


def train_models(
    train_data: DataFrame,
    test_data: DataFrame,
    config: Dict
) -> Dict[str, Tuple[Any, DataFrame]]:
    """
    Train all classification models and generate predictions.
    
    Args:
        train_data: Training DataFrame
        test_data: Testing DataFrame
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping model name to (trained_model, predictions)
    """
    models = get_model_configs(config)
    results = {}
    
    logger.info("Training models...")
    
    for name, model in models.items():
        logger.info(f"  Training: {name}")
        try:
            trained_model = model.fit(train_data)
            predictions = trained_model.transform(test_data)
            results[name] = (trained_model, predictions)
            logger.info(f"  ✓ {name} trained successfully")
        except Exception as e:
            logger.error(f"  ✗ Error training {name}: {e}")
    
    return results


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_model(predictions: DataFrame, model_name: str) -> Dict[str, float]:
    """
    Evaluate a model's predictions using multiple metrics.
    
    Args:
        predictions: DataFrame with predictions
        model_name: Name of the model for logging
        
    Returns:
        Dictionary of metric names to values
    """
    evaluator = MulticlassClassificationEvaluator(
        labelCol="Loan_Status_idx",
        predictionCol="prediction"
    )
    
    metrics = {
        "accuracy": evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"}),
        "precision": evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"}),
        "recall": evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"}),
        "f1": evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
    }
    
    # ROC-AUC for binary classification
    try:
        binary_evaluator = BinaryClassificationEvaluator(
            labelCol="Loan_Status_idx",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        metrics["roc_auc"] = binary_evaluator.evaluate(predictions)
    except Exception:
        metrics["roc_auc"] = None
    
    return metrics


def print_evaluation_results(all_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Print and return a formatted table of evaluation results.
    
    Args:
        all_results: Dictionary of model name to metrics
        
    Returns:
        DataFrame with results
    """
    print("\n" + "=" * 80)
    print("MODEL EVALUATION RESULTS")
    print("=" * 80)
    
    results_df = pd.DataFrame(all_results).T
    results_df = results_df.round(4)
    
    print(results_df.to_string())
    print("=" * 80)
    
    return results_df


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_loan_status_distribution(df: DataFrame) -> None:
    """Plot the distribution of loan approval status."""
    eda_df = df.select("Loan_Status").toPandas()
    
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=eda_df, x="Loan_Status", palette="viridis")
    plt.title("Loan Status Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Loan Status")
    plt.ylabel("Count")
    
    # Add value labels on bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig("loan_status_distribution.png", dpi=150)
    plt.show()


def plot_income_by_status(df: DataFrame) -> None:
    """Plot income distribution by loan status."""
    eda_df = df.select("Loan_Status", "Combined_Income").toPandas()
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=eda_df, x="Loan_Status", y="Combined_Income", palette="Set2")
    plt.title("Income Distribution by Loan Status", fontsize=14, fontweight='bold')
    plt.xlabel("Loan Status")
    plt.ylabel("Combined Income")
    plt.tight_layout()
    plt.savefig("income_by_status.png", dpi=150)
    plt.show()


def plot_credit_history_approval(df: DataFrame) -> None:
    """Plot loan approval rate by credit history."""
    eda_df = df.select("Credit_History", "Loan_Status").toPandas()
    eda_df["Approved"] = eda_df["Loan_Status"].apply(lambda x: 1 if x == "Y" else 0)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(data=eda_df, x="Credit_History", y="Approved", palette="coolwarm")
    plt.title("Loan Approval Rate by Credit History", fontsize=14, fontweight='bold')
    plt.xlabel("Credit History")
    plt.ylabel("Approval Rate")
    plt.tight_layout()
    plt.savefig("credit_history_approval.png", dpi=150)
    plt.show()


def plot_confusion_matrix(predictions: DataFrame, model_name: str) -> None:
    """
    Plot confusion matrix for a model.
    
    Args:
        predictions: DataFrame with predictions
        model_name: Name of the model
    """
    pred_rdd = predictions.select("Loan_Status_idx", "prediction").rdd.map(tuple)
    metrics = MulticlassMetrics(pred_rdd)
    cm_array = metrics.confusionMatrix().toArray()
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_array, 
        annot=True, 
        fmt="g", 
        cmap="Blues",
        xticklabels=["Rejected", "Approved"],
        yticklabels=["Rejected", "Approved"]
    )
    plt.title(f"{model_name} - Confusion Matrix", fontsize=12, fontweight='bold')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png", dpi=150)
    plt.show()


def plot_roc_curves(model_results: Dict[str, Tuple[Any, DataFrame]]) -> None:
    """
    Plot ROC curves for all models on a single figure.
    
    Args:
        model_results: Dictionary of model results
    """
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for (name, (model, predictions)), color in zip(model_results.items(), colors):
        try:
            # Get true labels
            true_labels = predictions.select("Loan_Status_idx").rdd.flatMap(lambda x: x).collect()
            
            # Get prediction scores
            if 'probability' in predictions.columns:
                pred_scores = predictions.select("probability").rdd.map(
                    lambda row: float(row[0][1])
                ).collect()
            elif 'rawPrediction' in predictions.columns:
                pred_scores = predictions.select("rawPrediction").rdd.map(
                    lambda row: float(row[0][1])
                ).collect()
            else:
                continue
            
            fpr, tpr, _ = roc_curve(true_labels, pred_scores)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=color, lw=2, 
                     label=f'{name} (AUC = {roc_auc:.3f})')
                     
        except Exception as e:
            logger.warning(f"Could not plot ROC for {name}: {e}")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curves_comparison.png", dpi=150)
    plt.show()


def plot_model_comparison(results_df: pd.DataFrame) -> None:
    """
    Plot bar chart comparing all models across metrics.
    
    Args:
        results_df: DataFrame with model results
    """
    # Prepare data
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    plot_df = results_df[metrics_to_plot].copy()
    
    x = np.arange(len(plot_df.index))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, plot_df[metric], width, label=metric.capitalize(), color=color)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.show()


def plot_feature_importance(model, feature_names: List[str], model_name: str) -> None:
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with featureImportances attribute
        feature_names: List of feature names
        model_name: Name of the model
    """
    try:
        importances = model.featureImportances.toArray()
        
        # Handle case where feature names don't match importances
        if len(feature_names) != len(importances):
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(15)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=importance_df, 
            x="Importance", 
            y="Feature", 
            palette="viridis"
        )
        plt.title(f"Top 15 Feature Importance - {model_name}", fontsize=12, fontweight='bold')
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(f"feature_importance_{model_name.replace(' ', '_').lower()}.png", dpi=150)
        plt.show()
        
    except AttributeError:
        logger.warning(f"{model_name} does not support feature importance")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function for the Bank Loan Approval prediction pipeline."""
    
    print("\n" + "=" * 80)
    print("  BANK LOAN APPROVAL PREDICTION SYSTEM")
    print("=" * 80 + "\n")
    
    spark = None
    
    try:
        # Step 1: Initialize Spark
        spark = create_spark_session()
        
        # Step 2: Load and validate data
        loan_data = load_and_validate_data(spark, CONFIG["dataset_path"])
        
        # Step 3: Drop unnecessary columns
        for col_name in CONFIG["drop_columns"]:
            if col_name in loan_data.columns:
                loan_data = loan_data.drop(col_name)
        
        # Display schema
        print("\nDataset Schema:")
        loan_data.printSchema()
        print("\nSample Data:")
        loan_data.show(5)
        
        # Step 4: Handle missing values
        loan_data = handle_missing_values(loan_data, CONFIG)
        
        # Step 5: Feature engineering
        loan_data = engineer_features(loan_data)
        
        # Step 6: Create preprocessing pipeline
        categorical_cols = ["Gender", "Married", "Dependents", "Education", 
                           "Self_Employed", "Property_Area"]
        numeric_cols = ["LoanAmount", "Loan_Amount_Term", "Credit_History",
                       "Combined_Income", "Income_Loan_Ratio"]
        
        pipeline_stages, feature_names = create_feature_engineering_pipeline(
            categorical_cols, numeric_cols
        )
        
        # Fit and transform data
        preprocessing_pipeline = Pipeline(stages=pipeline_stages)
        loan_data = preprocessing_pipeline.fit(loan_data).transform(loan_data)
        
        # Step 7: EDA Visualizations
        logger.info("Generating EDA visualizations...")
        plot_loan_status_distribution(loan_data)
        plot_income_by_status(loan_data)
        plot_credit_history_approval(loan_data)
        
        # Step 8: Train-test split
        train_data, test_data = loan_data.randomSplit(
            [CONFIG["train_ratio"], CONFIG["test_ratio"]], 
            seed=CONFIG["random_seed"]
        )
        logger.info(f"Train size: {train_data.count()}, Test size: {test_data.count()}")
        
        # Step 9: Train models
        model_results = train_models(train_data, test_data, CONFIG)
        
        # Step 10: Evaluate models
        all_metrics = {}
        for name, (model, predictions) in model_results.items():
            metrics = evaluate_model(predictions, name)
            all_metrics[name] = metrics
        
        results_df = print_evaluation_results(all_metrics)
        
        # Step 11: Generate visualizations
        logger.info("Generating model evaluation visualizations...")
        
        # Confusion matrices
        for name, (model, predictions) in model_results.items():
            plot_confusion_matrix(predictions, name)
        
        # ROC curves
        plot_roc_curves(model_results)
        
        # Model comparison chart
        plot_model_comparison(results_df)
        
        # Feature importance (for tree-based models)
        for name in ["Decision Tree", "Random Forest", "Gradient Boosted Trees"]:
            if name in model_results:
                model, _ = model_results[name]
                plot_feature_importance(model, feature_names, name)
        
        # Step 12: Find best model
        best_model = results_df["f1"].idxmax()
        best_f1 = results_df.loc[best_model, "f1"]
        
        print("\n" + "=" * 80)
        print(f"  BEST MODEL: {best_model}")
        print(f"  F1-Score: {best_f1:.4f}")
        print("=" * 80 + "\n")
        
        logger.info("Pipeline completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        print("\nPlease ensure the dataset file exists at the specified path.")
        print("You can download the dataset from Kaggle or use the sample data.")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
        
    finally:
        # Cleanup
        if spark:
            logger.info("Stopping SparkSession...")
            spark.stop()


if __name__ == "__main__":
    main()
