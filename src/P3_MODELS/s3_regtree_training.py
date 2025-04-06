"""
@roman_avj
5/4/25

This module is used to train a Regression Tree
"""
# %% Imports
import joblib
import logging
import warnings
import yaml

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker

import mlflow
from mlflow.models import infer_signature

import numpy as np
import pandas as pd
import shap
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler


# Settings
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# dir
DIR_DATA = '../../data'
DIR_RESULTS = '../../results'
DIR_MODELS = '../../models'
FILE_PROPERTIES = f"{DIR_DATA}/clean/properties_shif.parquet"


# Functions
# configs
def get_configs(filepath):
    with open(filepath, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            config_model = config['model']
            recaster_mappers = config['categorical_recasters']
            mlflow_config = config['mlflow']
        except yaml.YAMLError as exc:
            print(exc)
    return config_model, recaster_mappers, mlflow_config


# utils
def get_target(df, config):
    # read
    target_name = list(config['target'].keys())[0]

    # if ppsm2 is in target_name, then we need to calculate the target
    if 'ppsm' in target_name:
        if 'built' in target_name:
            y = df['price'] / df['built_area']
        elif 'saleable' in target_name:
            y = df['price'] / df['saleable_area']
        elif 'land' in target_name:
            y = df['price'] / df['land_area']
        else:
            raise ValueError('Target is not supported')
    elif 'price' in target_name:
        y = df['price']
    else:
        raise ValueError('Target is not supported')
    return y.copy()


def get_target_value(y, transformer):
    return transformer.inverse_transform_target(y)


def get_predictions(model, data, transformer):
    return get_target_value(model.predict(data), transformer)


# data
def read_data(file, config):
    # read data
    df_work = pd.read_parquet(file)

    # filter data
    n_rows = config.get('data').get('n_rows')
    if n_rows:
        if n_rows < df_work.shape[0]:
            df_work = df_work.sample(n_rows, random_state=42).reset_index(drop=True)

    # set observation_id as index
    df_work = df_work.set_index('observation_id')

    # new features
    first_date_obs = df_work['valuation_date'].min()
    df_work = (
        df_work
        .assign(
            # quarters since appraisal
            quarters_since_first_appraisal=lambda x: (
                x['valuation_date'] - first_date_obs
                ).dt.days / (30.4 * 3),
            # recategorizations
            is_new=lambda x:
                np.where(x['age_in_months'].le(1), 'new', 'used'),
            conservacion_recat=lambda x:
                x['conservation_status_id'].replace({7: 3.5}).astype(int),
            is_renovated=lambda x:
                np.where(x['conservation_status_id'].ge(7), 'renovated', 'not_renovated'),
        )
        .assign(
            is_new=lambda x: x['is_new'].astype('category'),
        )
    )

    # set target
    features_name = list(config['features'].keys())
    X = df_work.loc[:, features_name].copy()
    y = get_target(df_work, config)

    return X, y


# feature engineering
def feature_engineering(X, y, config, recaster_mapper=None):
    transformers = {}  # Dictionary to store all fitted transformers

    # Step 1: Check and handle missing values
    missing_values = X.isnull().sum().sum()
    if missing_values > 0:
        logger.warning(f"{missing_values} missing values detected in the dataset")

    # Step 2: Apply recaster mapping if provided
    if recaster_mapper:
        try:
            X = X.replace(recaster_mapper)
            logger.info("Applied recaster mapping successfully")
        except Exception as e:
            logger.error(f"Error applying recaster mapping: {str(e)}")
            raise ValueError(f"Failed to apply recaster mapping: {str(e)}")

    # Step 3: Handle categorical columns
    try:
        cols_categorical = list(set(
            X.select_dtypes(include=['string', 'category', 'object']).columns.tolist()
        ))
        X[cols_categorical] = X[cols_categorical].astype('category')
        transformers['categorical'] = cols_categorical
        logger.info(f"Processed {len(cols_categorical)} categorical columns")
    except Exception as e:
        logger.error(f"Error processing categorical columns: {str(e)}")
        raise ValueError(f"Failed to process categorical columns: {str(e)}")

    # Step 4: Apply feature transformations based on config
    valid_transformations = ['identity', 'log', 'sqrt', 'standardize', 'boxcox']
    X_transformed = X.copy()

    # Validate and group transformations
    transformation_groups = {trans_type: [] for trans_type in valid_transformations}
    for feature, transformation in config['features'].items():
        if transformation not in valid_transformations:
            raise ValueError(f"Unsupported transformation type for feature '{feature}': '{transformation}'. "
                             f"Supported types are: {', '.join(valid_transformations)}")
        if feature not in X.columns:
            logger.warning(f"Feature '{feature}' not found in the dataset. Skipping transformation.")
            continue
        if transformation in ['log', 'boxcox'] and (X[feature].min() <= 0 or pd.isna(X[feature]).any()):
            logger.warning(f"Feature '{feature}' contains values <= 0 or NaN. Using standardize instead.")
            transformation = 'standardize'
        if transformation == 'sqrt' and (X[feature].min() < 0 or pd.isna(X[feature]).any()):
            logger.warning(f"Feature '{feature}' contains values < 0 or NaN. Using standardize instead.")
            transformation = 'standardize'
        transformation_groups[transformation].append(feature)

    # Apply transformations
    for transformation, features in transformation_groups.items():
        if not features or transformation == 'identity':
            logger.info(f"Skipping '{transformation}' transformation for {len(features)} features")
            continue
        try:
            logger.info(f"Applying '{transformation}' transformation to {len(features)} features")
            X_transformed[features], transformer = apply_transformation(
                X[features], transformation, feature_name=features
            )
            if transformer is not None:
                transformers[transformation] = transformer
        except Exception as e:
            logger.error(f"Error applying {transformation} transformation to features {features}: {str(e)}")
            raise ValueError(f"Failed to apply {transformation} transformation: {str(e)}")

    logger.info("Applied transformations to features successfully")

    # Step 5: Transform target variable if specified
    if 'target' in config:
        target_name = list(config['target'].keys())[0]
        target_transformation = config['target'][target_name]
        if target_transformation not in valid_transformations:
            raise ValueError(f"Unsupported transformation type for target: '{target_transformation}'. "
                             f"Supported types are: {', '.join(valid_transformations)}")
        try:
            if target_transformation in ['log', 'boxcox', 'sqrt'] and (np.min(y) <= 0 or np.isnan(y).any()):
                logger.warning(f"Target contains values <= 0 or NaN. Using standardize instead.")
                target_transformation = 'standardize'
            y_transformed, target_transformer = apply_transformation(
                y.to_frame(), target_transformation, feature_name=target_name
            )
            if isinstance(y_transformed, np.ndarray):
                y_transformed = pd.Series(y_transformed.flatten(), index=y.index)
            else:
                y_transformed = pd.Series(y_transformed.to_numpy().flatten(), index=y.index)
            if target_transformer is not None:
                transformers["target"] = target_transformer
            logger.info(f"Applied '{target_transformation}' transformation to target variable")
        except Exception as e:
            logger.error(f"Error transforming target variable: {str(e)}")
            raise ValueError(f"Failed to transform target variable: {str(e)}")
    else:
        logger.warning("No target transformation specified in config")

    return X_transformed, y_transformed, transformers


def apply_transformation(data, transformation_type, feature_name=None):
    name_info = f" for '{feature_name}'" if feature_name else ""

    if transformation_type == 'identity':
        return data, None

    if transformation_type == 'log':
        transformed_data = np.log(data)
        transformer = StandardScaler()
        transformed_data = transformer.fit_transform(transformed_data)
        return transformed_data, transformer

    if transformation_type == 'sqrt':
        transformed_data = np.sqrt(data)
        transformer = StandardScaler()
        transformed_data = transformer.fit_transform(transformed_data)
        return transformed_data, transformer

    if transformation_type == 'standardize':
        transformer = StandardScaler()
        transformed_data = transformer.fit_transform(data)
        return transformed_data, transformer

    if transformation_type == 'boxcox':
        transformer = PowerTransformer(method='box-cox', standardize=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            transformed_data = transformer.fit_transform(data)
        return transformed_data, transformer

    raise ValueError(f"Unsupported transformation type{name_info}: '{transformation_type}'")


def save_transformers(transformers, filepath):
    try:
        joblib.dump(transformers, filepath)
        logger.info(f"Saved transformers to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving transformers: {str(e)}")
        return False


def load_transformers(filepath):
    try:
        transformers = joblib.load(filepath)
        logger.info(f"Loaded transformers from {filepath}")
        return transformers
    except Exception as e:
        logger.error(f"Error loading transformers: {str(e)}")
        raise ValueError(f"Failed to load transformers: {str(e)}")


def transform_features(X, transformers):
    X_transformed = X.copy()

    # Apply categorical transformations
    if 'categorical' in transformers:
        cols_categorical = transformers['categorical']
        present_cols = [col for col in cols_categorical if col in X.columns]
        X[present_cols] = X[present_cols].astype('category')

    for transformation_type, features in transformers.items():
        if transformation_type == 'categorical' and len(features) > 0:
            cols_categorical = transformers['categorical']
            present_cols = [col for col in cols_categorical if col in X.columns]
            X[present_cols] = X[present_cols].astype('category')

        elif transformation_type == 'log' and len(features) > 0:
            X_transformed[features] = np.log(X[features])
            transformer = transformers['log']
            X_transformed[features] = transformer.transform(X_transformed[features])

        elif transformation_type == 'sqrt' and len(features) > 0:
            X_transformed[features] = np.sqrt(X[features])
            transformer = transformers['sqrt']
            X_transformed[features] = transformer.transform(X_transformed[features])

        elif transformation_type == 'standardize' and len(features) > 0:
            transformer = transformers['standardize']
            X_transformed[features] = transformer.transform(X[features])

        elif transformation_type == 'boxcox' and len(features) > 0:
            transformer = transformers['boxcox']
            X_transformed[features] = transformer.transform(X[features])

    return X_transformed


def inverse_transform_target(y_pred, transformation_type, transformers):
    if "target" not in transformers or transformers["target"] is None:
        return y_pred

    try:
        # inverse transform target
        y_pred_reshaped = y_pred.reshape(-1, 1) if len(y_pred.shape) == 1 else y_pred
        inverse_transformed = transformers["target"].inverse_transform(y_pred_reshaped)

        # case for log and sqrt (because is not a pipeline)
        if transformation_type == 'log':
            inverse_transformed = np.exp(inverse_transformed)
        elif transformation_type == 'sqrt':
            inverse_transformed = inverse_transformed ** 2

        if inverse_transformed.shape[1] == 1:
            inverse_transformed = inverse_transformed.flatten()

        return inverse_transformed
    except Exception as e:
        logger.error(f"Error inverse transforming target: {str(e)}")
        raise ValueError(f"Failed to inverse transform target: {str(e)}")


class FeatureTransformer:
    def __init__(self, config, recaster_mapper=None):
        self.config = config
        self.recaster_mapper = recaster_mapper
        self.transformers_ = None
        self.target_transformation_ = list(config.get('target').values())[0]

    def fit(self, X, y=None):
        """Fit the transformer on the training data."""
        _, _, self.transformers_ = feature_engineering(X, y, self.config, self.recaster_mapper)
        return self

    def transform(self, X):
        """Transform the features."""
        return transform_features(X, self.transformers_)

    def fit_transform(self, X, y=None):
        """Fit the transformer and transform the features."""
        X_transformed, y_transformed, self.transformers_ = feature_engineering(
            X, y, self.config, self.recaster_mapper
        )
        return X_transformed, y_transformed

    def get_transformers(self):
        """Get the fitted transformers."""
        return self.transformers_

    def save_transformers(self, filepath):
        """Save the fitted transformers to a file."""
        return save_transformers(self.transformers_, filepath)

    def load_transformers(self, filepath):
        """Load transformers from a file."""
        self.transformers_ = load_transformers(filepath)
        return self

    def inverse_transform_target(self, y_pred):
        """Inverse transform the target variable."""
        return inverse_transform_target(
            y_pred, self.target_transformation_, self.transformers_
        )


# split data and prepare CatBoost data format
def split_randomly_data(X, y, config, categorical_features=None):
    """
    Split data into train, validation, and test sets based on the provided configuration.

    Args:
        X (pd.DataFrame): Features dataset.
        y (pd.Series): Target dataset.
        config (dict): Configuration dictionary containing:
            - train_size (float): Proportion of data for training.
            - stratify (str or None): Column name for stratification.
            - validation_size (float or None): Proportion of training data for validation.
        categorical_features (list or None): List of categorical feature names.

    Returns:
        dict: A dictionary containing the train, validation, and test sets as DataFrames/Series.
    """
    # For DecisionTree, we need to encode categorical features
    X_encoded = X.copy()

    # Convert categorical features to numeric using one-hot encoding
    if categorical_features:
        for col in categorical_features:
            if col in X_encoded.columns:
                # Get dummies for each categorical feature and prefix with column name
                dummies = pd.get_dummies(X_encoded[col], prefix=col, drop_first=False)
                # Drop the original column and join the dummies
                X_encoded = X_encoded.drop(col, axis=1).join(dummies)

    # Split train and test
    stratify_col = X[config['stratify']] if config.get('stratify') else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y,
        train_size=config['train_size'],
        random_state=42,
        stratify=stratify_col
    )

    # Split validation set if specified
    val_percentage = config.get('validation_size')
    if val_percentage and val_percentage > 0:
        stratify_col_train = X_train[config['stratify']] if config.get('stratify') and config['stratify'] in X_train else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=val_percentage,
            random_state=42,
            stratify=stratify_col_train
        )
    else:
        X_val, y_val = None, None

    if X_val is not None:
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    else:
        return {
            'train': (X_train, y_train),
            'test': (X_test, y_test)
        }


# train model
def train_regtree_model(data_dict, config):
    """
    Train a regression tree model using the provided data and configuration.

    Args:
        data_dict (dict): Dictionary containing the train, validation, and test sets.
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        DecisionTreeRegressor: The trained regression tree model.
    """
    # Init model
    params = config['hyperparameters'].copy()

    # Set up the model
    model = DecisionTreeRegressor(**params)
    model.fit(data_dict['train'][0], data_dict['train'][1])

    return model


# metrics
def calculate_metrics(y, y_pred, best_percent=1.0):
    # Create a DataFrame to hold y, y_pred, and MAPE
    df = pd.DataFrame({'y': y, 'y_pred': y_pred})

    # Calculate APE
    df['perc_error'] = 1 - df['y_pred'] / df['y']
    df['ape'] = np.abs(df['perc_error'])

    # Determine the threshold ape to filter the best_percent data
    threshold_ape = df['ape'].quantile(best_percent)

    # Filter the best_percent of the data
    df_best = df[df['ape'] <= threshold_ape]

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(df_best['y'], df_best['y_pred']))
    mape = df_best['ape'].mean()
    meape = df_best['ape'].median()
    r2 = r2_score(df_best['y'], df_best['y_pred'])
    n_size = int(len(y))
    worst_negative_error = df['perc_error'].min()
    worst_positive_error = df['perc_error'].max()

    return pd.Series({
        "mape": mape,
        "meape": meape,
        "rmse": rmse,
        "r2": r2,
        "n_size": n_size,
        "worst_negative_error": worst_negative_error,
        "worst_positive_error": worst_positive_error
    })


# plots
def plot_y(y):
    with plt.style.context(style='tableau-colorblind10'):
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        sns.histplot(y, kde=True, ax=ax[0])
        ax[0].set_title('Histogram of y transformed')
        ax[0].set_xlabel('y')
        ax[0].set_ylabel('Frequency')

        sns.boxplot(y, ax=ax[1])
        ax[1].set_title('Boxplot of y transformed')
        ax[1].set_xlabel('y')

        plt.tight_layout()
    plt.close(fig)
    return fig


def plot_error_histogram(y, y_pred):
    # create error
    error = 1 - y_pred / y

    # plot
    with plt.style.context(style='tableau-colorblind10'):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(
            error,
            kde=False,
            bins=50,
            label='Percentual Error'
        )

        # labels
        plt.xlabel('Percentual Error')
        plt.ylabel('Frequency')
        plt.title('Error Histogram')

        # x ticks in percentage
        plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(1, decimals=0))
        plt.tight_layout()

    # save
    plt.close(fig)
    return fig


def plot_results_table(tbl):
    # roand to decimals
    tbl = tbl.round(4)

    # plot
    with plt.style.context(style='tableau-colorblind10'):
        fig, ax = plt.subplots()

        # remove axis
        ax.axis('off')
        ax.axis('tight')

        # plot table
        ax.table(
            cellText=tbl.values,
            colLabels=tbl.columns,
            rowLabels=tbl.index,
            loc='center'
        )
        plt.tight_layout()
    plt.close(fig)
    return fig


def plot_predicted_vs_real(y, y_pred):
    # residuals
    residuals = y - y_pred

    # plot
    with plt.style.context(style='tableau-colorblind10'):
        # y pred vs y
        fig, ax = plt.subplots(1, 3, figsize=(14, 6))
        sns.scatterplot(x=y, y=y_pred, ax=ax[0])
        ax[0].set_title('Predicted vs Real')
        ax[0].set_xlabel('Real')
        ax[0].set_ylabel('Predicted')
        # add identity
        ax[0].plot([y.min(), y.max()], [y.min(), y.max()], color='gray', lw=2, linestyle='--')
        # add regression line
        sns.regplot(y=y, x=y_pred, ax=ax[0], scatter=False, color='red')
        # x ticks 90 degrees
        ax[0].tick_params(axis='x', rotation=90)

        # residuals
        sns.scatterplot(x=y, y=residuals, ax=ax[1])
        ax[1].set_title('Residuals vs Real')
        ax[1].set_xlabel('Real')
        ax[1].set_ylabel('Residuals')
        # add zero line
        ax[1].axhline(0, color='gray', lw=2, linestyle='--')
        # x ticks 90 degrees
        ax[1].tick_params(axis='x', rotation=90)

        # percentual error
        sns.scatterplot(x=y, y=residuals / y, ax=ax[2])
        ax[2].set_title('Percentual Error vs Real')
        ax[2].set_xlabel('Real')
        ax[2].set_ylabel('Percentual Error')
        # add zero line
        ax[2].axhline(0, color='gray', lw=2, linestyle='--')
        # y label in percentage
        ax[2].yaxis.set_major_formatter(ticker.PercentFormatter(1, decimals=0))
        # x ticks 90 degrees
        ax[2].tick_params(axis='x', rotation=90)

        plt.tight_layout()
    plt.close(fig)
    return fig


def plot_tree_path(model, feature_names):
    """
    Plot the decision tree.

    Args:
        model (DecisionTreeRegressor): Trained decision tree model.
        feature_names (list): List of feature names.

    Returns:
        matplotlib.figure.Figure: Figure object containing the plot.
    """
    with plt.style.context(style='tableau-colorblind10'):
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(
            model,
            feature_names=feature_names,
            filled=True,
            fontsize=10,
            ax=ax,
            rounded=True,
            precision=2
        )
        plt.title('Decision Tree')
        plt.tight_layout()
    plt.close(fig)
    return fig


def create_mlflow_dicts(config_model, tbl, data_dict):
    """
    Create dictionaries for parameters, metrics, and tags to upload to MLFlow.

    Args:
        config_model (dict): Configuration dictionary for the model.
        tbl (pd.DataFrame): Table containing metrics for train, validation, and test sets.
        data_dict (dict): Dictionary containing Reg Tree Pool objects for train, validation, and test sets.

    Returns:
        tuple: A tuple containing three dictionaries (parameters, metrics, tags).
    """
    # s0: transpose tbl
    tbl = tbl.T.copy() 

    # s1: parameters
    dict_params = {
        f"hyperparameters__{k}": v
        for k, v in config_model['hyperparameters'].items()
    }

    dict_params.update({
        f"data__{k}": v
        for k, v in config_model['data'].items()
        if k not in ['split_type']
    })

    # s2: create metrics
    dict_metrics = {
        f"test__{k}": v
        for k, v in tbl.query('index == "test"').to_dict(orient='records')[0].items()
        if k not in ['worst_negative_error', 'worst_positive_error', 'r2', 'n_size']
    }
    dict_metrics.update({
        f"train__{k}": v
        for k, v in tbl.query('index == "train"').to_dict(orient='records')[0].items()
        if k not in ['worst_negative_error', 'worst_positive_error', 'meape', 'r2']
    })
    dict_metrics.update({
        "n_features": data_dict['train'][0].shape[1]
    })

    # s3: tags
    dict_tags = {
        'model': config_model['model_name'],
        'purpose': config_model['purpose'],
        'target': list(config_model['target'].keys())[0],
        'split_type': config_model['data']['split_type'],
        'objective_variable_transformation': config_model['target'].get(
            list(config_model['target'].keys())[0]
        ),
    }

    return dict_params, dict_metrics, dict_tags


def save_to_mlflow(
        dict_params, dict_metrics, dict_tags, figs, model, X_sample, y_sample
        ):
    """
    Save model parameters, metrics, tags, plots, and the model itself to MLFlow.

    Args:
        dict_params (dict): Model parameters to log.
        dict_metrics (dict): Model metrics to log.
        dict_tags (dict): Tags to set in MLFlow.
        figs (dict): Dictionary of matplotlib figures to log as artifacts.
        model (Regressor): Trained Reg Tree model.
        X_sample (pd.DataFrame): Sample of features for signature inference.
        y_sample (pd.Series): Sample of target values for signature inference.
    """
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(dict_params)

        # Log metrics
        mlflow.log_metrics(dict_metrics)

        # Set tags
        mlflow.set_tags(dict_tags)

        # Log figures
        for fig_name, fig in figs.items():
            mlflow.log_figure(fig, f"{fig_name}.png")

        # Log the model
        signature = infer_signature(X_sample, y_sample)
        mlflow.sklearn.log_model(
            model,
            artifact_path="models",
            signature=signature,
        )


# main
# def main():
# %% S0: Load configurations
logger.info("Loading configurations...")
config_model, recaster_mappers, mlflow_config = get_configs('config_regtree.yaml')

# %% S1: Read data
logger.info("Reading data...")
X, y = read_data(FILE_PROPERTIES, config_model)
logger.info(f"Data read successfully. X shape: {X.shape}, y shape: {y.shape}")

# %% S2: Feature engineering
logger.info("Applying feature engineering...")
transformer = FeatureTransformer(config_model, recaster_mappers)
X_transformed, y_transformed = transformer.fit_transform(X, y)
logger.info(f"Feature engineering completed. X shape: {X_transformed.shape}, y shape: {y_transformed.shape}")

# %% S3: Train-test split
logger.info("Splitting data into train-test sets...")
pools = split_randomly_data(
    X_transformed,
    y_transformed,
    config_model['data'],
    categorical_features=transformer.get_transformers()['categorical']
)
# delete X, y to free memory
del X, y, X_transformed

# %% S4: Train Reg Tree model
logger.info("Training Reg Tree model...")
model = train_regtree_model(pools, config_model)

# %% S5: Calculate metrics
logger.info("Calculating metrics...")
y_obs_test = get_target_value(
    pools['test'][1].to_numpy(),
    transformer
)
y_pred_test = get_predictions(model, pools['test'][0], transformer)
tbl_results = pd.DataFrame({
    k: calculate_metrics(
        y=get_target_value(v[1].to_numpy(), transformer),
        y_pred=get_predictions(model, v[0], transformer),
        best_percent=config_model['data'].get('best_percentage')
    )
    for k, v in pools.items()
})
logger.info("Test MAPE: {:.4f}".format(tbl_results.loc['mape', 'test']))

# %% S6: Plot results
logger.info("Plotting results...")
figs = {
    "y_transformed": plot_y(y_transformed),
    "error_histogram": plot_error_histogram(y_obs_test, y_pred_test),
    "results_table": plot_results_table(tbl_results),
    "predicted_vs_real": plot_predicted_vs_real(y_obs_test, y_pred_test),
    "tree_path": plot_tree_path(model, pools['train'][0].columns),
}

# %% S7: Save results to MLFlow
logger.info("Saving results to MLFlow")
mlflow.set_tracking_uri(f"http://{mlflow_config['host']}:{mlflow_config['port']}")
mlflow.set_experiment(mlflow_config['experiment_name'])
dict_params, dict_metrics, dict_tags = create_mlflow_dicts(config_model, tbl_results, pools)

save_to_mlflow(
    dict_params, dict_metrics, dict_tags, figs,
    model, pools['train'][0].sample(100), y_pred_test
)

logger.info("All done!")


# if __name__ == '__main__':
#     main()
