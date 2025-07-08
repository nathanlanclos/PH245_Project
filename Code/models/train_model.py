import numpy as np
import pandas as pd
import optuna
import json
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm
from xgboost import XGBRegressor, XGBClassifier
import matplotlib.pyplot as plt

class SolubilityPredictor:
    def __init__(self, continuous_column='solubility', binary_column='binary_solubility', k_folds=5, seed = 42):
        self.continuous_column = continuous_column
        self.binary_column = binary_column
        self.k_folds = k_folds
        self.scalar_features = []
        self.models = {}
        self.seed = seed

    @staticmethod
    def try_json_loads(val):
        try:
            loaded = json.loads(val)
            if isinstance(loaded, list):
                return loaded
        except (TypeError, json.JSONDecodeError):
            pass
        return val  # return original if not a JSON list
    
    def preprocess_data(self, input_data):
        if isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        elif input_data.endswith('.csv'):
            df = pd.read_csv(input_data)
            # Apply json.loads only to columns that contain JSON strings for lists
            for col in df.columns:
                if df[col].dtype == object:  # Only check object/string columns
                    sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                    if isinstance(sample_val, str) and sample_val.strip().startswith("["):
                        df[col] = df[col].apply(SolubilityPredictor.try_json_loads)
                        

        elif input_data.endswith('.pkl'):
            df = pd.read_pickle(input_data)
            
        else:
            raise ValueError("Input must be a DataFrame, .csv, or .pkl file.")
        new_frames   = []          # store expanded list columns
        cols_to_drop = []          # original columns to remove

        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).any():
                cols_to_drop.append(col)
                # correct length calculation
                max_len = df[col].dropna().apply(lambda x: len(np.array(x).flatten())).max()

                expanded = pd.DataFrame(df[col].apply(lambda x: np.array(x).flatten()).tolist(), index=df.index)
                expanded.columns = [f"{col}_{i+1}" for i in range(max_len)]

                new_frames.append(expanded)
        # -------------------------------------------------
        if new_frames:
            df = pd.concat([df] + new_frames, axis=1)

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        df = df.select_dtypes(include=["number"]).copy()
        return df

    def split_data(self, train_data, test_data = None):
        if test_data is None:
            df = self.preprocess_data(train_data)
            # ------------------------------------------------------------
            # 0)  Make three splits up front
            # ------------------------------------------------------------
            y_full_bin = df[self.binary_column].values
            y_full_cont = df[self.continuous_column].values
            X_full  = df.drop(columns=[self.continuous_column, self.binary_column])

            # First: hold-out TEST
            self.X_train, self.X_test, self.y_train_bin, self.y_test_bin, self.y_train_cont, self.y_test_cont = train_test_split(
                X_full, y_full_bin, y_full_cont,
                test_size=0.20, stratify=y_full_bin, random_state=self.seed
            )
        else:
            train_df = self.preprocess_data(train_data)
            test_df = self.preprocess_data(test_data)

            self.y_train_bin = train_df[self.binary_column].values
            self.y_train_cont = train_df[self.continuous_column].values
            self.X_train  = train_df.drop(columns=[self.continuous_column, self.binary_column])

            self.y_test_bin = test_df[self.binary_column].values
            self.y_test_cont = test_df[self.continuous_column].values
            self.X_test  = test_df.drop(columns=[self.continuous_column, self.binary_column])
        self.scalar_features = [c for c in self.X_train.columns if 'embedding' not in c.lower()]
        self.scaler = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.scalar_features)   # ← only these get scaled
            ],
            remainder='passthrough'      # embeddings & any other cols stay untouched
        )
        self.scaler = StandardScaler()



    def cross_validate_xgboost(
            self,
            model_type=XGBRegressor,
            best_params=None,
            n_optuna_trials=20):
        """
        Train the final model with an internal validation split for early stopping.
        The *test* set is kept blind.
        """
        # ----------------------- choose y & metric ------------------------- #
        if model_type is XGBRegressor:
            y_train, y_test = self.y_train_cont, self.y_test_cont
            score_method = r2_score
        elif model_type is XGBClassifier:
            y_train, y_test = self.y_train_bin, self.y_test_bin
            score_method = accuracy_score
        else:
            raise ValueError("model_type must be XGBRegressor or XGBClassifier")

        # -------------------- get / search best params -------------------- #
        if best_params is None and (model_type is XGBRegressor or model_type is XGBClassifier):
            best_params = self.optimize_hyperparameters(
                model_type=model_type,
                n_trials=n_optuna_trials
            )
            print(f"Best parameters found: {best_params}")

        # ------------------------------------------------------------------ #
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled  = self.scaler.transform(self.X_test)

        # ---------------------- fit with early stop ----------------------- #
        model = model_type(**best_params)
        model.fit(X_train_scaled, y_train)

        # -------------------------- evaluation ---------------------------- #
        train_score = score_method(y_train, model.predict(X_train_scaled))
        test_score  = score_method(y_test,  model.predict(X_test_scaled))
        print(f"TRAIN {score_method.__name__}: {train_score:.4f}")
        print(f"TEST  {score_method.__name__}: {test_score:.4f}")
        #self.plot_predicted_vs_real(y_train, model.predict(X_train_scaled), train_score)
        #self.plot_predicted_vs_real(y_test, model.predict(X_test_scaled), test_score)
        return model


    # ----------------------------------------------------------------------
    def optimize_hyperparameters(
            self,
            model_type=XGBRegressor,
            n_trials=20):
        """
        Optuna study with per-fold early stopping.
        """
        # --------------- branch-specific settings ------------------------- #
        if model_type is XGBClassifier:
            obj, eval_met = "binary:logistic", "error"
            score_method  = accuracy_score
            target        = self.y_train_bin
            cv            = StratifiedKFold(n_splits=self.k_folds,
                                            shuffle=True, random_state=self.seed)
        elif model_type is XGBRegressor:
            obj, eval_met = "reg:squarederror", "r2"
            score_method  = r2_score
            target        = self.y_train_cont
            cv            = KFold(n_splits=self.k_folds,
                                shuffle=True, random_state=self.seed)
        else:
            raise ValueError("model_type must be XGBRegressor or XGBClassifier")

        def objective(trial):
            params = dict(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                learning_rate=trial.suggest_float('learning_rate', 1e-2, 1e-1, log=True),
                max_depth=trial.suggest_int('max_depth', 3, 8),
                min_child_weight=trial.suggest_float('min_child_weight', 1.0, 10.0),
                gamma=trial.suggest_float('gamma', 0.0, 0.5),
                subsample=trial.suggest_float('subsample', 0.5, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
                reg_alpha=trial.suggest_float('alpha', 0.0, 2.0),
                reg_lambda=trial.suggest_float('lambda', 0.0, 2.0),
                objective=obj,
                random_state=self.seed,
            )

            fold_scores = []
            for tr_idx, te_idx in cv.split(self.X_train,
                                        target if model_type is XGBClassifier else None):
                # split train fold into train/val for early stopping
                X_tr_full = self.X_train.iloc[tr_idx]
                y_tr_full = target[tr_idx]

                X_te = self.X_train.iloc[te_idx]

                X_tr = self.scaler.fit_transform(X_tr_full)
                X_te  = self.scaler.transform(X_te)

                model = model_type(**params)
                model.fit(X_tr, y_tr_full, verbose=False)

                preds = model.predict(X_te)
                fold_scores.append(score_method(target[te_idx], preds))

            return float(np.mean(fold_scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        return best_params
    
    def plot_predicted_vs_real(self, y_true, y_pred, r2):
        """
        Plot predicted solubility values against real values for the test set, including line of best fit.
        """
        if y_true is None or y_pred is None:
            raise ValueError("y_true and y_pred must be provided for plotting.")
        plt.figure(figsize=(7,7))
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolor='k', label='Data')
        # Ideal line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        # Line of best fit
        fit = np.polyfit(y_true, y_pred, 1)
        fit_fn = np.poly1d(fit)
        plt.plot([min_val, max_val], fit_fn([min_val, max_val]), 'b-', lw=2, label='Best Fit')
        plt.xlabel('Real Solubility')
        plt.ylabel('Predicted Solubility')
        plt.title(f'Predicted vs Real Solubility (R² = {r2:.2f})')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def train_linear_regressor(self):
        """
        Train and evaluate a Linear Regression model for solubility regression.
        """
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        model = LinearRegression()
        model.fit(X_train_scaled, self.y_train_cont)
        train_score = r2_score(self.y_train_cont, model.predict(X_train_scaled))
        test_score = r2_score(self.y_test_cont, model.predict(X_test_scaled))
        print(f"LinearRegression TRAIN R2: {train_score:.4f}")
        print(f"LinearRegression TEST  R2: {test_score:.4f}")
        return model

    def train_logistic_classifier(self):
        """
        Train and evaluate a Logistic Regression model for binary solubility classification.
        """
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        model = LogisticRegression(max_iter=1000, random_state=self.seed)
        model.fit(X_train_scaled, self.y_train_bin)
        train_score = accuracy_score(self.y_train_bin, model.predict(X_train_scaled))
        test_score = accuracy_score(self.y_test_bin, model.predict(X_test_scaled))
        print(f"LogisticRegression TRAIN Accuracy: {train_score:.4f}")
        print(f"LogisticRegression TEST  Accuracy: {test_score:.4f}")
        print("Classification report (test):\n", classification_report(self.y_test_bin, model.predict(X_test_scaled)))
        return model