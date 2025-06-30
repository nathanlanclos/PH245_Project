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
            #COMMENT OUT LATER
            df = df.drop(["aac"], axis = 1)
            df = df.drop(["blosum62_embedding", "flexibility"], axis = 1)
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
        #self.scaler = StandardScaler()

    def cross_validate_xgboost(self, model_type=XGBRegressor, best_params=None, n_optuna_trials=20, early_stopping=True):
        if best_params is None:
            best_params = self.optimize_hyperparameters(model_type, n_optuna_trials, early_stopping)

        if model_type is XGBRegressor:
            y_train = self.y_train_cont
            y_test = self.y_test_cont
            score_method = r2_score
        elif model_type is XGBClassifier:
            y_train = self.y_train_bin
            y_test = self.y_test_bin
            score_method = accuracy_score
        else: 
            print("Invalid Model type. Please choose XGBRegressor or XGBClassifier")
            return

        # Scale full training and test sets
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        xgb_model = model_type(**best_params, early_stopping_rounds = 15)


        # Split off validation set for early stopping (only from training set)
        if early_stopping:
            X_train_core, X_val, y_train_core, y_val = train_test_split(
                X_train_scaled, y_train, test_size=0.1, random_state=self.seed
            )
            xgb_model.fit(
                X_train_core,
                y_train_core,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            xgb_model.fit(X_train_scaled, y_train)

            

        train_score = score_method(y_train, xgb_model.predict(X_train_scaled))
        test_score = score_method(y_test, xgb_model.predict(X_test_scaled))

        print(f"Final (TRAIN) Score for Model: {train_score:.4f}")
        print(f"Final (TEST) Score for Model: {test_score:.4f}")
        return xgb_model

    
    def optimize_hyperparameters(self, model_type = XGBRegressor, n_trials = 20, early_stopping = True):
        results = []
        if model_type is XGBClassifier:
            obj = "binary:logistic"
            eval_met = "logloss"
            score_method = accuracy_score
            model_name = "XGBClassifier"
            target = self.y_train_bin
            cv = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
        elif model_type is XGBRegressor:
            obj = "reg:squarederror"
            eval_met = "rmse"
            score_method = r2_score
            model_name = "XGBRegressor"
            target = self.y_train_cont
            cv = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
        else:
            print("INVALID MODEL TYPE")
            return
        
        def objective(trial):
            params = dict(
                n_estimators     = (300 if early_stopping else trial.suggest_int('n_estimators', 50, 200)),
                learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                max_depth        = trial.suggest_int ('max_depth', 3, 8),
                min_child_weight = trial.suggest_float('min_child_weight', 1.0, 10.0),
                gamma            = trial.suggest_float('gamma', 0.0, 0.5),
                subsample        = trial.suggest_float('subsample', 0.5, 1.0),
                colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0),
                reg_alpha        = trial.suggest_float('reg_alpha', 0.0, 2.0),
                reg_lambda       = trial.suggest_float('reg_lambda', 0.0, 2.0),
                objective        = obj,
                eval_metric      = eval_met,
                random_state     = self.seed,
            )

            fold_scores = []
            for tr_idx, te_idx in cv.split(self.X_train, target if model_type is XGBClassifier else None):
                scaler = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), self.scalar_features)   # ← only these get scaled
                    ],
                    remainder='passthrough'      # embeddings & any other cols stay untouched
                )
                X_tr = scaler.fit_transform(self.X_train.iloc[tr_idx])
                X_te = scaler.transform    (self.X_train.iloc[te_idx])

                model = model_type(**params)
                model.fit(X_tr, target[tr_idx])
                pred = model.predict(X_te)
                fold_scores.append(score_method(target[te_idx], pred))

            score = np.mean(fold_scores)
            results.append({**params, 'Score': score})
            return score
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"Code\models\optuna_results_{model_name}.csv", index=False)
        best_params = study.best_params
        return best_params