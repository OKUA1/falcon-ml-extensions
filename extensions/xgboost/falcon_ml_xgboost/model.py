from numpy import typing as npt
from typing import Optional, Union, Callable, Dict, Any
from skl2onnx.common.data_types import FloatTensorType
from onnxmltools import convert_xgboost
from skl2onnx import to_onnx, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
    calculate_linear_regressor_output_shapes,
)

import xgboost
import numpy as np
import optuna
from falcon.abstract.model import Model
from falcon.abstract.onnx_convertible import ONNXConvertible
from falcon.abstract.optuna import OptunaMixin
from falcon.serialization import SerializedModelRepr
from falcon.config import ONNX_OPSET_VERSION, ML_ONNX_OPSET_VERSION


class _XGBoostBase(Model, ONNXConvertible, OptunaMixin):
    def __init__(
        self,
        verbosity: int = 0,
        objective: Optional[str] = None,
        tree_method: str = "auto",
        booster: str = "dart",
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        *args,
        **kwargs,
    ):

        params = kwargs
        params["verbosity"] = verbosity
        params["objective"] = objective
        params["tree_method"] = tree_method
        params["booster"] = booster
        params["reg_lambda"] = reg_lambda
        params["reg_alpha"] = reg_alpha
        self.params = params
        self.objective = "multi:softmax"

    def predict(self, X: npt.NDArray, *args: Any, **kwargs: Any) -> npt.NDArray:
        preds = self.bst.predict(X)
        return preds

    def to_onnx(self) -> SerializedModelRepr:
        """
        Serializes the model to onnx.

        Returns
        -------
        SerializedModelRepr
        """
        initial_type = [("model_input", FloatTensorType(self._shape))]
        # options = self._get_onnx_options()
        onnx_model = to_onnx(
            self.bst,
            initial_types=initial_type,
            target_opset={"": ONNX_OPSET_VERSION, "ai.onnx.ml": ML_ONNX_OPSET_VERSION},
            # options=options,
        )
        n_inputs = len(onnx_model.graph.input)
        n_outputs = len(onnx_model.graph.output)

        return SerializedModelRepr(
            onnx_model,
            n_inputs,
            n_outputs,
            ["FLOAT32"],
            [self._shape],
        )


class FalconXGBoostClassifier(_XGBoostBase):
    def __init__(
        self,
        verbosity: int = 0,
        tree_method: str = "auto",
        booster: str = "dart",
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(
            verbosity=verbosity,
            objective="multi:softmax",
            tree_method=tree_method,
            booster=booster,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            **kwargs,
        )

    def fit(self, X: npt.NDArray, y: npt.NDArray, *args: Any, **kwargs: Any) -> None:
        self._shape = [None, *X.shape[1:]]
        self.params["num_class"] = len(np.unique(y))
        self.bst = xgboost.XGBClassifier(**self.params)
        self.bst.fit(X, y)

    @classmethod
    def get_search_space(cls, X: npt.NDArray, y: npt.NDArray) -> Callable:
        obj_ = "multi:softmax"

        def _objective_fn(trial, X, Xt, y, yt):
            return _objective(trial, X, Xt, y, yt, obj_)

        return _objective_fn


class FalconXGBoostRegressor(_XGBoostBase):
    def __init__(
        self,
        verbosity: int = 0,
        tree_method: str = "auto",
        booster: str = "dart",
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(
            verbosity=verbosity,
            objective="reg:squarederror",
            tree_method=tree_method,
            booster=booster,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            **kwargs,
        )

    @classmethod
    def get_search_space(cls, X: npt.NDArray, y: npt.NDArray) -> Callable:
        def _objective_fn(trial, X, Xt, y, yt):
            return _objective(trial, X, Xt, y, yt, "reg:squarederror")

        return _objective_fn

    def fit(self, X: npt.NDArray, y: npt.NDArray, *args: Any, **kwargs: Any) -> None:
        self._shape = [None, *X.shape[1:]]
        self.bst = xgboost.XGBRegressor(**self.params)
        self.bst.fit(X, y)


def _objective(trial, X, Xt, y, yt, _objective):
    with xgboost.config_context(verbosity=0):
        n_targets = len(np.unique(y))

        param = {
            "verbosity": 0,
            "objective": _objective,
            "tree_method": "auto",
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
            "n_estimators": trial.suggest_int("n_estimators", 10, 500, step=10),
        }

        if _objective == "multi:softmax":
            param["num_class"] = n_targets

        if param["booster"] in ["gbtree", "dart"]:
            param["max_depth"] = trial.suggest_int("max_depth", 2, 10, step=2)
            param["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 10)
            param["eta"] = trial.suggest_float("eta", 0.01, 1.0)  # lr
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical(
                "grow_policy", ["depthwise"]  # , "lossguide"]        
             )

        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical(
                "sample_type", ["uniform", "weighted"]
            )
            param["normalize_type"] = trial.suggest_categorical(
                "normalize_type", ["tree", "forest"]
            )
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        if _objective == "reg:squarederror":
            bst = xgboost.XGBRFRegressor(**param)
        else:
            bst = xgboost.XGBRFClassifier(**param)
        bst.fit(X, y)
        preds = bst.predict(Xt)
        return {"predictions": preds, "loss": None}


update_registered_converter(
    xgboost.XGBRegressor,
    "XGBoostXGBRegressor",
    calculate_linear_regressor_output_shapes,
    convert_xgboost,
)

update_registered_converter(
    xgboost.XGBClassifier,
    "XGBoostXGBClassifier",
    calculate_linear_classifier_output_shapes,
    convert_xgboost,
    options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
)
