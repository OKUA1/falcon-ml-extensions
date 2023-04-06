from falcon.tabular.pipelines import SimpleTabularPipeline
from falcon.tabular.learners import OptunaLearner, PlainLearner
from falcon_ml_xgboost.model import FalconXGBoostClassifier, FalconXGBoostRegressor

xg_clf = {
    "XGBOOST::OptunaLearner": {
        "pipeline": SimpleTabularPipeline,
        "extra_pipeline_options": {
            "learner": OptunaLearner,
            "learner_kwargs": {"model_class": FalconXGBoostClassifier},
        },
    }
}

xg_regr = {
    "XGBOOST::OptunaLearner": {
        "pipeline": SimpleTabularPipeline,
        "extra_pipeline_options": {
            "learner": OptunaLearner,
            "learner_kwargs": {"model_class": FalconXGBoostRegressor},
        },
    }
}

xg_clf_plain = {
    "XGBOOST::PlainLearner": {
        "pipeline": SimpleTabularPipeline,
        "extra_pipeline_options": {
            "learner": PlainLearner,
            "learner_kwargs": {"model_class": FalconXGBoostClassifier},
        },
    }
}


xg_regr_plain = {
    "XGBOOST::PlainLearner": {
        "pipeline": SimpleTabularPipeline,
        "extra_pipeline_options": {
            "learner": PlainLearner,
            "learner_kwargs": {"model_class": FalconXGBoostRegressor},
        },
    }
}


def self_register():
    from falcon.task_configurations import TaskConfigurationRegistry as _registry

    _registry.register_configurations("tabular_regression", xg_regr)
    _registry.register_configurations("tabular_classification", xg_clf)
    _registry.register_configurations("tabular_regression", xg_regr_plain)
    _registry.register_configurations("tabular_classification", xg_clf_plain)
