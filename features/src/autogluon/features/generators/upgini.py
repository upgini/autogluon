import logging
from typing import Optional
from pandas import DataFrame, Series

from autogluon.features.generators import PipelineFeatureGenerator

try:
    from upgini.features_enricher import FeaturesEnricher
except ImportError:
    FeaturesEnricher = None


class UpginiPipelineFeatureGenerator(PipelineFeatureGenerator):
    """
    Feature generator that uses Upgini to enrich features with external data sources.

    Parameters
    ----------
    pre_generators : list, optional
        List of feature generators to apply before main processing
    generators : list, optional
        List of main feature generators
    post_generators : list, optional
        List of feature generators to apply after main processing
    pre_drop_useless : bool, default True
        Whether to drop useless features before processing
    pre_enforce_types : bool, default True
        Whether to enforce types before processing
    reset_index : bool, default True
        Whether to reset index
    post_drop_duplicates : bool, default True
        Whether to drop duplicate features after processing
    verbosity : int, default 3
        Verbosity level for logging
    api_key : str, optional
        Upgini API key for feature enrichment
    country_code : str, optional
        Country code for Upgini search
    search_keys : dict, optional
        Dictionary mapping column names to Upgini SearchKey objects
    eval_set : list, optional
        List of tuples (X_eval, y_eval) for validation sets
    **kwargs
        Additional keyword arguments passed to parent class
    """

    def __init__(
        self,
        pre_generators=None,
        generators=None,
        post_generators=None,
        pre_drop_useless=True,
        pre_enforce_types=True,
        reset_index=True,
        post_drop_duplicates=True,
        verbosity=3,
        api_key: Optional[str] = None,
        country_code: Optional[str] = None,
        search_keys=None,
        eval_set=None,
        **kwargs,
    ):
        if api_key is not None and FeaturesEnricher is None:
            raise ImportError("upgini is not installed. Please install it with: pip install 'autogluon.features[upgini]'")
        super().__init__(
            pre_generators=[],
            generators=generators,
            post_generators=[],
            pre_drop_useless=False,
            reset_index=False,
        )
        self.features_enricher = None
        self.api_key = api_key
        self.country_code = country_code
        self.search_keys = search_keys
        self.eval_set = eval_set

    def _fit_transform(self, X: DataFrame, y: Series = None, **kwargs):
        if self.api_key is not None and y is not None:
            return self._fit_transform_upgini(X=X, y=y, **kwargs)
        else:
            self._log(logging.WARNING, f"API_KEY: {'missing' if self.api_key is None else 'present'}")
            self._log(logging.WARNING, f"y: {'missing' if y is None else 'present'}")
            self._log(logging.WARNING, "No API key specified for Upgini or y is None, just fitting internal generators")
            return super()._fit_transform(X=X, y=y, **kwargs)

    def _fit_transform_upgini(self, X: DataFrame, y: Series, **kwargs):
        self._log(logging.INFO, "Fitting generators on train set...")
        print(X.columns.tolist())
        X_train, type_group_map_special = super()._fit_transform(X=X, y=y, **kwargs)

        eval_transformed = None
        if self.eval_set is not None:
            eval_transformed = []
            for i, eval_set in enumerate(self.eval_set):
                self._log(logging.INFO, f"Transforming generators on eval set {i}...")
                y_eval = eval_set[1]
                X_eval = super()._transform(X=eval_set[0])
                if y_eval is not None:
                    X_eval.drop(columns=y_eval.name, errors="ignore", inplace=True)
                eval_transformed.append((X_eval, y_eval))

        self._log(logging.INFO, "Fitting FeaturesEnricher...")
        if FeaturesEnricher is None:
            raise ImportError("upgini is not installed")
        enricher = FeaturesEnricher(
            search_keys=self.search_keys,
            country_code=self.country_code,
            api_key=self.api_key,
        )

        X_out = enricher.fit_transform(X_train, y, eval_set=eval_transformed, calculate_metrics=False)
        type_group_map_special = {k: [c for c in v if c in X_out] for k, v in type_group_map_special.items()}

        print(X_out.columns.tolist())

        self.features_enricher = enricher

        return X_out, type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        if self.api_key is None:
            self._log(logging.WARNING, "No API key specified for Upgini, just transforming internal generators")
            return super()._transform(X)
        if self.features_enricher is None:
            raise ValueError("FeaturesEnricher is not fitted")

        X_out = super()._transform(X)
        return self.features_enricher.transform(X_out)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return {}
