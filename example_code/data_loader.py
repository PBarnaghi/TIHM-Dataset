"""
In this file, the two most important objects are the classes:

- TIHM: The data loading class for the tihm data.

- TIHMDataset: The pytorch dataset wrapping the TIHM class.

"""

from __future__ import annotations
import torch
import typing
import os
import logging
import warnings
import copy
import datetime as dt
import pandas as pd
import numpy as np
from sklearn import impute
from sklearn import preprocessing
from sklearn import base as skbase


TEST_START = "2019-06-15"  # The date to start the test set


class StandardGroupScaler(skbase.BaseEstimator, skbase.TransformerMixin):
    def __init__(self):
        """
        This class allows you to scale the data based on a group.

        When calling transform, if the group has not been seen
        in the fitting method, then the global statistics will
        be used to scale the data (global = across all groups).

        Where the mean or standard deviation are equal to :code:`NaN`,
        in any axis on any group, that particular value will be
        replaced with the global mean or standard deviation for that
        axis (global = across all groups). If the standard deviation
        is returned as :code:`0.0` then the global standard deviation
        and mean is used.

        """
        self.scalers = {}
        self.means_ = {}
        self.vars_ = {}
        self.global_scalar = None
        self.global_mean_ = None
        self.global_var_ = None
        self.scalars_fitted = False
        self.groups_fitted = []

    def fit(
        self,
        X: np.ndarray,
        y: typing.Union[np.ndarray, None] = None,
        groups: typing.Union[np.ndarray, None] = None,
    ) -> StandardGroupScaler:
        """
        Compute the mean and std to be used for later scaling.



        Arguments
        ---------

        - X: np.ndarray:
            The data used to compute the mean and standard deviation used for later
            scaling along the features axis. This should be of shape
            :code:`(n_samples, n_features)`.

        - y: typing.Union[np.ndarray, None], optional:
            Igorned.
            Defaults to :code:`None`.

        - groups: typing.Union[np.ndarray, None], optional:
            The groups to split the scaling by. This should be of shape
            :code:`(n_samples,)`.
            Defaults to :code:`None`.


        Returns
        --------

        - self: sku.StandardGroupScaler:
            The fitted scaler.


        """
        # if no groups are given then all points are
        # assumed to be from the same group
        if groups is None:
            logging.warning(
                "You are using the grouped version of StandardScaler, yet you have "
                "not passed any groups. Using sklearn.preprocessing.StandardScaler "
                "will be faster if you have no groups to use."
            )
            groups = np.ones((X.shape[0]))

        self.global_mean_ = np.nanmean(X, axis=0)
        self.global_var_ = np.nanvar(X, axis=0)

        # creating an instance of the sklearn StandardScaler
        # for each group
        groups_unique = np.unique(groups)
        for group_name in groups_unique:
            # get the data from that group
            mask = groups == group_name
            X_sub = X[mask]

            # calculating the statistics
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", r"Mean of empty slice.*")
                # calculating mean
                group_means = np.nanmean(X_sub, axis=0)
                warnings.filterwarnings(
                    "ignore", r"Degrees of freedom <= 0 for slice.*"
                )
                # calculating var
                group_vars = np.nanvar(X_sub, axis=0)

            # replace NaN with global statistics
            replace_with_global_mask = (
                np.isnan(group_means) | np.isnan(group_vars) | (group_vars == 0)
            )
            group_means[replace_with_global_mask] = self.global_mean_[
                replace_with_global_mask
            ]
            group_vars[replace_with_global_mask] = self.global_var_[
                replace_with_global_mask
            ]

            # saving group statistics
            self.means_[group_name] = group_means
            self.vars_[group_name] = group_vars

            self.groups_fitted.append(group_name)

        # flag to indicate the scalars have been fitted
        self.scalars_fitted = True

        return self

    def transform(
        self,
        X: np.ndarray,
        y: typing.Union[np.ndarray, None] = None,
        groups: typing.Union[np.ndarray, None] = None,
    ) -> np.ndarray:
        """
        Perform standardization by centering and scaling by group.


        Arguments
        ---------

        - X: np.ndarray:
            The data used to scale along the features axis. This should be of shape
            :code:`(n_samples, n_features)`.

        - y: typing.Union[np.ndarray, None], optional:
            Ignored.
            Defaults to :code:`None`.

        - groups: typing.Union[np.ndarray, None], optional:
            The groups to split the scaling by. This should be of shape
            :code:`(n_samples,)`.
            Defaults to :code:`None`.



        Returns
        --------

        - X_norm: np.ndarray:
            The transformed version of :code:`X`.


        """

        X_norm = copy.deepcopy(X)

        # if no groups are given then all points are
        # assumed to be from the same group
        if groups is None:
            logging.warning(
                "You are using the grouped version of StandardScaler, yet you have "
                "not passed any groups. Using sklearn.preprocessing.StandardScaler "
                "will be faster if you have no groups to use."
            )
            groups = np.ones((X_norm.shape[0]))

        # transforming the data in each group
        groups_unique = np.unique(groups)
        for group_name in groups_unique:
            mask = groups == group_name
            try:
                X_norm[mask] = (X_norm[mask] - self.means_[group_name]) / np.sqrt(
                    self.vars_[group_name]
                )
            except KeyError:
                X_norm[mask] = (X_norm[mask] - self.global_mean_) / np.sqrt(
                    self.global_var_
                )

        return X_norm

    def fit_transform(
        self,
        X: np.ndarray,
        y: typing.Union[np.ndarray, None] = None,
        groups: typing.Union[np.ndarray, None] = None,
    ) -> np.ndarray:
        """
        Fit to data, then transform it. Fits transformer to X using the groups
        and returns a transformed version of X.



        Arguments
        ---------

        - X: np.ndarray:
            The data used to compute the mean and standard deviation used for later
            scaling along the features axis. This should be of shape
            :code:`(n_samples, n_features)`.

        - groups: typing.Union[np.ndarray, None], optional:
            The groups to split the scaling by. This should be of shape
            :code:`(n_samples,)`.
            Defaults to :code:`None`.

        - y: typing.Union[np.ndarray, None], optional:
            Igorned.
            Defaults to :code:`None`.



        Returns
        --------

        - self:
            The fitted scaler.


        """

        self.fit(X=X, groups=groups, y=y)
        return self.transform(X=X, groups=groups, y=y)


def make_input_roll(
    array: np.ndarray,
    sequence_length: int,
) -> np.ndarray:
    """
    This function will produce an array that is a rolled
    version of the original data sequence. The original
    sequence must be 2D.

    Examples
    ---------

    .. code-block::

        >>> make_input_roll(np.array([[1],[2],[3],[4],[5]]), sequence_length=2)
        array([[[1],
                [2]],

                [[2],
                [3]],

                [[3],
                [4]],

                [[4],
                [5]]]

    similarly:

    .. code-block::

        >>> make_input_roll(np.array([[1, 2],[3, 4],[5, 6]]), sequence_length=2)
        array([[[1,2],
                [3,4]],

                [[3,4],
                [5,6]]]


    Arguments
    ---------

    - array: numpy.ndarray:
        This is the array that you want transformed. Please use the shape (n_datapoints, n_features).

    - sequence_length: int:
        This is an integer that contains the length of each of the returned sequences.


    Returns
    ---------

    - output: ndarray:
        This is an array with the rolled data.

    """
    if type(sequence_length) != int:
        raise TypeError("Please ensure that sequence_length is an integer.")
    if type(array) not in [np.ndarray, list]:
        raise TypeError("Please ensure that array is an array or list.")

    array = np.array(array)

    if array.shape[0] < sequence_length:
        raise ValueError(
            "Please ensure that the input can be rolled "
            "by the specified sequence_length. Input size was "
            f"{array.shape} and the sequence_length was {sequence_length}."
        )

    output = array[
        np.lib.stride_tricks.sliding_window_view(
            np.arange(array.shape[0]), window_shape=sequence_length
        )
    ]

    return output


class TIHM:
    def __init__(
        self,
        root: str = "./",
    ):
        """
        The TIHM dataset
        as is here: https://github.com/PBarnaghi/TIHM1.5-Data.

        This class allows you to load the different csvs
        using the attributes of the class. If the data
        is not at the :code:`root` given, it will be
        downloaded.


        Examples
        ---------

        The following are examples for loading the data
        using this class.

        .. code-block::

            >>> dataset = TIHM(root='./data/')
            >>> activity_data = dataset.activity
            >>> all_data = dataset.data
            >>> len(dataset)
            2802


        Arguments
        ---------

        - root: str, optional:
            The file path to where the TIHM
            data files are stored.
            Defaults to :code:`'./'`.


        """

        self.root = root
        self.data_names = ["activity", "sleep", "physiology", "labels", "demographics"]

        if not self._check_exists():
            raise FileNotFoundError(
                "The files were not found in the root directory provided."
            )

        self._activity_raw = None
        self._activity_df = None
        self._activity_types = None

        self._sleep_raw = None
        self._sleep_df = None
        self._sleep_types = None

        self._physiology_raw = None
        self._physiology_df = None
        self._physiology_types = None

        self._data_df = None
        self._data_types = None

        self._target_raw = None
        self._target_df = None
        self._target_types = None

        self._demographic_raw = None
        self._demographic_df = None
        self._demographic_types = None

        return

    @property
    def activity_raw(self) -> pd.DataFrame:
        """
        The raw activity csv without any aggregation.
        """
        if self._activity_raw is None:
            data_name = "activity"
            data_path = os.path.join(self.root, f"{data_name.title()}.csv")
            # load the data
            self._activity_raw = pd.read_csv(data_path)
        return self._activity_raw

    @property
    def activity(self) -> pd.DataFrame:
        """
        The aggregated activity data.
        """
        if self._activity_df is None:
            self._activity_df = self.process_activity().sort_values(
                ["patient_id", "date"]
            )
        return self._activity_df

    @property
    def activity_types(self) -> typing.List[str]:
        """
        The names of the activity features that are aggregated.
        """
        if self._activity_types is None:
            self._activity_types = list(self.activity_raw["location_name"].unique())
        return self._activity_types

    @property
    def sleep_raw(self) -> pd.DataFrame:
        """
        The raw sleep csv without any aggregation.
        """
        if self._sleep_raw is None:
            data_name = "sleep"
            data_path = os.path.join(self.root, f"{data_name.title()}.csv")
            # load the data
            self._sleep_raw = pd.read_csv(data_path)
        return self._sleep_raw

    @property
    def sleep(self) -> pd.DataFrame:
        """
        The aggregated sleep data.
        """
        if self._sleep_df is None:
            self._sleep_df = self.process_sleep().sort_values(["patient_id", "date"])
        return self._sleep_df

    @property
    def sleep_types(self) -> typing.List[str]:
        """
        The names of the sleep features that are aggregated.
        """
        if self._sleep_types is None:
            self._sleep_types = ["heart_rate", "respiratory_rate"]
        return self._sleep_types

    @property
    def physiology_raw(self) -> pd.DataFrame:
        """
        The raw physiology csv without any aggregation.
        """
        if self._physiology_raw is None:
            data_name = "physiology"
            data_path = os.path.join(self.root, f"{data_name.title()}.csv")
            # load the data
            self._physiology_raw = pd.read_csv(data_path)
        return self._physiology_raw

    @property
    def physiology(self) -> pd.DataFrame:
        """
        The aggregated physiology data.
        """
        if self._physiology_df is None:
            self._physiology_df = self.process_physiology().sort_values(
                ["patient_id", "date"]
            )
        return self._physiology_df

    @property
    def physiology_types(self) -> typing.List[str]:
        """
        The names of the physiology features that are aggregated.
        """
        if self._physiology_types is None:
            self._physiology_types = list(self.physiology_raw["device_type"].unique())
        return self._physiology_types

    @property
    def data(self) -> pd.DataFrame:
        """
        The aggregated activity, sleep and physiology data.
        """
        if self._data_df is None:
            self._data_df = pd.merge(
                left=self.activity,
                right=self.sleep,
                how="outer",
                on=["patient_id", "date"],
            )
            self._data_df = pd.merge(
                left=self._data_df,
                right=self.physiology,
                how="outer",
                on=["patient_id", "date"],
            )
            self._data_df = self._data_df.sort_values(["patient_id", "date"])
        return self._data_df

    @property
    def data_types(self) -> typing.List[str]:
        """
        The names of the data features that are aggregated.
        """
        if self._data_types is None:
            self._data_types = (
                list(self.activity_types)
                + list(self.sleep_types)
                + list(self.physiology_types)
            )
        return self._data_types

    @property
    def target_raw(self) -> pd.DataFrame:
        """
        The raw label csv without any aggregation.
        """
        if self._target_raw is None:
            data_name = "labels"
            data_path = os.path.join(self.root, f"{data_name.title()}.csv")
            # load the data
            self._target_raw = pd.read_csv(data_path)
        return self._target_raw

    @property
    def target(self) -> pd.DataFrame:
        """
        The targets for the attribute :code:`.data`.
        """
        if self._target_df is None:
            self._target_df = self.process_target()
            data_labelled = pd.merge(
                left=self.data,
                right=self._target_df,
                how="left",
                on=["patient_id", "date"],
            )
            data_labelled.loc[:, self.target_types] = (
                # filling in unlabelled data with value given
                data_labelled.loc[:, self.target_types]
            )
            self._target_df = data_labelled[
                ["patient_id", "date"] + list(self.target_types)
            ].sort_values(["patient_id", "date"])
        return self._target_df

    @property
    def target_types(self) -> typing.List[str]:
        """
        The names of the targets that are aggregated.
        """
        if self._target_types is None:
            self._target_types = self.target_raw["type"].unique()  # getting label types
        return self._target_types

    @property
    def demographic_raw(self) -> pd.DataFrame:
        """
        The raw demographics csv data.
        """
        if self._demographic_raw is None:
            data_name = "demographics"
            data_path = os.path.join(self.root, f"{data_name.title()}.csv")
            # load the data
            self._demographic_raw = pd.read_csv(data_path)
        return self._demographic_raw

    @property
    def demographic(self) -> pd.DataFrame:
        """
        The demographics data.
        """
        if self._demographic_df is None:
            self._demographic_df = self.process_demographics().sort_values(
                ["patient_id"]
            )
        return self._demographic_df

    @property
    def demographic_types(self) -> typing.List[str]:
        """
        The demographic features that are available.
        """
        if self._demographic_types is None:
            self._demographic_types = self.demographic_raw.drop(
                "patient_id", axis=1
            ).columns.tolist()
        return self._demographic_types

    def process_activity(self) -> pd.DataFrame:
        """
        Process the activity data by taking the sum
        of the frequency of sensor firings for each
        patient_id, each day.
        """
        activity_data = (
            self.activity_raw.assign(date=lambda x: pd.to_datetime(x["date"]))
            # grouping by id and date
            .groupby(["patient_id", pd.Grouper(key="date", freq="1d"), "location_name"])
            .size()  # counting number of labels of each type
            .unstack()  # long to wide data frame
            .reset_index()
        )
        return activity_data

    def process_sleep(self) -> pd.DataFrame:
        """
        Process the sleep data by taking the mean
        and std of the heart rate and respiratory rate
        for each patient_id, each day.
        """

        # take means of the HR and RR over the day (00:00-23:59, not through the night)
        sleep = (
            self.sleep_raw.assign(date=lambda x: pd.to_datetime(x["date"]))
            # grouping by id and date
            .groupby(
                [
                    "patient_id",
                    pd.Grouper(key="date", freq="1d"),
                ]
            )
            # aggregating data
            .agg({"heart_rate": ["mean", "std"], "respiratory_rate": ["mean", "std"]})
        )
        # formatting column names
        sleep.columns = sleep.columns.map("_".join).str.strip("_")
        sleep = sleep.reset_index()
        return sleep

    def process_physiology(self) -> pd.DataFrame:
        """
        Process the physiology data by taking the mean
        and std of all of the device values for each
        patient_id, each day.
        """

        physiology = (
            self.physiology_raw.assign(date=lambda x: pd.to_datetime(x["date"]))
            # grouping by id and date
            .groupby(["patient_id", pd.Grouper(key="date", freq="1d"), "device_type"])
            .agg({"value": ["mean", "std"]})
            .unstack()  # long to wide data frame
        )
        physiology.columns = physiology.columns.map("_".join).str.strip("_")
        physiology = physiology.reset_index()
        return physiology

    def process_target(self) -> pd.DataFrame:
        labels = (
            self.target_raw.assign(
                date=lambda x: pd.to_datetime(x["date"])
            )  # ensuring date time is date_time
            .groupby(["patient_id", pd.Grouper(key="date", freq="1d"), "type"])
            .size()
            .unstack()
            .reset_index()
        )

        return labels

    def _check_exists(self) -> bool:
        """
        Checks if the data exists in the root directory.
        """
        return np.all(
            [
                os.path.exists(os.path.join(self.root, f"{data_name.title()}.csv"))
                for data_name in self.data_names
            ]
        )

    def process_demographics(self) -> pd.DataFrame:
        return self.demographic_raw

    def __len__(self) -> int:
        return len(self.data)


class TIHMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str = "./",
        train=True,
        imputer=impute.SimpleImputer(),
        n_days: int = 1,
        normalise: typing.Union[str, None] = "global",
    ):
        """
        A pytorch dataset which wraps the TIHM data. If 
        the TIHM data is not downloaded, then it will be
        in the directory given.

        This can be used with dataloaders or on its own.     
        
        
        Examples
        ---------

        .. code-block::
        
            >>> dataset = TIHMDataset(root='./data/')
        
        Arguments
        ---------

        - root: str, optional:
            The file path to where the TIHM
            data files are stored.
            Defaults to :code:`'./'`.
        
        - train: bool, optional:
            Whether to return the training or testing
            data when indexed. 
            Defaults to :code:`True`.
        
        - imputer: _type_, optional:
            The imputer to use to impute the missing
            values. This should be of the same structure
            as a Scikit-Learn imputer.
            Defaults to :code:`impute.SimpleImputer()`.
        
        - n_days: int, optional:
            The number of days to wrap into
            a single data point. This is useful
            when using recurrent networks. If this
            value is more than :code:`1`, then the 
            returned data will be of shape
            (n_data_points, n_days, n_features).
            Defaults to :code:`1`.
        
        - normalise: typing.Union[str, None], optional:
            Whether to normalise the data before 
            rolling it using n_days. This can
            be any of:

            - :code:`'global'`: Statistics are based \
            on all of the data. 

            - :code:`'id'`: Statistics are based on the \
            patient id if available. Otherwise global \
            statistics are used.

            - :code:`None`: No normalisation is done.
            
            Defaults to :code:`'global'`.
        
        
        """

        self.train = train  # saving whether training or testing
        self._dataset = TIHM(root=root)  # the dataset

        # splitting the data by date to get train-test split
        train_data, test_data, train_target, test_target = self._train_test_split(
            data=self._dataset.data, target=self._dataset.target, test_start=TEST_START
        )

        ## getting arrays from data frames
        # train
        train_patient_id = train_data["patient_id"].values
        train_date = train_data["date"].dt.date.values
        train_data = train_data.drop(["patient_id", "date"], axis=1).values
        train_target = train_target.drop(["patient_id", "date"], axis=1).values
        # test
        test_patient_id = test_data["patient_id"].values
        test_date = test_data["date"].dt.date.values
        test_data = test_data.drop(["patient_id", "date"], axis=1).values
        test_target = test_target.drop(["patient_id", "date"], axis=1).values

        # impute the data with the given imputer
        train_data, test_data = self._impute(
            train_data=train_data, test_data=test_data, imputer=imputer
        )

        if not normalise is None:
            # scale the data with the sklearn StandardScaler
            train_data, test_data = self._normalise(
                train_data=train_data,
                test_data=test_data,
                train_patient_id=train_patient_id,
                test_patient_id=test_patient_id,
                normalise=normalise,
            )

        # reformatting the training and testing data to contain the n_days in each data point
        if n_days > 1:
            (
                train_data,
                train_target,
                train_patient_id,
                train_date,
            ) = self._reformat_n_days(
                data=train_data,
                target=train_target,
                patient_id=train_patient_id,
                date=train_date,
                n_days=n_days,
            )

            test_data, test_target, test_patient_id, test_date = self._reformat_n_days(
                data=test_data,
                target=test_target,
                patient_id=test_patient_id,
                date=test_date,
                n_days=n_days,
            )

        # saving data to class attributes
        self.train_data, self.test_data = train_data, test_data
        self.train_target, self.test_target = train_target, test_target
        self.train_patient_id, self.test_patient_id = train_patient_id, test_patient_id
        self.train_date, self.test_date = train_date, test_date

        return

    @property
    def feature_names(self) -> typing.List[str]:
        """
        The names of the features in the x data.
        """
        return list(self._dataset.data.drop(["patient_id", "date"], axis=1).columns)

    @property
    def target_names(self) -> typing.List[str]:
        """
        The names of the features in the y data.
        """
        return list(self._dataset.target.drop(["patient_id", "date"], axis=1).columns)

    def _train_test_split(
        self,
        data: pd.DataFrame,
        target: pd.DataFrame,
        test_start: str,
    ) -> typing.Tuple[pd.DataFrame]:

        # train
        train_data = data[data["date"] < pd.to_datetime(test_start)]
        train_target = target[target["date"] < pd.to_datetime(test_start)]

        # test
        test_data = data[data["date"] >= pd.to_datetime(test_start)]
        test_target = target[target["date"] >= pd.to_datetime(test_start)]

        return train_data, test_data, train_target, test_target

    def _impute(
        self, train_data: np.ndarray, test_data: np.ndarray, imputer
    ) -> typing.Tuple[np.ndarray]:

        try:
            train_data = imputer.fit_transform(
                train_data
            )  # fit and transform with the train data
            test_data = imputer.transform(test_data)  # transform with the test data
        except AttributeError:
            raise TypeError(
                "Please ensure that the imputer is a sklearn imputer, "
                + "or implements the fit_transform and transform methods."
            )
        return train_data, test_data

    def _normalise(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        train_patient_id: np.ndarray,
        test_patient_id: np.ndarray,
        normalise: str,
    ) -> typing.Tuple[np.ndarray]:

        if normalise == "global":
            scaler = preprocessing.StandardScaler()
            train_data = scaler.fit_transform(train_data)
            test_data = scaler.transform(test_data)

        elif normalise == "id":
            scaler = StandardGroupScaler()
            train_data = scaler.fit_transform(train_data, groups=train_patient_id)
            test_data = scaler.transform(test_data, groups=test_patient_id)

        else:
            raise ValueError(
                f"normalise must be None, 'global' or 'id', not {normalise}."
            )

        return train_data, test_data

    def _reformat_n_days(
        self,
        data: np.ndarray,
        target: np.ndarray,
        patient_id: np.ndarray,
        date: np.ndarray,
        n_days: int,
    ) -> typing.Tuple[np.ndarray]:

        # new arrays
        data_out = []
        target_out = []
        patient_id_out = []
        date_out = []

        # iterate over patient_ids
        for n_id, id_val in enumerate(np.unique(patient_id)):
            idx_id = np.arange(data.shape[0])[patient_id == id_val][
                np.argsort(date[patient_id == id_val])
            ]
            idx_split = np.where(
                date[idx_id][1:] - date[idx_id][:-1] > dt.timedelta(days=1)
            )[0]
            idx_split = np.split(np.arange(idx_id.shape[0]), idx_split + 1)

            for n_split, i_split in enumerate(idx_split):
                data_i = data[idx_id][i_split]
                target_i = target[idx_id][i_split]
                patient_id_i = patient_id[idx_id][i_split]
                date_i = date[idx_id][i_split]

                # if X_i is not long enough to build a sequence
                # then we skip it
                if data_i.shape[0] < n_days:
                    continue

                # roll the data
                data_i_rolled = make_input_roll(data_i, n_days)
                target_i_rolled = make_input_roll(target_i, n_days)
                patient_id_i_rolled = make_input_roll(patient_id_i, n_days)
                date_i_rolled = make_input_roll(date_i, n_days)

                # append to the outputs
                data_out.append(data_i_rolled)
                target_out.append(target_i_rolled)
                patient_id_out.append(patient_id_i_rolled)
                date_out.append(date_i_rolled)

        # make outputs arrays
        data_out = np.vstack(data_out)
        target_out = np.vstack(target_out)
        patient_id_out = np.vstack(patient_id_out)
        date_out = np.vstack(date_out)

        return data_out, target_out, patient_id_out, date_out

    def __getitem__(self, index: int):
        if self.train:
            x, y = self.train_data[index], self.train_target[index]
        else:
            x, y = self.test_data[index], self.test_target[index]
        return x, y

    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)
