from typing import Iterable, Union, Any
from types import NoneType

import numpy as np
from numpy import ndarray
import awkward as ak
from awkward import Array as akArray, Record as akRecord

from sigmaepsilon.core.wrapping import Wrapper
from sigmaepsilon.core.typing import issequence

AwkwardLike = Union[akArray, akRecord]


__all__ = ["AkWrapper"]


class AkWrapper(Wrapper):
    """
    A wrapper for Awkward objects. This is the base class of many
    database classes in SigmaEpsilon projects.
    """

    _attr_map_ = {}

    def __init__(
        self,
        *args,
        wrap: Any = None,
        fields: Iterable[str] | NoneType = None,
        **kwargs,
    ):
        fields = {} if fields is None else fields
        assert isinstance(fields, dict)

        if wrap is None and (len(kwargs) + len(fields)) > 0:
            for k, v in kwargs.items():
                if isinstance(v, np.ndarray):
                    fields[k] = v

            if len(fields) > 0:
                wrap = ak.zip(fields, depth_limit=1)

        if len(kwargs) > 0:
            [kwargs.pop(k, None) for k in fields.keys()]

        super().__init__(*args, wrap=wrap, **kwargs)

    @property
    def db(self) -> akRecord:
        """
        Returns the wrapped Awkward object.
        """
        return self._wrapped

    def to_numpy(self, key: str) -> ndarray:
        """
        Returns a data with the specified key as a numpy array, if
        possible.
        """
        return self._wrapped[key].to_numpy()

    def to_dataframe(self, *args, fields: Iterable[str] | NoneType = None, **kwargs):
        """
        Returns the data of the database as a DataFrame.

        Parameters
        ----------
        *args: tuple, Optional
            Positional arguments to specify fields.
        fields: Iterable[str], Optional
            Valid field names to include in the parquet files.
        **kwargs: dict, Optional
            Keyword arguments forwarded to :func:`awkward.to_dataframe`.

        Returns
        -------
        pandas.DataFrame
        """
        akdb = self.to_ak(*args, fields=fields)
        return ak.to_dataframe(akdb, **kwargs)

    def to_parquet(
        self,
        path: str,
        *args,
        fields: Iterable[str] | NoneType = None,
        **kwargs,
    ) -> None:
        """
        Saves the data of the database to a parquet file.

        Parameters
        ----------
        *args: tuple, Optional
            Positional arguments to specify fields.
        path: str
            Path of the file being created.
        fields: Iterable[str], Optional
            Valid field names to include in the parquet files.
        **kwargs: dict, Optional
            Keyword arguments forwarded to :func:`awkward.to_parquet`.
        """
        if fields is None and len(args) == 0:
            ak.to_parquet(self.db, path, **kwargs)
        else:
            akdb = self.to_ak(*args, fields=fields, asarray=False)
            ak.to_parquet(akdb, path, **kwargs)

    @classmethod
    def from_parquet(cls, path: str) -> "AkWrapper":
        """
        Saves the data of the database to a parquet file.

        Parameters
        ----------
        path: str
            Path of the file being created.
        """
        return cls(wrap=ak.from_parquet(path))

    def to_ak(
        self,
        *args,
        fields: Iterable[str] | NoneType = None,
        asarray: bool = False,
    ) -> Union[akArray, akRecord]:
        """
        Returns the database with a specified set of fields as either
        an Awkward Record, or an Awkward Array. If there are no fields
        specified and the output is a record, the original database is
        returned.

        Parameters
        ----------
        *args: tuple, Optional
            Positional arguments to specify fields.
        fields: Iterable[str], Optional
            Valid field names to include in the returned objects.
        asarray: bool, Optional
            If True, the database is turned onto an Awkward Array before
            saving to file. Default is False.
        """
        if asarray:
            return self.to_akarray(*args, fields=fields)
        else:
            return self.to_akrecord(*args, fields=fields)

    def to_akarray(self, *args, fields: Iterable[str] | NoneType = None) -> akArray:
        """
        Returns the data of the mesh as an Awkward array.

        Parameters
        ----------
        *args: tuple, Optional
            Positional arguments to specify fields.
        fields: Iterable[str], Optional
            Valid field names to include in the returned objects.
        """
        ldb = self.to_list(*args, fields=fields)
        return ak.from_iter(ldb)

    def to_akrecord(self, *args, fields: Iterable[str] | NoneType = None) -> akRecord:
        """
        Returns the data of the mesh as an Awkward record.

        Parameters
        ----------
        *args: tuple, Optional
            Positional arguments to specify fields.
        fields: Iterable[str], Optional
            Valid field names to include in the returned objects.
        """
        d = self.to_dict(*args, fields=fields)
        return ak.zip(d, depth_limit=1)

    def to_dict(self, *args, fields: Iterable[str] | NoneType = None) -> dict:
        """
        Returns data of the object as a dictionary. Unless fields
        are specified, all fields are returned.

        Parameters
        ----------
        *args: tuple, Optional
            Positional arguments to specify fields.
        fields: Iterable[str], Optional
            A list of keys that might identify data in a database.
            Default is None.
        """
        db = self.db
        res = None

        if fields is None:
            fields = []

        fields.extend(args)

        if len(fields) == 0:
            fields = db.fields

        res = {}

        for f in fields:
            if f in db.fields:
                res[f] = db[f]
            else:
                raise KeyError(f"Field {f} not found.")

        return res

    def to_list(self, *args, fields: Iterable[str] | NoneType = None) -> list:
        """
        Returns data of the object as lists. Unless fields are
        specified, all fields are returned.

        Parameters
        ----------
        *args: tuple, Optional
            Positional arguments to specify fields.
        fields: Iterable[str], Optional
            A list of keys that might identify data in a database.
            Default is None.
        """
        db = self.db
        res = None

        if fields is None:
            if len(args) > 0:
                fields = []

        if isinstance(fields, Iterable):
            fields.extend(args)
            db_ = {}
            for f in fields:
                if f in db.fields:
                    db_[f] = db[f]
                else:
                    raise KeyError(f"Field {f} not found.")

            res = AkWrapper(fields=db_).to_list()
        else:
            res = db.to_list()
        return res

    def __len__(self) -> int:
        return len(self._wrapped)

    def __hasattr__(self, attr):
        if attr in self.__class__._attr_map_:
            attr = self.__class__._attr_map_[attr]
        return any([attr in self.__dict__, attr in self._wrapped.__dict__])

    def __getattr__(self, attr):
        if attr in self.__class__._attr_map_:
            attr = self.__class__._attr_map_[attr]

        if attr in self.__dict__:
            return self.__dict__[attr]

        try:
            return getattr(self._wrapped, attr)
        except Exception:
            name = self.__class__.__name__
            raise AttributeError(f"'{name}' object has no attribute called '{attr}'")

    def __getitem__(self, index: str) -> Any:
        is_str = isinstance(index, str)

        if is_str and index in self.__class__._attr_map_:
            index = self.__class__._attr_map_[index]

        return self.db[index]

    def __setitem__(self, index: str, value: Iterable[Any]) -> None:
        if not isinstance(index, str):
            raise TypeError(f"Expected a string, got {type(index)}")

        if not issequence(value):
            raise TypeError(f"Expected a sequence, got {type(value)}")

        if not len(value) == len(self):
            raise ValueError(
                "The provided value must have the same length as the database."
            )

        self._wrapped[index] = value

    def __contains__(self, item: str) -> bool:
        if not isinstance(item, str):
            return False

        if item in self._wrapped.fields:
            return True

        if item in self.__class__._attr_map_:
            return self.__class__._attr_map_[item] in self._wrapped.fields

        return False
