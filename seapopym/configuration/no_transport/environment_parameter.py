"""A module to manage the environment parameters of the simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from attrs import field, frozen, validators
from dask.distributed import Client

from seapopym.configuration.abstract_configuration import (
    AbstractChunkParameter,
    AbstractClientParameter,
    AbstractEnvironmentParameter,
)
from seapopym.logging.custom_logger import logger

if TYPE_CHECKING:
    from numbers import Number


@frozen
class ChunkParameter(AbstractChunkParameter):
    """The chunk size of the different dimensions."""

    functional_group: Literal["auto"] | int | None = field(
        default=1,
        validator=validators.optional(validators.instance_of((str, int))),
        metadata={"description": "The chunk size of the functional group dimension."},
    )
    latitude: Literal["auto"] | int | None = field(
        default=None,
        validator=validators.optional(validators.instance_of((str, int))),
        metadata={"description": "The chunk size of the latitude dimension."},
    )
    longitude: Literal["auto"] | int | None = field(
        default=None,
        validator=validators.optional(validators.instance_of((str, int))),
        metadata={"description": "The chunk size of the longitude dimension."},
    )
    time: None = field(
        init=False,
        default=None,
        metadata={
            "description": (
                "The chunk size of the time dimension. "
                "Present only to remind us that time is not divisible due to time dependencies."
            )
        },
    )

    def as_dict(self: ChunkParameter, *, with_fgroup: bool = True) -> dict:
        """Format to a dictionary as expected by xarray."""
        chunks = {}
        if with_fgroup:
            chunks["functional_group"] = self.functional_group
        if self.latitude is not None:
            chunks["latitude"] = self.latitude
        if self.longitude is not None:
            chunks["longitude"] = self.longitude
        return chunks


@frozen(kw_only=True)
class ClientParameter(AbstractClientParameter):
    """
    The client parameter for the Dask client.

    If an address is provided, the client will be initialized with this address and other parameters will be ignored.
    If no address is provided, a LocalCluster will be initialized with the other parameters.
    For more information about this class check the Dask documentation about LocalCluster and Client.
    """

    n_workers: int | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(int)),
        metadata={"description": "The number of workers."},
    )
    threads_per_worker: int | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(int)),
        metadata={"description": "The number of threads per worker."},
    )
    memory_limit: str | Number = field(default="auto", metadata={"description": "The memory limit of each worker."})
    client: Client | None = field(init=False, default=None, metadata={"description": "The Dask client."})

    @classmethod
    def from_address(cls: ClientParameter, address: str) -> ClientParameter:
        """
        Create a ClientParameter from an address.

        Example:
        -------
        ```python
        client = Client()
        client_param = ClientParameter.from_address(address=client.scheduler.address)
        print(client_param.client)
        ```

        """
        client = Client(address=address)
        workers = client.scheduler_info()["workers"]
        infos = ((w_info["nthreads"], w_info["memory_limit"]) for w_info in workers.values())
        nthreads, memory_limit = tuple(map(list, zip(*infos)))

        n_workers = len(nthreads)
        nthreads = int(np.mean(nthreads))
        memory_limit = int(np.mean(memory_limit))

        client_param = ClientParameter(n_workers=n_workers, threads_per_worker=nthreads, memory_limit=memory_limit)
        client_param.client = client
        return client_param

    def initialize_client(self: ClientParameter) -> None:
        """Initialize the client."""
        if self.client is None:
            client = Client(
                n_workers=self.n_workers,
                threads_per_worker=self.threads_per_worker,
                memory_limit=self.memory_limit,
            )
            object.__setattr__(self, "client", client)
        else:
            msg = "Trying to initialize an already initialized client."
            logger.warning(msg)

    def close_client(self: ClientParameter) -> None:
        """Close the client."""
        if self.client is None:
            msg = "Trying to close an already closed client."
            logger.warning(msg)
            return
        if self.client.cluster is None:
            msg = "The client has no cluster to close. If you are using a remote cluster, you should close it manually."
            logger.warning(msg)
        else:
            self.client.cluster.close()
        self.client.close()
        del self.client
        object.__setattr__(self, "client", None)


@frozen(kw_only=True)
class EnvironmentParameter(AbstractEnvironmentParameter):
    """Manage the different environment parameters of the simulation."""

    chunk: ChunkParameter = field(
        factory=ChunkParameter,
        validator=validators.instance_of(ChunkParameter),
        metadata={"description": "The chunk size of the different dimensions."},
    )
    client: ClientParameter = field(
        factory=ClientParameter,
        validator=validators.instance_of(ClientParameter),
        metadata={"description": "The client parameter."},
    )
