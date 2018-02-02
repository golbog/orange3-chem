import logging
import random
import uuid
from itertools import islice
from os.path import join, isfile
from AnyQt.QtCore import QSettings

import cachecontrol.caches
import numpy as np
import requests

from Orange.data import ContinuousVariable, Domain, Table
from Orange.misc.environ import cache_dir

from orangecontrib.chem.http2_client import Http2Client
from orangecontrib.chem.http2_client import MaxNumberOfRequestsError
from orangecontrib.chem.utils import md5_hash
from orangecontrib.chem.utils import save_pickle, load_pickle

log = logging.getLogger(__name__)


MODELS = {
    'smiles': {
        'name': 'CNN-Based SMILES Embedder',
        'description': 'CNN model trained on Pharmacologic Action MeSH terms classification',
        'target_smiles_length': 1021,
        'layers': ['penultimate'],
        'order': 0
    },
}


class EmbeddingCancelledException(Exception):
    """Thrown when the embedding task is cancelled from another thread.
    (i.e. MoleculeEmbedder.cancelled attribute is set to True).
    """


class MoleculeEmbedder(Http2Client):
    """"Client side functionality for accessing a remote http2
    molecule embedding backend.

    """
    _cache_file_blueprint = '{:s}_{:s}_embeddings.pickle'
    MAX_REPEATS = 4
    CANNOT_LOAD = "cannot load"

    def __init__(self, model="smiles", layer="penultimate",
                 server_url='api.garaza.io:443'):
        super().__init__(server_url)
        model_settings = self._get_model_settings_confidently(model, layer)
        self._model = model
        self._layer = layer
        self._target_smiles_length = model_settings['target_smiles_length']

        cache_file_path = self._cache_file_blueprint.format(model, layer)
        self._cache_file_path = join(cache_dir(), cache_file_path)
        self._cache_dict = self._init_cache()

        self._session = cachecontrol.CacheControl(
            requests.session(),
            cache=cachecontrol.caches.FileCache(
                join(cache_dir(), __name__ + ".MoleculeEmbedder.httpcache"))
        )

        # attribute that offers support for cancelling the embedding
        # if ran in another thread
        self.cancelled = False
        self.machine_id = \
            QSettings().value('error-reporting/machine-id', '', type=str)  \
            or str(uuid.getnode())
        self.session_id = None

    @staticmethod
    def _get_model_settings_confidently(model, layer):
        if model not in MODELS.keys():
            model_error = "'{:s}' is not a valid model, should be one of: {:s}"
            available_models = ', '.join(MODELS.keys())
            raise ValueError(model_error.format(model, available_models))

        model_settings = MODELS[model]

        if layer not in model_settings['layers']:
            layer_error = (
                "'{:s}' is not a valid layer for the '{:s}'"
                " model, should be one of: {:s}")
            available_layers = ', '.join(model_settings['layers'])
            raise ValueError(layer_error.format(layer, model, available_layers))

        return model_settings

    def _init_cache(self):
        if isfile(self._cache_file_path):
            try:
                return load_pickle(self._cache_file_path)
            except EOFError:
                return {}

        return {}

    def __call__(self, *args, **kwargs):
        if len(args) and isinstance(args[0], Table) or \
                ("data" in kwargs and isinstance(kwargs["data"], Table)):
            return self.from_table(*args, **kwargs)
        elif (len(args) and isinstance(args[0], (np.ndarray, list))) or \
                ("smiles" in kwargs and isinstance(kwargs["smiles"], (np.ndarray, list))):
            return self.from_smiles(*args, **kwargs)
        else:
            raise TypeError

    def from_table(self, data, col="SMILES", smiles_processed_callback=None):  # !!
        smiles = data[:, col].metas.flatten()
        embeddings = self.from_smiles(smiles, smiles_processed_callback)
        return MoleculeEmbedder.prepare_output_data(data, embeddings)

    def from_smiles(self, smiles, smiles_processed_callback=None):
        """Send the smiles to the remote server in batches. The batch size
        parameter is set by the http2 remote peer (i.e. the server).

        Parameters
        ----------
        smiles: list
            A list of smiles for moelcules to be embedded.

        smiles_processed_callback: callable (default=None)
            A function that is called after each smiles is fully processed
            by either getting a successful response from the server,
            getting the result from cache or skipping the smiles.

        Returns
        -------
        embeddings: array-like
            Array-like of float16 arrays (embeddings) for
            successfully embedded smiles and Nones for skipped smiles.

        Raises
        ------
        ConnectionError:
            If disconnected or connection with the server is lost
            during the embedding process.

        EmbeddingCancelledException:
            If cancelled attribute is set to True (default=False).
        """
        if not self.is_connected_to_server():
            self.reconnect_to_server()

        self.session_id = str(random.randint(1, 1e10))
        all_embeddings = [None] * len(smiles)
        repeats_counter = 0

        # repeat while all smiles has embeddings or
        # while counter counts out (prevents cycling)
        while len([el for el in all_embeddings if el is None]) > 0 and \
            repeats_counter < self.MAX_REPEATS:

            # take all smiles without embeddings yet
            selected_indices = [i for i, v in enumerate(all_embeddings)
                                if v is None]
            smiles_wo_emb = [(smiles[i], i) for i in selected_indices]

            for batch in self._yield_in_batches(smiles_wo_emb):
                b_smiles, b_indices = zip(*batch)
                try:
                    embeddings = self._send_to_server(
                        b_smiles, smiles_processed_callback, repeats_counter
                    )
                except MaxNumberOfRequestsError:
                    # maximum number of http2 requests through a single
                    # connection is exceeded and a remote peer has closed
                    # the connection so establish a new connection and retry
                    # with the same batch (should happen rarely as the setting
                    # is usually set to >= 1000 requests in http2)
                    self.reconnect_to_server()
                    embeddings = [None] * len(batch)

                # insert embeddings into the list
                for i, emb in zip(b_indices, embeddings):
                    all_embeddings[i] = emb

                self.persist_cache()
            repeats_counter += 1

        # # change smiles that were not loaded from 'cannot loaded' to None
        all_embeddings = \
            [None if not isinstance(el, np.ndarray) and el == self.CANNOT_LOAD
             else el for el in all_embeddings]

        return np.array(all_embeddings)

    def _yield_in_batches(self, list_):
        gen_ = (s for s in list_)
        batch_size = self._max_concurrent_streams

        num_yielded = 0

        while True:
            batch = list(islice(gen_, batch_size))
            num_yielded += len(batch)

            yield batch

            if num_yielded == len(list_):
                return

    def _send_to_server(self, smiles_list, smiles_processed_callback, retry_n):
        """ Load smiles and compute cache keys and send requests to
        an http2 server for valid ones.
        """
        cache_keys = []
        http_streams = []

        for smiles in smiles_list:
            if self.cancelled:
                raise EmbeddingCancelledException()

            if not smiles:
                # skip the sending because smiles was skipped at loading
                http_streams.append(None)
                cache_keys.append(None)
                continue

            cache_key = md5_hash(smiles.encode('utf-8'))
            cache_keys.append(cache_key)
            if cache_key in self._cache_dict:
                # skip the sending because smiles is present in the
                # local cache
                http_streams.append(None)
                continue

            try:
                headers = {
                    'Content-Type': 'text/plain',
                    'Content-Length': str(len(smiles))
                }
                stream_id = self._send_request(
                    method='POST',
                    url='/chem/' + self._model +
                        '?machine={}&session={}&retry={}'
                        .format(self.machine_id, self.session_id, retry_n),
                    headers=headers,
                    body_bytes=smiles.encode('utf-8')
                )
                http_streams.append(stream_id)
            except ConnectionError:
                self.persist_cache()
                raise

        # wait for the responses in a blocking manner
        return self._get_responses_from_server(
            http_streams,
            cache_keys,
            smiles_processed_callback
        )

    def _get_responses_from_server(self, http_streams, cache_keys,
                                   smiles_processed_callback):
        """Wait for responses from an http2 server in a blocking manner."""
        embeddings = []

        for stream_id, cache_key in zip(http_streams, cache_keys):
            if self.cancelled:
                raise EmbeddingCancelledException()

            if not stream_id and not cache_key:
                # when smiles cannot be loaded
                embeddings.append(self.CANNOT_LOAD)

                if smiles_processed_callback:
                    smiles_processed_callback(success=False)
                continue

            if not stream_id:
                # skip rest of the waiting because smiles was either
                # skipped at loading or is present in the local cache
                embedding = self._get_cached_result_or_none(cache_key)
                embeddings.append(embedding)

                if smiles_processed_callback:
                    smiles_processed_callback(success=embedding is not None)
                continue

            try:
                response = self._get_json_response_or_none(stream_id)
            except ConnectionError:
                self.persist_cache()
                raise

            if not response or 'embedding' not in response:
                # returned response is not a valid json response
                # or the embedding key not present in the json
                embeddings.append(None)
            else:
                # successful response
                embedding = np.array(response['embedding'], dtype=np.float16)
                embeddings.append(embedding)
                self._cache_dict[cache_key] = embedding

            if smiles_processed_callback:
                smiles_processed_callback(embeddings[-1] is not None)

        return embeddings

    def _get_cached_result_or_none(self, cache_key):
        if cache_key in self._cache_dict:
            return self._cache_dict[cache_key]
        return None

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.disconnect_from_server()

    def clear_cache(self):
        self._cache_dict = {}
        self.persist_cache()

    def persist_cache(self):
        save_pickle(self._cache_dict, self._cache_file_path)

    @staticmethod
    def construct_output_data_table(embedded_smiles, embeddings):
        X = np.hstack((embedded_smiles.X, embeddings))
        Y = embedded_smiles.Y

        attributes = [ContinuousVariable.make('n{:d}'.format(d))
                      for d in range(embeddings.shape[1])]
        attributes = list(embedded_smiles.domain.attributes) + attributes

        domain = Domain(
            attributes=attributes,
            class_vars=embedded_smiles.domain.class_vars,
            metas=embedded_smiles.domain.metas
        )

        return Table(domain, X, Y, embedded_smiles.metas)

    @staticmethod
    def prepare_output_data(input_data, embeddings):
        skipped_smiles_bool = np.array([x is None for x in embeddings])

        if np.any(skipped_smiles_bool):
            skipped_smiles = input_data[skipped_smiles_bool]
            skipped_smiles = Table(skipped_smiles)
            skipped_smiles.ids = input_data.ids[skipped_smiles_bool]
            num_skipped = len(skipped_smiles)
        else:
            num_skipped = 0
            skipped_smiles = None

        embedded_smiles_bool = np.logical_not(skipped_smiles_bool)

        if np.any(embedded_smiles_bool):
            embedded_smiles = input_data[embedded_smiles_bool]

            embeddings = embeddings[embedded_smiles_bool]
            embeddings = np.stack(embeddings)

            embedded_smiles = MoleculeEmbedder.construct_output_data_table(
                embedded_smiles,
                embeddings
            )
            embedded_smiles.ids = input_data.ids[embedded_smiles_bool]
        else:
            embedded_smiles = None

        return embedded_smiles, skipped_smiles, num_skipped

    @staticmethod
    def filter_string_attributes(data):
        metas = data.domain.metas
        return [m for m in metas if m.is_string]
