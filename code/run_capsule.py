# stdlib imports --------------------------------------------------- #
import argparse
import dataclasses
import json
import functools
import logging
import pathlib
import time
import types
import typing
import uuid
from typing import Any, Literal

# 3rd-party imports necessary for processing ----------------------- #
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import pynwb
import upath
import zarr


# logging configuration -------------------------------------------- #
# use `logger.info(msg)` instead of `print(msg)` so we get timestamps and origin of log messages
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(name)s.%(funcName)s | %(message)s", 
    datefmt="%Y-%d-%m %H:%M:%S",
    )
logger = logging.getLogger(__name__)

# general configuration -------------------------------------------- #
matplotlib.rcParams['pdf.fonttype'] = 42
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR) # suppress matplotlib font warnings on linux


# utility functions ------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--session_id', type=str, default=None)
    parser.add_argument('--logging_level', type=str, default='INFO')
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--update_packages_from_source', type=int, default=1)
    parser.add_argument('--override_params_json', type=str, default="{}")
    for field in dataclasses.fields(Params):
        if field.name == 'decoder_type':
            print(field)
            print(type(field.type))
        if field.name in [getattr(action, 'dest') for action in parser._actions]:
            # already added field above
            continue
        logger.debug(f"adding argparse argument {field}")
        kwargs = {}
        if isinstance(field.type, str):
            kwargs = {'type': eval(field.type)}
        else:
            kwargs = {'type': field.type}
        if kwargs['type'] in (list, tuple):
            logger.debug(f"Cannot correctly parse list-type arguments from App Builder: skipping {field.name}")
        if isinstance(field.type, str) and field.type.startswith('Literal'):
            kwargs['type'] = str
        if isinstance(kwargs['type'], (types.UnionType, typing._UnionGenericAlias)):
            kwargs['type'] = typing.get_args(kwargs['type'])[0]
            logger.info(f"setting argparse type for union type {field.name!r} ({field.type}) as first component {kwargs['type']!r}")
        parser.add_argument(f'--{field.name}', **kwargs)
    args = parser.parse_args()
    list_args = [k for k,v in vars(args).items() if type(v) in (list, tuple)]
    if list_args:
        raise NotImplementedError(f"Cannot correctly parse list-type arguments from App Builder: remove {list_args} parameter and provide values via `override_params_json` instead")
    logger.info(f"{args=}")
    return args

@functools.cache
def get_datacube_dir() -> pathlib.Path:
    for p in get_data_root().iterdir():
        if p.is_dir() and p.name.startswith('dynamicrouting_datacube'):
            path = p
            break
    else:
        for p in get_data_root().iterdir():
            if any(pattern in p.name for pattern in ('session_table', 'nwb', 'consolidated', )):
                path = get_data_root()
                break
        else:
            raise FileNotFoundError(f"Cannot determine datacube dir: {list(get_data_root().iterdir())=}")
    logger.info(f"Using files in {path}")
    return path

@functools.cache
def get_data_root(as_str: bool = False) -> pathlib.Path:
    expected_paths = ('/data', '/tmp/data', )
    for p in expected_paths:
        if (data_root := pathlib.Path(p)).exists():
            logger.info(f"Using {data_root=}")
        return data_root.as_posix() if as_str else data_root
    else:
        raise FileNotFoundError(f"data dir not present at any of {expected_paths=}")

@functools.cache
def get_nwb_paths() -> tuple[pathlib.Path, ...]:
    return tuple(get_data_root().rglob('*.nwb'))
    
def get_nwb(session_id_or_path: str | pathlib.Path, raise_on_missing: bool = True, raise_on_bad_file: bool = True) -> pynwb.NWBFile:
    if isinstance(session_id_or_path, (pathlib.Path, upath.UPath)):
        nwb_path = session_id_or_path
    else:
        if not isinstance(session_id_or_path, str):
            raise TypeError(f"Input should be a session ID (str) or path to an NWB file (str/Path), got: {session_id_or_path!r}")
        if pathlib.Path(session_id_or_path).exists():
            nwb_path = session_id_or_path
        elif session_id_or_path.endswith(".nwb") and any(p.name == session_id_or_path for p in get_nwb_paths()):
            nwb_path = next(p for p in get_nwb_paths() if p.name == session_id_or_path)
        else:
            try:
                nwb_path = next(p for p in get_nwb_paths() if p.stem == session_id_or_path)
            except StopIteration:
                msg = f"Could not find NWB file for {session_id_or_path!r}"
                if not raise_on_missing:
                    logger.error(msg)
                    return
                else:
                    raise FileNotFoundError(f"{msg}. Available files: {[p.name for p in get_nwb_paths()]}") from None
    logger.info(f"Reading {nwb_path}")
    try:
        nwb = pynwb.NWBHDF5IO(nwb_path).read()
    except RecursionError:
        msg = f"{nwb_path.name} cannot be read due to RecursionError (hdf5 may still be accessible)"
        if not raise_on_bad_file:
            logger.error(msg)
            return
        else:
            raise RecursionError(msg)
    else:
        return nwb
        
def ensure_nonempty_results_dir() -> None:
    # pipeline can crash if a results folder is expected and not found, and requires creating manually:
    results = pathlib.Path("/results")
    results.mkdir(exist_ok=True)
    if not list(results.iterdir()):
        (results / uuid.uuid4().hex).touch()

# processing function ---------------------------------------------- #
# modify the body of this function, but keep the same signature

def process_session(session_id: str, params: "Params", test: int = 0) -> None:
    """Process a single session with parameters defined in `params` and save results + params to
    /results.
    
    A test mode should be implemented to allow for quick testing of the capsule (required every time
    a change is made if the capsule is in a pipeline) 
    """
    # Get nwb file
    # Currently this can fail for two reasons: 
    # - the file is missing from the datacube, or we have the path to the datacube wrong (raises a FileNotFoundError)
    # - the file is corrupted due to a bad write (raises a RecursionError)
    # Choose how to handle these as appropriate for your capsule
    try:
        nwb = get_nwb(session_id, raise_on_missing=True, raise_on_bad_file=True) 
    except (FileNotFoundError, RecursionError) as exc:
        logger.info(f"Skipping {session_id}: {exc!r}")
        return
    
    # Get components from the nwb file:
    trials_df = nwb.trials[:]
    units_df = nwb.units[:]
    
    # Process data here, with test mode implemented to break out of the loop early:
    logger.info(f"Processing {session_id} with {params.to_json()}")
    results = {}
    for structure, structure_df in units_df.groupby('structure'):
        results[structure] = len(structure_df)
        if test:
            logger.info("Test mode: exiting after first structure")
            break

    # Save data to files in /results
    # If the same name is used across parallel runs of this capsule in a pipeline, a name clash will
    # occur and the pipeline will fail, so use session_id as filename prefix:
    #   /results/<sessionId>.suffix
    logger.info(f"Writing results for {session_id}")
    np.savez(f'/results/{session_id}.npz', **results)
    params.write_json(f'/results/{session_id}.json')

# define run params here ------------------------------------------- #

# The `Params` class is used to store parameters for the run, for passing to the processing function.
# @property fields (like `bins` below) are computed from other parameters on-demand as required:
# this way, we can separate the parameters dumped to json from larger arrays etc. required for
# processing.

# - if needed, we can get parameters from the command line (like `nUnitSamples` below) and pass them
#   to the dataclass (see `main()` below)

# this is an example from Sam's processing code, replace with your own parameters as needed:
@dataclasses.dataclass
class Params:
    nUnitSamples: int = 20
    unitSampleSize: int = 20
    windowDur: float = 1
    binSize: float = 1
    nShuffles: int | str = 100
    binStart: int = -windowDur
    n_units: list = dataclasses.field(default_factory=lambda: [5, 10, 20, 40, 60, 'all'])
    decoder_type: str | Literal['linearSVC', 'LDA', 'RandomForest', 'LogisticRegression'] = 'LogisticRegression'

    @property
    def bins(self) -> npt.NDArray[np.float64]:
        return np.arange(self.binStart, self.windowDur+self.binSize, self.binSize)

    @property
    def nBins(self) -> int:
        return self.bins.size - 1
    
    def to_dict(self) -> dict[str, Any]:
        """dict of field name: value pairs, including values from property getters"""
        return dataclasses.asdict(self) | {k: getattr(self, k) for k in dir(self.__class__) if isinstance(getattr(self.__class__, k), property)}

    def to_json(self, **dumps_kwargs) -> str:
        """json string of field name: value pairs, excluding values from property getters (which may be large)"""
        return json.dumps(dataclasses.asdict(self), **dumps_kwargs)

    def write_json(self, path: str | upath.UPath = '/results/params.json') -> None:
        path = upath.UPath(path)
        logger.info(f"Writing params to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=2))

# ------------------------------------------------------------------ #


def main():
    t0 = time.time()

    # get arguments passed from command line (or "AppBuilder" interface):
    args = parse_args()
    logger.setLevel(args.logging_level)

    # if any of the parameters required for processing are passed as command line arguments, we can
    # get a new params object with these values in place of the defaults:

    params = {}
    for field in dataclasses.fields(Params):
        if (val := getattr(args, field.name, None)) is not None:
            params[field.name] = val
    
    override_params = json.loads(args.override_params_json)
    if override_params:
        for k, v in override_params.items():
            if k in params:
                logger.info(f"Overriding value of {k!r} from command line arg with value specified in `override_params_json`")
            params[k] = v
            
    # if session_id is passed as a command line argument, we will only process that session,
    # otherwise we process all session IDs that match filtering criteria:    
    session_table = pd.read_parquet(get_datacube_dir() / 'session_table.parquet')
    session_ids: list[str] = session_table.query(
        "is_ephys & project=='DynamicRouting' & is_task & is_annotated & ~is_context_naive"
    )['session_id'].values.tolist()
    if args.session_id is not None:
        if args.session_id not in session_ids:
            logger.info(f"{args.session_id!r} not in filtered sessions list")
            exit()
        logger.info(f"Using single session_id {args.session_id} provided via command line argument")
        session_ids = [args.session_id]
    else:
        logger.info(f"Using list of {len(session_ids)} session_ids")

    # run processing function for each session, with test mode implemented:
    for session_id in session_ids:
        process_session(session_id, params=Params(**params), test=args.test)
        if args.test:
            logger.info("Test mode: exiting after first session")
            break
    ensure_nonempty_results_dir()
    logger.info(f"Time elapsed: {time.time() - t0:.2f} s")

if __name__ == "__main__":
    main()
