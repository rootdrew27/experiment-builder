from typing import Callable, Dict, Any
import logging
from pathlib import Path
from expb.experiments._helpers import _setup_logger, _create_exp_dir


class Experiment(object):
    def __init__(
        self,
        name: str,
        path_to_exp_script: Path | str,
        log_level: int = logging.INFO,
        save_path: Path | str | None = None,
        description: str | None = None,
    ):
        """
        An object of this class is used to run and manage an experiment.
        Args:
            name (str): _description_
            path_to_exp_script (Path | str): The path to the script containing
            log_level (int, optional): The logging level. Pass None, if you wish logging to be disabled. Defaults to logging.INFO.
            save_path (Path | str, optional): A path to a directory. If set to None, the current working directory is used. **Note**: results will be stored in a sub-folder of the specified directory path, with a name passed for the 'name' argument. Defaults to None.
            description (str, optional): _description_. Defaults to None.
        """
        self._name = name
        self._description = description
        self._save_path = save_path if save_path is not None else Path.cwd()
        self.preprocesser:Callable = None
        self.trainer:Callable = None
        self.tester:Callable = None
        self.metric_calculator:Callable = None
        self.result_visualizer: Callable = None

        self._logger = _setup_logger(save_path, log_level, name)
        _create_exp_dir(name, save_path)

    def __str__(self):
        return f"""
        Experiment: {self.name}
        \n=====\n
        Description: {self.description}
        \n===\n
        Logger:\n
        name: {self.logger.name}
        """

    def run(self, params:Dict[str, Any]):
        pass