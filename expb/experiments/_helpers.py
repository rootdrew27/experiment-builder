import logging
from pathlib import Path

def _setup_logger(log_path:Path|str, log_level:int, name:str=None):
    name = __name__ if name is None else name
    logger = logging.getLogger(name)
    # console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(
        filename=log_path,
        mode="a",
        encoding='utf-8',
    )
    logger.addHandler(file_handler)
    formatter = logging.Formatter(
       "{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    return logger
    

def _create_exp_dir(name:str, save_path:Path|str):
    if isinstance(save_path, str):
        save_path = Path(save_path)

    exp_path = save_path/name
    if not exp_path.exists():
        save_path.mkdir(parents=True)