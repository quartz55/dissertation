import logging

import coloredlogs
from tqdm import tqdm

colored_formatter = coloredlogs.ColoredFormatter(
    fmt=
    "%(asctime)s %(levelname)8s %(filename)s (%(funcName)s:%(lineno)d)\n%(message)s",
    field_styles={
        'asctime': {
            'color': 'magenta'
        },
        'levelname': {
            'color': 'cyan',
            'bold': True
        },
        'filename': {
            'color': 'red'
        },
        'funcName': {
            'color': 'black',
            'inverse': True
        }
    })


class TqdmLoggingHandler(logging.StreamHandler):
    def __init__(self, formatter=None, stream=None):
        super().__init__(stream)
        if formatter is None:
            formatter = colored_formatter
        self.setFormatter(formatter)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)
