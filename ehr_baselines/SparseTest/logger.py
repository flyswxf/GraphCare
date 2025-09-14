import logging
def get_logger(exp_name: str):
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    os.makedirs('./training_logs', exist_ok=True)
    file_handler = logging.FileHandler(f'./training_logs/{exp_name}.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith(f'{exp_name}.log') for h in logger.handlers):
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(stream_handler)

    return logger

