import logging
import logging.handlers

class Logger(object):
    def __init__(self,log_name,log_level):
        self.log_name = log_name
        self.log_level = log_level
        self.logger  = self.logger_init()


    def logger_init(self):
        logger_name = self.log_name
        # log_file = self.config.get("log", "log_file")
        level = self.log_level
        logger = logging.getLogger(logger_name)
        formatter = logging.Formatter("%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s")

        # fileHandler = logging.handlers.RotatingFileHandler(log_file, maxBytes=8388608, backupCount=5, encoding='utf-8')
        # fileHandler.setFormatter(formatter)

        stremaHandler = logging.StreamHandler()
        stremaHandler.setFormatter(formatter)

        logger.setLevel(level)
        # logger.addHandler(fileHandler)
        logger.addHandler(stremaHandler)
        return logger