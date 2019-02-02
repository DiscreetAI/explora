import requests


class StatusServerClient(object):
    """
    """
    def __init__(self, config):
        self.server_get_url = config['SERVER']['server_get_url']
        