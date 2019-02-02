import requests
import json
import logging


logging.basicConfig(level=logging.DEBUG,
    format='[StatusServerClient] %(message)s')

class StatusServerClient(object):
    """
    """
    def __init__(self, config):
        self.server_get_url = config['SERVER']['server_get_url']

    def get_latest_stats(self, job_uuid):
        response = requests.get(self.server_get_url.format(job_uuid=job_uuid))
        response_dict = json.loads(response.text)
        assert 'status' in response_dict and response_dict['status'] == 'success', \
            "GET failed!"
        return response_dict['job_dict']
        