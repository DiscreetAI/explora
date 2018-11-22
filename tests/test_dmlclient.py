import tests.context
import pytest

from core.dml_client import DMLClient

@pytest.fixture(scope='session')
def dml_client():
    """
    DMLClient instance
    """
    return DMLClient(
        config_filepath='tests/artifacts/blockchain_client/blockchain_config.json'
    )

@pytest.fixture(scope='session')
def model():
    """
    Returns a model in Keras that has been compiled with its optimizer
    """

@pytest.fixture(scope='session')
def participants():
    """
    Returns a dict of participants
    """

def test_dml_client_serializes_job_correctly(dml_client):
    key = dml_client.decentralized_learn(

    )