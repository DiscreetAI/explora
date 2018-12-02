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
def ipfs_client(dml_client):
    """
    IPFS Client instance
    """
    return dml_client.client

@pytest.fixture(scope='session')
def model():
    """
    Returns a model in Keras that has been compiled with its optimizer
    """
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    model = Sequential([
        Dense(32, input_shape=(784,)),
        Activation('relu'),
        Dense(10),
        Activation('softmax'),
    ])
    model.compile(optimizer='rmsprop',
                  loss='mse')
    return model

@pytest.fixture(scope='session')
def participants():
    """
    Returns a dict of participants = [
        (data_provider_name, {'dataset_uuid': uuid,
        'label_column_name': 'label'})
    ]
    For the MVP, all participants will see this long dict. They will lookup their own
    name, to get the uuid of their dataset and the label_column_name for their dataset.
    """
    return {"pandata": {"dataset_uuid": 1234}, "needless": {"dataset_uuid": 4567}}

def test_dml_client_serializes_job_correctly(dml_client, ipfs_client, model, participants):
    key = dml_client.decentralized_learn(
        model, participants
    )
    content = ipfs_client.get_json(key)["CONTENT"]
    true_model_json = model.to_json()
    assert true_model_json == content["serialized_job"]["job_data"]["serialized_model"]
    participants["pandata"]["label_column_name"] = "label"
    participants["needless"]["label_column_name"] = "label"
    assert participants == content["participants"]
    assert content["optimizer_params"]["optimizer_type"] == "fed_avg"