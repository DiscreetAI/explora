import json
import logging
import requests
import time

import ipfsapi

from blockchain_client import BlockchainClient


logging.basicConfig(level=logging.DEBUG,
    format='[DMLClient] %(message)s')

class DMLClient(BlockchainClient):
    """
    Has the same prerequisites as BlockchainClient
    """

    def __init__(self, config_filepath: str = 'blockchain_config.json') -> None:
        """
        Connect with running IPFS node.
        """
        super.__init__()

    # helper function implementation

    def _learn(self, model=dict, participants=dict, optimizer=dict):
        # NOTE: Optimizer is FedAvgOptimizer by default for now
        """
        Provided a model and a list of participants who are expected to
        train this model, uploads the model packaged with the optimizer to
        IPFS and then stores a pointer on the blockchain.

        params
        @model: dict returned by make_model()
        @participants: dict returned by make_participants()
        @optimizer: dict returned by make_optimizer()
        """
        job_to_post = {}
        job_to_post["job_type"] = ""
        job_to_post["serialized_model"] = model["serialized_model"]
        # NOTE: Currently we only support Keras, so...
        job_to_post["framework_type"] = model.get("framework_type", "keras")
        job_to_post["hyperparams"] = model["hyperparams"]
        # NOTE: Currently we can't really send out a custom raw_filepath for
        # every data provider, so...
        # TODO: Something something map dataset_uuid to raw_filepath?
        job_to_post["raw_filepath"] = next(iter(participants.values()))["raw_filepath"]
        # NOTE: Currently we can't really send out a custom label_column_name for
        # every data provider, so...
        job_to_post["label_column_name"] = next(iter(participants.values()))["label_column_name"]
        serialized_job = {
            "job_data": job_to_post
        }
        new_session_event = {
            BlockchainClient.KEY: None,
            BlockchainClient.CONTENT: {
                "optimizer_params": optimizer,
                "serialized_job": serialized_job
            }
        }
        on_chain_value = self.client.add_json(new_session_event)
        tx = {BlockchainClient.KEY: on_chain_value, BlockchainClient.CONTENT: on_chain_value}
        timeout = time.time() + self.timeout
        tx_receipt = None
        while time.time() < timeout:
            try:
                tx_receipt = self._make_setter_call(tx)
                break
            except (UnboundLocalError, requests.exceptions.ConnectionError) as e:
                logging.info("HTTP SET error, got: {0}".format(e))
                continue
        return on_chain_value, tx_receipt_text

    def _make_model(self, model: object, batch_size: int=32, 
                    epochs: int=10, split: float=1, avg_type: str="data_size"):
        """
        Helper function for decentralized_learning

        params
        @model: Keras model
        @batch_size: minibatch size for data
        @epochs: number of epochs to train for
        @split: split for data in raw_filepath
        @avg_type: averaging type for decentralized learning
        """
        assert avg_type in ['data_size', 'val_acc'], \
            "Averaging type '{0}' is not supported.".format(avg_type)
        model_dict = {}
        model_json = model.to_json()
        model_dict["serialized_model"] = model_json
        hyperparams = {}
        hyperparams["batch_size"] = batch_size
        hyperparams["epochs"] = epochs
        hyperparams["split"] = split
        hyperparams["averaging_type"] = avg_type
        model_dict["hyperparams"] = hyperparams
        return model_dict
    
    def _make_participants(self, participants: list):
        """
        Returns a dict representing participants
        """
        # TODO: Fill this in once the communication schema is set
        # do dictionary comprehension
        return {participant["uuid"]: {
            "raw_filepath" : participant.get("raw_filepath", "datasets/mnist"),
            "label_column_name": participant.get("label_column_name", "label")
            } for participant in participants
        }
    
    def _make_optimizer(self, opt_type="fed_avg", 
                        num_rounds=1, num_averages_per_round=1):
        """
        Returns a dict optimizer_params
        """
        assert opt_type in ["fed_avg"], \
            "Optimizer '{0}' is not supported.".format(opt_type)
        optimizer_params = {
            "optimizer_type": opt_type,
            "num_averages_per_round": num_averages_per_round, 
            "max_rounds": num_rounds
        }
        return optimizer_params
    
    # one big function implementation

    # def train(self, model: object, participants, batch_size: int=32, 
    #         epochs: int=10, split: float=1, avg_type: str="data_size",
    #         opt_type="fed_avg", num_rounds=1):
    #     """
    #     model: Keras model
    #     participants: Comes from another component idrk
    #     """
    #     # asserts
    #     assert avg_type in ['data_size', 'val_acc'], \
    #         "Averaging type '{0}' is not supported.".format(avg_type)
    #     assert opt_type in ["fed_avg"], \
    #         "Optimizer '{0}' is not supported.".format(opt_type)
    #     # start making new job
    #     job_to_post = {}
    #     job_to_post["job_type"] = ""
    #     job_to_post["serialized_model"] = model.to_json()
    #     # now hyperparams
    #     hyperparams = {}
    #     hyperparams["batch_size"] = batch_size
    #     hyperparams["epochs"] = epochs
    #     hyperparams["split"] = split
    #     hyperparams["averaging_type"] = avg_type
    #     job_to_post["hyperparams"] = hyperparams
    #     # just make sure participants has defaults
    #     participants = {participant["uuid"]: {
    #         "raw_filepath" : participant.get("raw_filepath", "datasets/mnist"),
    #         "label_column_name": participant.get("label_column_name", "label")
    #         } for participant in participants
    #     }
    #     # NOTE: Currently we only support Keras, so...
    #     job_to_post["framework_type"] = "keras"
    #     # NOTE: Currently we can't really send out a custom raw_filepath for
    #     # every data provider, so...
    #     job_to_post["raw_filepath"] = next(iter(participants.values()))["raw_filepath"]
    #     # NOTE: Currently we can't really send out a custom label_column_name for
    #     # every data provider, so...
    #     job_to_post["label_column_name"] = next(iter(participants.values()))["label_column_name"]
    #     serialized_job = {
    #         "job_data": job_to_post
    #     }
    #     # now optimizer params
    #     optimizer_params = {
    #         "optimizer_type": opt_type,
    #         "num_averages_per_round": len(participants), 
    #         "max_rounds": num_rounds
    #     }
    #     new_session_event = {
    #         BlockchainClient.KEY: None,
    #         BlockchainClient.CONTENT: {
    #             "optimizer_params": optimizer,
    #             "serialized_job": serialized_job
    #         }
    #     }
    #     on_chain_value = self.client.add_json(new_session_event)
    #     tx = {BlockchainClient.KEY: on_chain_value, 
    #             BlockchainClient.CONTENT: on_chain_value}
    #     timeout = time.time() + self.timeout
    #     tx_receipt = None
    #     while time.time() < timeout:
    #         try:
    #             tx_receipt = self._make_setter_call(tx)
    #             break
    #         except (UnboundLocalError, requests.exceptions.ConnectionError) as e:
    #             logging.info("HTTP SET error, got: {0}".format(e))
    #             continue
    #     return tx_receipt.text

    def decentralized_learn(self, model: object, participants, batch_size: int=32, 
            epochs: int=10, split: float=1, avg_type: str="data_size",
            opt_type="fed_avg", num_rounds=1):
        """
        Calls a bunch of helper functions
        """
        model_dict = self._make_model(
            model=model,
            batch_size=batch_size,
            epochs=epochs,
            split=split,
            avg_type=avg_type
        )
        optimizer_params = self._make_optimizer(
            opt_type=opt_type,
            num_rounds=num_rounds,
            num_averages_per_round=len(participants)
        )
        participants=self._make_participants(
            participants=participants
        )
        key, receipt = self._learn(
            model=model_dict,
            optimizer=optimizer_params,
            participants=participants
        )
        return key
