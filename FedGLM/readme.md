# FedGLM
This project is based on the open source federated learning framework flower(https://github.com/adap/flower) and the open source large language model ChatGLM-6B.

## Setup the enviorment
Download chatglm-6b first.
```python
pip install -r requirements.txt
```

## Quick 

You can simply start the split-fed server in a terminal as follows:
```python
python ./flserver.py
```
you'll see:
```
WARNING flwr 2024-03-27 20:54:35,679 | app.py:211 | Both server and strategy were provided, ignoring strategy
INFO flwr 2024-03-27 20:54:35,680 | app.py:163 | Starting Flower server, config: ServerConfig(num_rounds=3, round_timeout=None)
INFO flwr 2024-03-27 20:54:35,712 | app.py:176 | Flower ECE: gRPC server running (3 rounds), SSL is disabled
INFO flwr 2024-03-27 20:54:35,712 | flserver.py:89 | Initializing fed-split learning!
```

wait a few secends,than you can start a client in the terminal:
```python
python ./flclient.py
```
if the terminal print:
```
INFO flwr 2024-03-27 20:55:03,538 | grpc.py:52 | Opened insecure gRPC connection (no certificates were passed)
DEBUG flwr 2024-03-27 20:55:03,548 | connection.py:55 | ChannelConnectivity.IDLE
DEBUG flwr 2024-03-27 20:55:03,551 | connection.py:55 | ChannelConnectivity.CONNECTING
```
that means the connection is failed, you can retry this command until you see:
```
INFO flwr 2024-03-27 20:56:27,450 | grpc.py:52 | Opened insecure gRPC connection (no certificates were passed)
DEBUG flwr 2024-03-27 20:56:27,451 | connection.py:55 | ChannelConnectivity.IDLE
DEBUG flwr 2024-03-27 20:56:27,452 | connection.py:55 | ChannelConnectivity.CONNECTING
DEBUG flwr 2024-03-27 20:56:27,452 | connection.py:55 | ChannelConnectivity.READY
```
If you need to increase the number of clients, you need to modify the 'min_fit_clients', 'min_evaluate_clients', 'min_available_clients' parameters in 'flserver.py' and use the same commands in a new terminal.

Once the training is completed, the training metrics and time spent will be printed out on the server-side terminal, as shown below:
```
WARNING flwr 2024-03-27 20:54:35,679 | app.py:211 | Both server and strategy were provided, ignoring strategy
INFO flwr 2024-03-27 20:54:35,680 | app.py:163 | Starting Flower server, config: ServerConfig(num_rounds=3, round_timeout=None)
...
```
## Hyparameters
By adjusting the hyperparameters 'do_train', 'do_eval' and 'do_predict' in flserver.py as well as flclient.py ' can be selected for training(set 'do_train=true'), validation or inference, and other important parameters such as the learning rate can be adjusted through the parameter configuration. It should be noted that the hyperparameters with the same name in flserver.py and flclient.py need to be kept consistent, or else it is very likely that errors will be triggered due to parameter errors.

## TO DO
The current open source programme only supports serial training, the parallel training part of the programme will be open source after a period of time.


