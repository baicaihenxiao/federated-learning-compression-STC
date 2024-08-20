# Federated Learning Simulator

Simulate Federated Learning with compressed communication on a large number of Clients.

Recreate experiments described in [*Sattler, F., Wiedemann, S., Müller, K. R., & Samek, W. (2019). Robust and Communication-Efficient Federated Learning from Non-IID Data. arXiv preprint arXiv:1903.02891.*](https://arxiv.org/abs/1903.02891)



## Usage
First, set environment variable 'TRAINING_DATA' to point to the directory where you want your training data to be stored. MNIST, FASHION-MNIST and CIFAR10 will download automatically. 

`python federated_learning.py`

will run the Federated Learning experiment specified in  

`federated_learning.json`.

You can specify:

### Task
- `"dataset"` : Choose from `["mnist", "cifar10", "kws", "fashionmnist"]`
- `"net"` : Choose from `["logistic", "lstm", "cnn", "vgg11", "vgg11s"]`

* -> See `default_hyperparameters.py`

net:

```python
hp_net_dict = {

  'logistic': 
          {'type' : 'CNN', 'lr' : 0.04, 'batch_size' : 100, 'lr_decay' : ['LambdaLR', {'lr_lambda' : lambda epoch: 1.0}], 
          'iterations' : 36000, 'momentum' : 0.0, 'lr_decay' : ['LambdaLR', {'lr_lambda' : lambda epoch: 1.0}]},

  'cnn': 
          {'type' : 'CNN', 'lr' : 0.1, 'batch_size' : 200, 'weight_decay' : 0.0, 'optimizer' : 'SGD', 'momentum' : 0.0,
          'lr_decay' : ['LambdaLR', {'lr_lambda' : lambda epoch: 1.0}], 'iterations' : 8000},

  'lstm': 
          {'type' : 'CNN', 'lr' : 0.1, 'momentum' : 0.9, 'batch_size' : 200, 'weight_decay' : 0.0, 'optimizer' : 'SGD',
          'lr_decay' : ['LambdaLR', {'lr_lambda' : lambda epoch: 1.0}], 'iterations' : 8000},

  'vgg11s': 
          {'type' : 'CNN', 'lr' : 0.016, 'batch_size' : 200, 'weight_decay' : 5e-5, "momentum" : 0.9,
          'lr_decay' : ['LambdaLR', {'lr_lambda' : lambda epoch: 1.0}], 'iterations' : 36000},

  'vgg11': 
          {'type' : 'CNN', 'lr' : 0.05, 'momentum' : 0.9, 'batch_size' : 200, 'weight_decay' : 5e-4,
          'lr_decay' : ['LambdaLR', {'lr_lambda' : lambda epoch: 0.99**epoch}], 'iterations' : 36000},
}
```

### Federated Learning Environment

- `"n_clients"` : Number of Clients
- `"classes_per_client"` : Number of different Classes every Client holds in it's local data
- `"participation_rate"` : Fraction of Clients which participate in every Communication Round
- `"batch_size"` : Batch-size used by the Clients
- `"balancedness"` : Default 1.0, if <1.0 data will be more concentrated on some clients
- `"iterations"` : Total number of training iterations
- `"momentum"` : Momentum used during training on the clients

### Compression Method

- `"compression"` : Choose from `[["none", {}], ["fed_avg", {"n" : ?}], ["signsgd", {"lr" : ?}], ["stc_updown", [{"p_up" : ?, "p_down" : ?}]], ["stc_up", {"p_up" : ?}], ["dgc_updown", [{"p_up" : ?, "p_down" : ?}]], ["dgc_up", {"p_up" : ?}] ]`

* -> See `default_hyperparameters.py`

Compression method:

```
def get_hp_compression(compression):

  c = compression[0]
  hp = compression[1]

  if c ==  "none" : 
    return  {"compression_up" : ["none", {}], "compression_down" : ["none", {}],
               "accumulation_up" : False, "accumulation_down" : False,  "aggregation" : "mean"}

  if c ==  "signsgd" : 
    return  {"compression_up" : ["signsgd", {}], "compression_down" : ["none", {}],
               "accumulation_up" : False, "accumulation_down" : False,  "aggregation" : "majority", "lr" : hp["lr"], "local_iterations" : 1}

  if c ==  "dgc_up" : 
    return  {"compression_up" : ["dgc", {"p" : hp["p_up"]}], "compression_down" : ["none", {}],
               "accumulation_up" : True, "accumulation_down" : False,  "aggregation" : "mean"}

  if c ==  "stc_up" : 
    return  {"compression_up" : ["stc", {"p" : hp["p_up"]}], "compression_down" : ["none", {}],
               "accumulation_up" : True, "accumulation_down" : False,  "aggregation" : "mean"}

  if c ==  "dgc_updown" : 
    return  {"compression_up" : ["dgc", {"p" : hp["p_up"]}], "compression_down" : ["dgc", {"p" : hp["p_down"]}],
               "accumulation_up" : True, "accumulation_down" : True,  "aggregation" : "mean"}    

  if c ==  "stc_updown" : 
    return {"compression_up" : ["stc", {"p" : hp["p_up"]}], "compression_down" : ["stc", {"p" : hp["p_down"]}],
               "accumulation_up" : True, "accumulation_down" : True,  "aggregation" : "mean"}

  if c ==  "fed_avg" : 
    return {"compression_up" : ["none", {}], "compression_down" : ["none", {}],
               "accumulation_up" : False, "accumulation_down" : False,  "aggregation" : "weighted_mean", "local_iterations" : hp["n"]}
```

### Logging 
- `"log_frequency"` : Number of communication rounds after which results are logged and saved to disk
- `"log_path"` : e.g. "results/experiment1/"

Run multiple experiments by listing different configurations.

## Options
- `--schedule` : specify which batch of experiments to run, defaults to "main"

## Citation 
[Paper](https://arxiv.org/abs/1903.02891)

Sattler, F., Wiedemann, S., Müller, K. R., & Samek, W. (2019). Robust and Communication-Efficient Federated Learning from Non-IID Data. arXiv preprint arXiv:1903.02891.
