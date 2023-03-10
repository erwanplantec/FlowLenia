# FlowLenia

Implementation of the Flow Lenia model. The model is implemented in JAX and so supports GPU acceleration.

Paper : https://arxiv.org/abs/2212.07906

Companion website : https://sites.google.com/view/flowlenia

# Files

## flowlenia

Flow Lenia implementations in JAX.

### flowlenia.py

Main implementation of the Flow Lenia system. Main components are :

- **Config** : dataclass containing Flow Lenia configuration variables.
- **State** : dataclass representing state of the system (activations (A))
- **FlowLenia** : class packing all the FlowLenia components (step function, kernel computer, rule space).

### flowlenia_params.py

Implementation of Flow Lenia with parameter embedding mechanism.

- **Config** (Imported as Config_P) : Configuration dataclass
- **State** : dataclass representing state of the system (activations (A) + parameter map (P))
- **FlowLeniaParams** : class packing all the components of the Flow lenia model withn parametr embedding (same as FlowLenia)


## examples

Small examples showing how to use the flowlenia codebase.

- example_1C.py : Instantiation of 1-channel multi-kernels Flow Lenia.
- example_2C.py : 2-channels multi-kernels Flow Lenia
- parameter_embedding.py : Instantitation of Flow Lenia with parameter embedding.
