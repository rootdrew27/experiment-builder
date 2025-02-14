# Experiment Builder

Description
---
This package facilitates and formalizes the experimentation process for computer vision. It provides the following features: logging, profiling, optimization, and most importantly organization. Furthermore, it aims promote a standardized, yet expressive, experimentation process through consistent, yet easily extensible, APIs (e.g. Dataset, Action, Experiment).

## Conceptual Structure (Core)

- **Experiment** The core element of this package, experiments are designed through, managed by, and run with this class.  
- **Dataset** Objects of this class are essentially beefy wrappers of NumPy arrays. They are interfaces to data and metadata of the dataset.
- **Action** Actions are an extensible interface that provide predictable and efficient operations to use on Dataset objects.


## TODO

- [ ] Experiment API
    - [ ] Logging
    - [ ] Profiling
    - [ ] Optimization (backend, threading)
    - [ ] Organization

- [ ] Dataset
    - [ ] Handle variable image shapes (use batching or padding). 

- [ ] Metadata
    - [ ] Write tests
    - [ ] Visualization
    