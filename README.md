# Experiment Builder

Description
---
Nothing here yet...

## Conceptual Structure (Core)

- **Experiment** The core element of this package, experiments are designed through, managed by, and run with this class.  
- **Dataset** Objects of this class are essentially beefy wrappers of NumPy arrays, with the added bonus of builtin memory mapping and batching. They are interfaces to the data and metadata of the dataset.
- **Apply/Execute Interface** 

## Progress

- [ ] Dataset
    - [ ] Batching and effective memory mapping
    - [ ] Handle variety of annotation formats
    - [ ] Handle variety of tasks
    - [ ] Handle variable image shapes (use batching or padding).

- [ ] Experiment API
    - [ ] Logging
    - [ ] Profiling
    - [ ] Optimization (backend, threading)
    - [ ] Organization (directory and file creation)
    - [ ] Metrics

- [ ] Metadata
    - [ ] Visualization
    - [x] Category Hierarchies
    - 
    