# Prun: Playground for Model Pruning Techniques

# # Reproducibility

The environment can be reproduced via `nix` using the provided `flake.nix` file. To enter the development shell, run:

```bash
nix develop
```
This will set up all necessary dependencies and tools required for the project.

To recreate the `Apptainer` container, use the following command:

```bash
nix build .#container
```