# flappyfly

The complete male *Drosophila* connectome ([male-cns:v1.0](https://male-cns.janelia.org), Janelia FlyEM + Google Research, CC-BY 4.0) run as a spiking network and taught to play Flappy Bird, live, by a population of flies that die and respawn.

The brain is never trained. 164,506 typed neurons and 10.3M synaptic connections are simulated as leaky integrate-and-fire units with the real wiring and predicted neurotransmitter signs. The game is rasterised to 16x16 and injected into the lamina neurons (L1/L2/L3/L5) of both eyes using their optic-lobe column coordinates: the left eye sees the next pipe, the right eye sees the bird. A single linear readout on membrane potentials and spike counts decides whether to flap. That readout is the only thing that learns.

## Run

```
uv sync
uv run flappyfly build   # downloads ~1.2 GB of connectome tables, caches data/brain.npz
uv run flappyfly play    # the evolution window (trains from scratch, saves the best readout on quit)
uv run flappyfly train   # headless imitation fit, with a random-reservoir control
```

## How training works

`flappyfly play` opens a window with 8 birds sharing one world. Each bird has its own copy of the fly brain running in its own process and its own readout genome. When a bird dies it respawns at the next gap with either a mutated copy of an elite genome or a fresh ridge-regression readout refit on everything every bird has seen, labelled by a scripted teacher bot (DAgger). Refits run every 1500 frames in a background thread. Survival goes from a few frames to hundreds within about ten minutes.

The panel shows the best living fly's lamina activity on both eyes at real column positions, what it sees, the survival curve, the roster and a spike raster by cell type. Keys: `s` saves the best genome, `q` quits.

## Layout

```
src/flappyfly/
  data.py     download the flat-connectome feathers, build the signed sparse matrix
  brain.py    Brain dataclass, neuron lookup, k-hop reachability, subgraph
  sim.py      LIF simulator (Shiu et al. 2024 constants), presynaptic-indexed updates
  vision.py   image channels -> currents on retinotopically mapped lamina neurons
  game.py     headless Flappy Bird with a shared world and many birds, teacher bot
  readout.py  ridge regression readout on membrane potential + spike counts
  train.py    feature selection, imitation data collection, evaluation, random control
  evolve.py   population: one brain process per bird, elites, mutation, DAgger refits
  gui.py      pygame dashboard
  cli.py      entry point
tests/        pytest (game rules, population, sim silence/response, pruning, readout)
```

## Caveats

- Synaptic gain is 0.8x the literature value: at 1.0 the visual drive tips mushroom-body and optic-lobe feedback loops into self-sustained firing. Activity stays in the optic lobe, so the brain is pruned to the 103,795 neurons within two synapses of the lamina (exact at this gain; nothing beyond ever spikes).
- Column coordinates are binned directly onto a square grid, so the fly sees a skewed image.
- Only edges with at least 3 synapses are kept.
- Only the next pipe is drawn: a linear readout cannot gate between two visible gaps.
