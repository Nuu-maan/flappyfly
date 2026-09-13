# flappyfly

The complete male *Drosophila* connectome ([male-cns:v1.0](https://male-cns.janelia.org), Janelia FlyEM + Google Research, CC-BY 4.0) run as a spiking network and taught to play Flappy Bird.

The brain is never trained. 164,506 typed neurons and 10.3M synaptic connections are simulated as leaky integrate-and-fire units with the real wiring and predicted neurotransmitter signs. The game is rasterised to 16×16 and injected into the right eye's lamina neurons (L2/L3/L5) using their optic-lobe column coordinates. A single linear readout on the resulting spike trains decides whether to flap. That readout — a few thousand weights — is the only thing that learns, by imitating a scripted bot (reservoir computing).

## Run

```
uv sync
uv run flappyfly build   # downloads ~1.2 GB of connectome tables, caches data/brain.npz
uv run flappyfly train   # fits the readout; also reports a random-reservoir control
uv run flappyfly play    # live window: game left, spike raster right, `k` knocks out Tm1
```

## Layout

```
src/flappyfly/
  data.py     download the flat-connectome feathers, build the signed sparse matrix
  brain.py    Brain dataclass + neuron lookup by type / superclass / side
  sim.py      LIF simulator (Shiu et al. 2024 constants), presynaptic-indexed updates
  vision.py   16×16 image -> currents on retinotopically mapped lamina neurons
  game.py     headless Flappy Bird + teacher bot
  readout.py  leaky spike trace + ridge regression readout
  train.py    imitation data collection, evaluation, random-reservoir control
  play.py     pygame demo
  cli.py      entry point
tests/        pytest (game rules; sim silence/response/input-specificity)
```

## Caveats

- Synaptic gain is 0.8× the literature value: at 1.0 the visual drive tips mushroom-body and optic-lobe feedback loops into self-sustained firing. Activity stays mostly in the optic lobe; descending neurons are not what drives the bird.
- Column coordinates are binned directly onto a square grid, so the fly sees a skewed image.
- Only edges with ≥3 synapses are kept.
