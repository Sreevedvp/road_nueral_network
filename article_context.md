# Road AI — Teaching Cars to Drive Themselves with Neuroevolution in Rust

## The Idea

What if you could draw any road on screen, drop a fleet of tiny cars onto it, and watch them *learn* to navigate it — from scratch — right in front of your eyes?

**Road AI** is a real-time neuroevolution sandbox built entirely in Rust. You paint a road with your mouse, lay down checkpoints, and press a single key. Forty AI-driven cars spawn at the start. Most crash immediately. A few stumble a little further. The best brains survive, mutate, and breed the next generation. Within minutes, a car is threading through curves you drew seconds ago — no training data, no gradient tapes, no Python, no GPUs. Just evolution, running at 60 fps in a native window.

---

## Why This Project Exists

Most introductions to AI either drown you in theory or hand you a pre-trained model to poke at. Neither teaches you what's really happening inside a neural network. Road AI strips the problem down to its bones:

- **No frameworks.** The neural network is a ~100-line, from-scratch implementation — forward pass, backprop, mutation — all visible in a single file.
- **No datasets.** The cars generate their own experience by driving (and crashing). Evolution is the only teacher.
- **Instant feedback.** You see natural selection play out in real time. Bad brains die. Good brains reproduce. Understanding clicks when you *watch* it happen.

The goal was to build the simplest possible system where a neural network visibly learns a non-trivial task, and to do it in Rust to prove that "AI" doesn't require a 200 MB Python environment.

---

## How It Works — The 30-Second Version

1. **You draw a road** (left-click drag) on a 1000 x 700 canvas.
2. **You place checkpoints** (right-click) — a start, waypoints, and a finish.
3. **Press T** to train. 40 cars spawn, each with a randomised neural-network brain.
4. Cars that reach more checkpoints score higher. Dead cars are replaced by mutated copies of the best.
5. **Press V** to test the best brain so far — frozen weights, no mutation, pure inference.

---

## Technical Architecture

### The Stack

| Layer | Technology | Role |
|---|---|---|
| Language | **Rust (2024 edition)** | Performance, safety, zero-cost abstractions |
| Graphics | **macroquad 0.4** | Immediate-mode 2D rendering, input handling, game loop |
| AI | **Custom neural net** | Hand-rolled forward pass, backprop, and evolutionary mutation |

The entire project is a single `main.rs` file (~612 lines) with zero runtime dependencies beyond macroquad. No serialisation libraries, no linear-algebra crates, no ML frameworks.

### The Neural Network

Each car carries a compact **feedforward neural network** (a `Brain`):

```
Architecture:  11 inputs -> 8 hidden (tanh) -> 2 outputs (tanh)
Total weights: 8 x (11 + 1) + 2 x (8 + 1) = 96 + 18 = 114 parameters
```

**Inputs (11):**

- 8 distance sensors fanning out from -80 deg to +80 deg relative to the car's heading (ray-cast against the road edges)
- 2 direction components (forward/right dot products) pointing toward the next checkpoint
- 1 normalised distance to the next checkpoint

**Outputs (2):**

- **Steering** (-1 to +1): mapped to a turning curvature
- **Throttle** (0 to 1): mapped to a speed up to 120 px/s, modulated by clearance, turning, and proximity to checkpoints

### The Driving Physics

Cars aren't just dots sliding on rails. The simulation includes:

- **Curvature-based steering** — the car follows an arc, not a straight line, during turns.
- **Sub-step collision detection** — each frame is divided into small sub-steps to prevent tunnelling through thin walls.
- **Speed modulation** — the car automatically slows for tight turns, low front clearance, and checkpoint approach, keeping the physics stable and the behaviour believable.
- **Body-aware collision** — 12 points around the car's radius are checked, not just the centre, so it can't clip through walls.

### Evolution Strategy

The training loop uses a **(mu + lambda) evolutionary strategy**:

1. **Population:** 40 cars per generation.
2. **Selection:** When all cars have crashed or timed out, the top 5 are chosen as parents.
3. **Elitism:** The all-time best brain is always copied unchanged into slot 0. The top-5 brains of the current generation fill slots 1-4.
4. **Mutation:** The remaining ~32 cars are mutated clones of a random parent — each weight has a 25% chance of being nudged by up to +/-0.35, clamped to [-5, +5].
5. **Immigration:** The last 4 slots are filled with completely random brains to maintain diversity.
6. **Fitness:** `(checkpoints reached x 1000) + (progress toward next x 900)`. Finishing the course awards a large bonus plus a time-based reward.

### Optional Backpropagation

The `Brain` also implements a full `learn()` method — manual backpropagation through the two-layer network. This was designed so a human player could drive (using keyboard controls) and the network could be supervised-trained on those examples, combining imitation learning with evolution. The infrastructure is in place even though the primary training mode is purely evolutionary.

---

## The User Experience

Road AI runs in three modes, toggled by single keypresses:

| Key | Mode | What Happens |
|---|---|---|
| **E** | Edit | Draw roads, place/remove checkpoints. The best brain is preserved in memory. |
| **T** | Train | Spawn 40 cars. Watch evolution in real time. Hold **Space** to run 8x faster. |
| **V** | Test | Deploy the best brain on the current road. Weights are frozen — pure inference. |
| **C** | Clear | Wipe the canvas but keep the learned brain, so it can be tested on a new road. |

The HUD shows the current generation, best fitness score, and number of cars still alive. Route checkpoints are colour-coded (green = start, yellow = waypoints, red = finish).

---

## Key Design Decisions

1. **Single-file architecture.** Everything — physics, rendering, neural network, evolution, UI — lives in one file. This is intentional: the project is meant to be read top-to-bottom as a learning resource. No hunting through module trees.

2. **No external ML crate.** Writing the forward pass and backprop by hand makes every matrix multiply visible. Readers can trace exactly how an input sensor reading becomes a steering command.

3. **Macroquad over Bevy/ggez.** Macroquad's immediate-mode API keeps the rendering code trivially simple — `draw_circle`, `draw_line`, `draw_text`. No ECS, no asset pipelines, no boilerplate.

4. **Pixel-based road detection.** The road is just a bitmap. Black pixels = road, white pixels = wall. This makes "draw your own track" trivially cheap — no polygon math, no tile maps. Sensor ray-casts simply walk pixels until they hit white.

5. **Brain persistence across maps.** Pressing **C** clears the canvas but keeps the best brain. This lets you test transfer learning: train on a curvy road, wipe it, draw a completely different one, and see how the brain adapts (or doesn't).

---

## What You'll Learn from This Project

- How a **feedforward neural network** works at the lowest level — weights, biases, activation functions, forward pass.
- How **neuroevolution** (genetic algorithms applied to neural networks) can solve control problems without labelled data.
- How **backpropagation** works from first principles — the `learn()` method is a clean, minimal implementation.
- How to build a **real-time simulation** in Rust with sub-stepping, ray-casting, and immediate-mode rendering.
- How surprisingly little code it takes to make something that *looks* intelligent.

---

## Running It

```bash
# Clone and run
git clone <repo-url>
cd road_ai
cargo run --release
```

**Requirements:** Rust toolchain (stable), a GPU that supports OpenGL 3 (virtually anything modern). No Python, no CUDA, no pip install.

---

## Future Directions

- **Crossover:** Implementing sexual reproduction (recombining weights from two parents) alongside mutation.
- **Speciation:** NEAT-style speciation to protect innovative topologies from being outcompeted before they mature.
- **Replay and export:** Serialise the best brain's weights so it can be loaded across sessions.
- **Human-in-the-loop training:** Fully wire up keyboard driving to supervised `learn()` calls, creating a hybrid imitation + evolution pipeline.
- **Multiplayer evolution:** Run multiple populations on different tracks and merge the best performers.

---

*Road AI is a proof that you don't need TensorFlow, a GPU cluster, or a PhD to build something that learns. 114 parameters. 612 lines of Rust. One file. Draw a road, and watch intelligence emerge.*
