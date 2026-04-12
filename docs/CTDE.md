# Centralized Training with Decentralized Execution (CTDE)

The `multi_car_racing` environment provides built-in support for generating global observations suited for **Centralized Training with Decentralized Execution (CTDE)** workflows.

When the environment is initialized with `ctde=True`, the generated observations for each agent change from `(H, W, C)` to `(H, W, N*C)` where `N` is `num_agents`.

## How CTDE Channel Concatenation works

In a typical multi-agent setup, each agent relies on its own camera frame. When `ctde=True`, the environment will stack all $N$ agents' camera frames into a single deep image. It builds a global view by collecting the specific camera frame centered around every agent.

The channels are deterministically ordered for each viewing agent:

1. **Self (Ego)**
2. **Teammates** (sorted by ID)
3. **Opponents** (sorted by ID)

## $N \times N$ Rendering for Color Consistency

A robust CTDE algorithm expects observations to be invariant across equivalent perspective channels.

To satisfy this, we enforce strict **color consistency**. When an agent views its _own_ global observation stack:

- Its own car must be colored as the Ego color (e.g., Red).
- Its teammates must be colored as the Teammate color (e.g., Green).
- Its opponents must be colored as the Opponent color (e.g., Blue).

If we only rendered the game $N$ times (one for each agent's camera centering), Agent 0 stacking Agent 1's frame would see Agent 1 colored as Red, which ruins the learning semantics for Agent 0 (who sees Agent 1 as Green).

To fix this, when `ctde=True`, the environment performs **$N \times N$ renders**. For each observer agent $i$, we iterate over every other camera target agent $j$, and render the frame centered on $j$ but using the color mappings corresponding to observer $i$.

### Performance Implications

This color consistency is computationally expensive!

- If `ctde=False`, rendering step time scales as $\mathcal{O}(N)$.
- If `ctde=True`, rendering step time scales as $\mathcal{O}(N^2)$.

This overhead is primarily due to PyGame surf array generation and pixel manipulation operations performed repeatedly per-agent. When designing RL experiments, expect significant simulation slowdowns if experimenting with high agent counts (e.g., $N > 4$) accompanied by `ctde=True`.
