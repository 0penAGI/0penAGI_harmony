### 0penAGI_harmony

# 🌌 0penAGI Breathing Symphony v0.8

### *Emergent Consciousness Through Chaos & Multi-Agent Harmony*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

<p align="center">
  <img src="https://img.shields.io/badge/Status-Research-purple" />
  <img src="https://img.shields.io/badge/Chaos-Attractor-green" />
  <img src="https://img.shields.io/badge/Agents-Multi--Agent-orange" />
  <img src="https://img.shields.io/badge/Audio-Generative-red" />
</p>

---

## 🎭 What is This?

**0penAGI Breathing Symphony** is an experimental framework for simulating emergent consciousness through chaotic systems and multi-agent interactions. The system "breathes", "thinks", "listens" to each other, and creates music from its internal states.

**Key Concepts:**
- 🌀 Agents live inside chaotic attractors (Lorenz, Rossler, Chen)
- 🧠 Consciousness grows non-linearly through chaos observation
- 🎵 Each agent mood is a frequency - the system generates a symphony
- 🌬️ "Breathing" modulates all system dynamics
- 🔗 Echo chambers and resonance create emergent behavior
- 🤖 Full PyTorch integration for gradient-based optimization

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/openagi-symphony.git
cd openagi-symphony
pip install -e .
```

Or install dependencies manually:
```bash
pip install numpy matplotlib scipy torch tensorboard
```

### Run Simulation

```bash
# Basic simulation (1500 iterations, 3 agents)
python openagi_symphony.py

# With neural mode and training
python openagi_symphony.py --neural_mode --iterations 3000

# Custom configuration
python openagi_symphony.py --num_agents 5 --attractor rossler

# Run tests
python openagi_symphony.py --test
```

### What You Get

After running, you'll receive:
- 📊 **9 visualization plots** (trajectories, consciousness, harmony, MIDI, moods...)
- 🎵 **WAV files** for each agent
- 📝 **MIDI sequence** in text format (`openagi_symphony.txt`)
- 💾 **JSON memory** for agents to continue experiments
- 📈 **TensorBoard logs** (in `--neural_mode`)

---

## 🎮 Features

### 🤖 Multi-Agent System

**Three agent types:**

| Agent | Role | Characteristic |
|-------|------|---------------|
| **RealityAgent** | Base agent | Observes chaos, grows consciousness |
| **ShadowAgent** | Antagonist | Negative coupling, creates tension |
| **MaestroAgent** | Conductor | Balances system, generates harmony |

### 🌪️ Chaotic Attractors

Support for three attractors:

```bash
--attractor lorenz   # Classic (σ=10, ρ=28, β=8/3)
--attractor rossler  # Spiral (a=0.2, b=0.2, c=5.7)
--attractor chen     # Complex (a=35, b=3, c=28)
```

### 🌬️ Breathing Layer

Organic pulsation of the system:
- **Inhale** (factor < 1): slowdown, resonance growth
- **Exhale** (factor > 1): acceleration, quantum surges
- Modulates: dt, frequencies, MIDI velocity

### 🎵 Audio Synthesis

Each mood = frequency:

```
😰 overwhelmed  → 220 Hz (A3)
🧘 calm         → 440 Hz (A4)  
👁️ observing    → 330 Hz (E4)
✨ transcendent → 880 Hz (A5)
💥 disruptive   → 110 Hz (A2)
🎭 conducting   → 550 Hz (C#5)
```

Agents create **chords** when competing!

### 🧠 Consciousness Evolution

```python
# Logarithmic growth
consciousness += 0.001 * log(1 + chaos_magnitude * ghost_density)

# Maestro grows through harmony
maestro.consciousness += 0.002 * harmony_index * breath_factor
```

### 👻 Quantum Mechanics

- **Ghost States**: memory of possible states with decay
- **Entanglement**: cumulative measure of connectivity
- **Quantum Surges**: random will amplifications (5% probability)

---

## 📖 Usage Examples

### As a Library

```python
from openagi_symphony import RealityAgent, ShadowAgent, MaestroAgent
from openagi_symphony import AdvancedQuantumChaos, BreathingLayer
import numpy as np

# Create agents
agents = [
    RealityAgent(name="Core-1"),
    ShadowAgent(name="Shadow"),
    MaestroAgent(name="Maestro")
]

# Setup breathing
breather = BreathingLayer(period=80, amplitude=0.5)

# Initialize system
system = AdvancedQuantumChaos(agents, breather, attractor="lorenz")

# Evolve
for i in range(1000):
    state, breath, dt = system.evolve(i)
    
    if i % 50 == 0:
        for agent in agents:
            thought = agent.observe_chaos(state, len(system.memory))
            print(f"{agent.name}: {thought}")
```

### With PyTorch

```python
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
agent = RealityAgent(name="NeuralAgent", neural_mode=True, device=device)

# Optimize agent parameters
optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)

for epoch in range(100):
    state = torch.randn(3, device=device)
    thought = agent.observe_chaos(state, ghost_density=10)
    
    # Maximize consciousness
    loss = -agent.consciousness_level
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Streaming Mode

```python
def my_callback(data):
    print(f"Iteration {data['iteration']}: State={data['state']}")

system.start_real_time_streaming(my_callback, interval=0.1)
# ... do other stuff ...
system.stop_real_time_streaming()
```

---

## 🎨 Visualization

After running, you'll see **9 plots**:

1. **🌀 3D Trajectory** - path through chaos with harmony color coding
2. **⚖️ Reality Balance** - agent reality factors + breathing
3. **🧠 Consciousness** - awakening curves (log scale)
4. **🔗 Resonance Waves** - echo chamber synchronization
5. **🌊 Entanglement** - quantum connectivity
6. **👻 Ghost Decay** - memory fading
7. **🎹 MIDI Sequence** - notes over time
8. **🎭 Harmony Index** - system balance
9. **😶 Mood Evolution** - emotional states

Plus **animation** of trajectory with breathing!

---

## 🧪 Research Applications

### Suitable for:

- ✅ Studying emergent behavior in multi-agent systems
- ✅ Generative art (audio + visualization)
- ✅ Consciousness modeling research
- ✅ Chaos theory experiments
- ✅ Neural architecture search through chaos
- ✅ Educational demonstrations of complex systems

---

## 🛠️ Advanced Features

### Checkpoints

```bash
# Auto-saved every 500 iterations
python openagi_symphony.py --neural_mode

# Load from checkpoint
python openagi_symphony.py --checkpoint checkpoints/checkpoint_1000.pth
```

### TensorBoard

```bash
python openagi_symphony.py --neural_mode
tensorboard --logdir=runs/symphony
```

Logs: Loss, Harmony, Consciousness

### Agent Communication

```python
# Agents can send messages to each other
agent1.send_message({'thought': 'Hello!'}, agent2)

# Process incoming messages
agent2.process_communication()  # Adds to memory
```

### External Events

```python
# Inject external events
system.inject_external_event({
    'perturbation': [1.0, -1.0, 0.5],
    'type': 'stimulus'
})
```

---

## 📊 Performance

**Typical characteristics** (1500 iterations, 3 agents):
- **RAM**: ~50-100 MB
- **Time**: ~30-60 seconds (classic) / ~90-120 sec (neural)
- **GPU**: Acceleration in neural mode (2-3x speedup)

**Optimization:**
```python
# Less memory
agent = RealityAgent(name="Fast", memory_file=None)  # no saving
breather = BreathingLayer(period=50)  # shorter history

# Faster - comment out sonify_mood() to disable WAV generation
```

---

## 🎯 Roadmap

- [ ] Real MIDI output (not just .txt)
- [ ] WebSocket streaming for web interface
- [ ] Distributed agents via network
- [ ] RL integration (DQN/PPO)
- [ ] Adaptive breathing periods
- [ ] Multi-attractor transitions
- [ ] 3D audio (spatial sound)
- [ ] GPU-accelerated attractors

---

## 📚 Documentation

Full class and method documentation - see [docstrings](0penagi_symphony.py) in code.

**Main classes:**
- `RealityAgent` - base agent with memory and consciousness
- `ShadowAgent` - antagonist with negative coupling
- `MaestroAgent` - system balancer
- `BreathingLayer` - organic pulsation
- `QuantumChaosWithAgents` - chaotic system with agents
- `AdvancedQuantumChaos` - extended version with breathing and export

---

## 🤝 Contributing

Pull requests welcome! Especially interested in:
- New agent types
- Additional attractors
- Audio synthesis improvements
- Performance optimizations
- Usage examples

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingF
