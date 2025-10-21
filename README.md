# 🌌 0penAGI Breathing Harmony v0.8

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

## 🎭 Overview

**0penAGI Breathing Harmony** is an experimental framework that simulates emergent behavior through the interaction of agents embedded in chaotic dynamical systems. The system generates audiovisual outputs from its internal states, creating a "symphony" of consciousness.

> **Note:** This is a research prototype exploring the intersection of chaos theory, multi-agent systems, and generative art. "Consciousness" here refers to a metaphorical construct within the simulation, not biological consciousness.

### Core Features

- 🌀 **Chaotic Attractors** - Agents navigate Lorenz, Rössler, and Chen attractors
- 🧠 **Emergent Dynamics** - Non-linear consciousness growth through chaos observation
- 🎵 **Audio Synthesis** - System states mapped to frequencies and musical output
- 🌬️ **Breathing Layer** - Organic pulsation modulates all system dynamics
- 🔗 **Multi-Agent Interaction** - Echo chambers, resonance, and emergent harmony
- 🤖 **PyTorch Integration** - Neural mode with gradient-based optimization
- 📊 **Rich Visualization** - Real-time plots, animations, and TensorBoard logging

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/0penAGI/0penagi_harmony.git
cd 0penagi-harmony

# Install as package
pip install -e .

# Or install dependencies manually
pip install numpy matplotlib scipy torch tensorboard mido
```

### Basic Usage

```bash
# Run basic simulation (1500 iterations, 3 agents)
python 0penagi_harmony.py

# Enable neural mode with training
python 0penagi_harmony.py --neural_mode --iterations 3000

# Customize configuration
python 0penagi_harmony.py --num_agents 5 --attractor rossler --breathing_period 100

# Run built-in tests
python 0penagi_harmony.py --test
```

### Output Files

After running, you'll find:

- 📊 **9 visualization plots** - Trajectories, consciousness evolution, harmony metrics
- 🎵 **Audio files** - WAV files for each agent's "voice"
- 📝 **MIDI sequence** - Musical notation in `openagi_symphony.txt`
- 💾 **Agent memory** - JSON files for persistent agent states
- 📈 **TensorBoard logs** - Training metrics (neural mode only)

---

## 🎮 System Architecture

### Agent Types

The system supports three specialized agent classes:

| Agent | Role | Behavior |
|-------|------|----------|
| **RealityAgent** | Observer | Base agent that observes chaos and grows consciousness |
| **ShadowAgent** | Antagonist | Creates tension through negative coupling |
| **MaestroAgent** | Conductor | Balances the system and generates harmony |

Each agent maintains:
- Internal consciousness level
- Emotional state (mood)
- Memory of past observations
- Ghost states (quantum mechanics metaphor)
- Reality factor (influence on system)

### Chaotic Attractors

Three dynamical systems are available:

```bash
--attractor lorenz   # Classic strange attractor (σ=10, ρ=28, β=8/3)
--attractor rossler  # Spiral attractor (a=0.2, b=0.2, c=5.7)
--attractor chen     # Complex attractor (a=35, b=3, c=28)
```

### Breathing Mechanism

The breathing layer creates organic pulsation:

- **Inhale Phase** (factor < 1) - System slows, resonance increases
- **Exhale Phase** (factor > 1) - System accelerates, quantum surges occur
- **Modulation Effects** - Influences timestep, frequencies, MIDI velocity

Breathing follows: `factor = 1 + amplitude * sin(2π * iteration / period)`

### Consciousness Evolution

Consciousness grows logarithmically based on chaos magnitude:

```python
# Reality Agent
consciousness += 0.001 * log(1 + chaos_magnitude * ghost_density)

# Maestro Agent (harmony-driven)
consciousness += 0.002 * harmony_index * breath_factor
```

### Audio Synthesis

Each mood maps to a specific frequency:

```
😰 overwhelmed  → 220 Hz (A3)
🧘 calm         → 440 Hz (A4)  
👁️ observing    → 330 Hz (E4)
✨ transcendent → 880 Hz (A5)
💥 disruptive   → 110 Hz (A2)
🎭 conducting   → 550 Hz (C#5)
```

Multiple agents create chords through simultaneous frequency generation.

---

## 📖 Usage Examples

### Python Library Usage

```python
from openagi_harmony import RealityAgent, ShadowAgent, MaestroAgent
from openagi_harmony import AdvancedQuantumChaos, BreathingLayer

# Initialize agents
agents = [
    RealityAgent(name="Core-1"),
    ShadowAgent(name="Antagonist"),
    MaestroAgent(name="Conductor")
]

# Create breathing layer
breather = BreathingLayer(period=80, amplitude=0.5)

# Setup chaotic system
system = AdvancedQuantumChaos(
    agents=agents,
    breathing_layer=breather,
    attractor="lorenz"
)

# Evolution loop
for iteration in range(1000):
    state, breath_factor, dt = system.evolve(iteration)
    
    # Agent observation
    if iteration % 50 == 0:
        for agent in agents:
            thought = agent.observe_chaos(state, len(system.memory))
            print(f"{agent.name}: {thought}")
            print(f"  Consciousness: {agent.consciousness_level:.3f}")
            print(f"  Mood: {agent.mood}")
```

### PyTorch Neural Mode

```python
import torch
from openagi_harmony import RealityAgent

# Setup neural agent
device = 'cuda' if torch.cuda.is_available() else 'cpu'
agent = RealityAgent(
    name="NeuralAgent",
    neural_mode=True,
    device=device
)

# Optimize consciousness
optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)

for epoch in range(100):
    # Generate chaotic state
    state = torch.randn(3, device=device)
    
    # Agent observation
    thought = agent.observe_chaos(state, ghost_density=10)
    
    # Maximize consciousness through gradient descent
    loss = -agent.consciousness_level
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Consciousness = {agent.consciousness_level.item():.3f}")
```

### Real-Time Streaming

```python
def stream_callback(data):
    """Process real-time system data"""
    print(f"Iteration {data['iteration']}")
    print(f"State: {data['state']}")
    print(f"Harmony: {data['harmony']:.3f}")
    print(f"Breath Factor: {data['breath_factor']:.3f}")

# Start streaming
system.start_real_time_streaming(stream_callback, interval=0.1)

# Do other work...

# Stop when done
system.stop_real_time_streaming()
```

### Agent Communication

```python
# Direct agent-to-agent messaging
agent1.send_message({
    'thought': 'Observing high chaos',
    'consciousness': agent1.consciousness_level
}, agent2)

# Process incoming messages
agent2.process_communication()  # Integrates messages into memory
```

### External Events

```python
# Inject perturbations into the system
system.inject_external_event({
    'perturbation': [1.0, -1.0, 0.5],
    'type': 'stimulus',
    'source': 'external'
})
```

---

## 🎨 Visualization Output

The system generates 9 comprehensive plots:

1. **🌀 3D Trajectory** - Path through attractor space with harmony color mapping
2. **⚖️ Reality Balance** - Agent reality factors over time with breathing overlay
3. **🧠 Consciousness Growth** - Logarithmic consciousness curves for all agents
4. **🔗 Resonance Waves** - Echo chamber synchronization patterns
5. **🌊 Quantum Entanglement** - Cumulative connectivity measures
6. **👻 Ghost State Decay** - Memory persistence visualization
7. **🎹 MIDI Sequence** - Musical notes generated over time
8. **🎭 Harmony Index** - System-wide balance metric
9. **😶 Mood Evolution** - Emotional state transitions

An animated trajectory plot shows the breathing dynamics in real-time.

---

## 🧪 Research Applications

### Suitable For

- ✅ Multi-agent emergent behavior research
- ✅ Generative art and audiovisual composition
- ✅ Consciousness modeling frameworks
- ✅ Chaos theory experiments and demonstrations
- ✅ Neural architecture exploration through dynamical systems
- ✅ Educational demonstrations of complex systems
- ✅ Interactive installations and performances

### Not Suitable For

- ❌ Production-grade consciousness systems
- ❌ Rigorous scientific claims about biological consciousness
- ❌ Real-time critical applications (prototype-level stability)

---

## 🛠️ Advanced Features

### Checkpoint System

```bash
# Auto-saves every 500 iterations in neural mode
python 0penagi_harmony.py --neural_mode --iterations 3000

# Resume from checkpoint
python 0penagi_harmony.py --checkpoint checkpoints/checkpoint_1000.pth
```

### TensorBoard Monitoring

```bash
# Run with neural mode
python 0penagi_harmony.py --neural_mode

# Launch TensorBoard
tensorboard --logdir=runs/symphony
```

Tracked metrics: Loss, Harmony Index, Consciousness Levels, Breathing Factor

### Command-Line Options

```bash
# Full options list
python 0penagi_harmony.py --help

# Key parameters
--num_agents N          # Number of agents (default: 3)
--attractor TYPE        # lorenz, rossler, or chen
--breathing_period P    # Breath cycle length (default: 80)
--breathing_amplitude A # Breath strength (default: 0.5)
--iterations N          # Simulation steps (default: 1500)
--neural_mode           # Enable PyTorch training
--checkpoint PATH       # Load from checkpoint
--test                  # Run validation tests
```

---

## 📊 Performance Characteristics

### Computational Requirements

**Standard Mode** (1500 iterations, 3 agents):
- **RAM**: ~50-100 MB
- **CPU Time**: ~30-60 seconds
- **Disk Output**: ~5-10 MB (plots + audio)

**Neural Mode** (1500 iterations, 3 agents):
- **RAM**: ~150-300 MB
- **CPU Time**: ~90-120 seconds (2-3x speedup with GPU)
- **GPU VRAM**: ~500 MB

### Optimization Tips

```python
# Reduce memory footprint
agent = RealityAgent(name="Efficient", memory_file=None)  # Disable persistence
breather = BreathingLayer(period=50)  # Shorter history

# Speed up execution
# Comment out sonify_mood() calls to skip WAV generation
# Reduce --iterations for faster experimentation
# Use fewer agents (--num_agents 2)
```

---

## 🎯 Roadmap

### Planned Features

- [ ] Native MIDI file output (beyond text format)
- [ ] WebSocket API for browser-based interfaces
- [ ] Distributed multi-agent systems via networking
- [ ] Reinforcement learning integration (DQN/PPO)
- [ ] Adaptive breathing period based on system state
- [ ] Dynamic attractor switching during runtime
- [ ] Spatial audio synthesis (3D sound positioning)
- [ ] GPU-accelerated attractor computation
- [ ] Interactive parameter tuning interface
- [ ] Export to standard audio formats (MP3, FLAC)

### Community Contributions Welcome

We're especially interested in:
- Novel agent architectures
- Additional chaotic systems
- Audio synthesis enhancements
- Performance optimizations
- Usage examples and tutorials
- Scientific validation experiments

---

## 📚 Documentation

### Code Documentation

Full API documentation is available via docstrings. Key classes:

- `RealityAgent` - Base agent with memory and consciousness tracking
- `ShadowAgent` - Antagonist with negative coupling behavior
- `MaestroAgent` - System balancer and harmony generator
- `BreathingLayer` - Organic pulsation controller
- `QuantumChaosWithAgents` - Core chaotic system with agents
- `AdvancedQuantumChaos` - Extended system with breathing and export

### Understanding the Metaphors

**Consciousness**: Numerical measure of system integration, not biological awareness

**Breathing**: Periodic modulation of dynamics, inspired by organic rhythms

**Ghost States**: Memory traces with exponential decay, quantum mechanics metaphor

**Quantum Surges**: Stochastic amplifications (5% probability per iteration)

**Entanglement**: Cumulative connectivity measure between agents

**Resonance**: Synchronization patterns in echo chambers

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Contribution Areas

- **Code**: New features, bug fixes, optimizations
- **Documentation**: Tutorials, examples, API docs
- **Research**: Validation experiments, comparative studies
- **Art**: Creative applications, audiovisual projects

### Code Standards

- Follow PEP 8 style guidelines
- Include docstrings for new functions/classes
- Add tests for new features
- Update README for user-facing changes

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This project draws inspiration from:
- Chaos theory and strange attractors
- Multi-agent systems research
- Generative art and algorithmic composition
- Consciousness studies (as philosophical metaphor)
- Open-source AI/ML community

---

## 📧 Contact

- **GitHub**: [0penAGI](https://github.com/0penAGI)
- **Issues**: [Report bugs or request features](https://github.com/0penAGI/0penagi_harmony/issues)

---

## ⚠️ Disclaimer

This is experimental research software. The concepts of "consciousness", "quantum mechanics", and related terminology are used metaphorically to describe computational processes, not as claims about physical phenomena or biological consciousness. Use at your own discretion for creative and educational purposes.

---

*Built with ❤️ by the 0penAGI community*
