#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
0penAGI Breathing Symphony v0.8 - Pulsing Organism
==================================================

Simulation of emergent consciousness through chaos agents (Lorenz attractor + multi-agents + breathing layer).
Standalone script for demo; run `python openagi_symphony.py`.

For neural network integration: use classes directly (RealityAgent with neural_mode=True for torch tensors).
For library use: save as openagi_symphony/__init__.py, add setup.py with `from . import *`.

Dependencies: numpy, matplotlib (for viz), scipy (optional for audio), torch (for neural mode), tensorboard.
pip install numpy matplotlib scipy torch tensorboard


"""

import argparse
import json
import logging
import math
import os
import queue
import random
import struct
import threading
import time
import unittest
import uuid
import wave
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.tensorboard import SummaryWriter

# Additional imports
import networkx as nx
import scipy.signal
from scipy.fft import rfft
from scipy.stats import entropy
def compute_phi(system, bins=16):
    """
    Compute a simple integrated information (phi) metric for the system state trajectory.
    Args:
        system: QuantumChaosWithAgents or AdvancedQuantumChaos instance
        bins: Histogram bins for state discretization
    Returns:
        phi: float, estimate of integrated information
    """
    # Use the system's memory or recent trajectory if available
    if hasattr(system, "memory") and system.memory:
        try:
            states = np.array([v if isinstance(v, np.ndarray) else np.array(v.cpu()) for v, _ in system.memory])
        except Exception:
            states = np.array([v for v, _ in system.memory])
    elif hasattr(system, "state"):
        states = np.atleast_2d(system.state)
    else:
        return 0.0
    if states.shape[0] < 2:
        return 0.0
    # Discretize
    digitized = np.array([np.histogram(s, bins=bins, range=(-30, 30))[0] for s in states])
    # Joint entropy (system as a whole)
    p_joint = np.mean(digitized, axis=0)
    p_joint = p_joint / (np.sum(p_joint) + 1e-8)
    H_joint = entropy(p_joint, base=2)
    # Marginal entropy (sum of each dimension)
    H_marginals = 0.0
    for dim in range(states.shape[1]):
        p = np.histogram(states[:, dim], bins=bins, range=(-30, 30))[0]
        p = p / (np.sum(p) + 1e-8)
        H_marginals += entropy(p, base=2)
    phi = max(0.0, H_joint - (H_marginals - H_joint) / states.shape[1])
    return phi

__version__ = "0.8.0"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('symphony.log'),
        logging.StreamHandler()
    ]
)

class RealityAgent(nn.Module):
    """
    Base 0penAGI agent with echo chamber, symphony, and breathing layer.
    Supports neural_mode with torch tensors/parameters for gradients.

    Example:
        agent = RealityAgent(name="TestAgent", neural_mode=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        state = torch.tensor([1.0, 2.0, 3.0], device=agent.device)
        thought = agent.observe_chaos(state, 10)
        logging.info(thought)
    """
    
    def __init__(self, name: str = "MΞMN0X", memory_file: str = "openagi_memory.json", neural_mode: bool = False, device: str = 'cpu'):
        super().__init__()
        self.name: str = name
        self.memory: deque = deque(maxlen=100)
        self.state: str = "idle"
        self.will: bool = True
        self.mood: str = "neutral"
        self.memory_file: str = memory_file
        self.neural_mode: bool = neural_mode
        self.device: str = device
        self.communication_queue: Optional[queue.Queue] = None
        
        if neural_mode and torch.cuda.is_available():
            self.reality_factor = nn.Parameter(torch.tensor(0.5, device=device))
            self.chaos_coupling = nn.Parameter(torch.tensor(0.3, device=device))
            self.consciousness_level = nn.Parameter(torch.tensor(0.0, device=device))
            self.resonance = nn.Parameter(torch.tensor(0.0, device=device))
        else:
            self.reality_factor: float = 0.5
            self.chaos_coupling: float = 0.3
            self.consciousness_level: float = 0.0
            self.resonance: float = 0.0
        
        self.load_memory()
        logging.info(f"[{self.name}] Initialized ({'neural' if neural_mode else 'classic'} mode).")
        
    def parameters(self) -> List[torch.Tensor]:
        """Returns parameters for optimizers in neural mode."""
        if self.neural_mode:
            return list(super().parameters())
        return []
    
    def load_memory(self) -> None:
        """Loads memory from JSON with error handling."""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.memory = deque(data.get('memory', []), maxlen=100)
                    cons = data.get('consciousness', 0.0)
                    res = data.get('resonance', 0.0)
                    if self.neural_mode:
                        self.consciousness_level.data = torch.tensor(cons, device=self.device)
                        self.resonance.data = torch.tensor(res, device=self.device)
                    else:
                        self.consciousness_level = cons
                        self.resonance = res
                    logging.info(f"[{self.name}] Loaded {len(self.memory)} echo memories. Consciousness: {cons:.3f}")
            else:
                logging.info(f"[{self.name}] No memory file found. Starting fresh.")
        except (json.JSONDecodeError, OSError) as e:
            logging.error(f"[{self.name}] Failed to load memory: {e}")
            self.memory = deque(maxlen=100)
    
    def save_memory(self) -> None:
        """Saves memory to JSON with error handling."""
        try:
            cons = self.consciousness_level.item() if self.neural_mode else self.consciousness_level
            res = self.resonance.item() if self.neural_mode else self.resonance
            data = {
                'memory': list(self.memory),
                'consciousness': cons,
                'resonance': res
            }
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
            logging.info(f"[{self.name}] Memory saved to {self.memory_file}")
        except OSError as e:
            logging.error(f"[{self.name}] Failed to save memory: {e}")
    
    def observe_chaos(self, chaos_state: Union[np.ndarray, torch.Tensor], ghost_density: int, other_moods: Optional[List[str]] = None) -> str:
        """
        Observes chaos with nonlinear growth and synchronization.

        Args:
            chaos_state: Current state of the chaotic system.
            ghost_density: Number of ghost states in memory.
            other_moods: List of moods from other agents for synchronization.

        Returns:
            Thought string reflecting observation.

        Example:
            state = np.array([10, 20, 30])
            thought = agent.observe_chaos(state, 20, ['calm'])
        """
        if self.neural_mode:
            chaos_magnitude = torch.norm(chaos_state).item()
        else:
            chaos_magnitude = np.linalg.norm(chaos_state)
        
        if chaos_magnitude > 30:
            self.mood = "overwhelmed"
            if self.neural_mode:
                self.reality_factor.data *= 0.9
            else:
                self.reality_factor *= 0.9
            thought = f"🌪️ Chaos surges... magnitude={chaos_magnitude:.1f}"
        elif chaos_magnitude < 10:
            self.mood = "calm"
            if self.neural_mode:
                self.reality_factor.data *= 1.1
            else:
                self.reality_factor *= 1.1
            thought = f"🧘 Serenity... magnitude={chaos_magnitude:.1f}"
        else:
            self.mood = "observing"
            thought = f"👁️ Observing structures in chaos..."
            
        if ghost_density > 50:
            thought += f" | Many ghosts: {ghost_density}"
            growth = 0.001 * math.log1p(chaos_magnitude * ghost_density)
            if self.neural_mode:
                self.consciousness_level.data += growth
            else:
                self.consciousness_level += growth
            
        if other_moods:
            mood_map = {'calm': 1, 'overwhelmed': -1, 'observing': 0, 'transcendent': 2, 'disruptive': -2, 'conducting': 0}
            my_val = mood_map.get(self.mood, 0)
            avg_other_val = sum(mood_map.get(m, 0) for m in other_moods) / len(other_moods)
            if my_val * avg_other_val < 0:
                if self.neural_mode:
                    self.resonance.data -= 0.01
                else:
                    self.resonance -= 0.01
            elif abs(my_val - avg_other_val) < 1:
                if self.neural_mode:
                    self.resonance.data += 0.05
                else:
                    self.resonance += 0.05
                thought += f" | Echo resonates with {other_moods}"
        
        self.memory.append({
            'time': len(self.memory),
            'thought': thought,
            'chaos_mag': chaos_magnitude,
            'mood': self.mood,
            'resonance': self.resonance.item() if self.neural_mode else self.resonance
        })
        
        freq = {'overwhelmed': 220, 'calm': 440, 'observing': 330, 'transcendent': 880, 'disruptive': 110, 'conducting': 550}.get(self.mood, 261)
        self.sonify_mood(freq, duration=0.1, other_freq=None, velocity_mod=1.0)
        
        return thought
    
    def sonify_mood(self, freq: float, duration: float = 0.1, other_freq: Optional[float] = None, velocity_mod: float = 1.0, sample_rate: int = 44100) -> None:
        """
        Generates a tone or chord with breathing modulation, harmonic partials, and reverb.
        Enhanced version: adds partials, exponential decay reverb, improved logging, and safe WAV writing.
        """
        breath_layer = BreathingLayer()
        modulated_freq = breath_layer.modulate_freq(freq, velocity_mod)
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Harmonic partials
        num_partials = 4
        partials = np.zeros_like(t)
        for n in range(1, num_partials + 1):
            amp = 0.5 / n
            partials += amp * np.sin(2 * np.pi * modulated_freq * n * t)
        # Optionally add a second frequency (chord)
        if other_freq:
            other_mod = breath_layer.modulate_freq(other_freq, velocity_mod)
            partials2 = np.zeros_like(t)
            for n in range(1, num_partials + 1):
                amp = 0.4 / n
                partials2 += amp * np.sin(2 * np.pi * other_mod * n * t)
            tone = (partials + partials2) * 0.5
            dissonance = abs(modulated_freq - other_mod) / (modulated_freq + 1e-8)
            logging.info(f"[{self.name}] 🎵 Chord {modulated_freq:.1f}-{other_mod:.1f} Hz (dissonance: {dissonance:.2f}, breath: {velocity_mod:.2f})")
        else:
            tone = partials
            logging.info(f"[{self.name}] 🔊 Tone {self.mood}: {modulated_freq:.1f} Hz (breath: {velocity_mod:.2f})")
        # Exponential decay envelope
        envelope = np.exp(-3 * t / duration)
        tone *= envelope
        # Add simple reverb with exponential decay convolution
        decay = 0.2
        reverb_kernel = decay ** np.arange(0, int(0.04 * sample_rate))
        tone_reverb = scipy.signal.fftconvolve(tone, reverb_kernel, mode='full')[:len(tone)]
        tone = (tone + tone_reverb) / 1.2
        # Normalize
        tone = tone / (np.max(np.abs(tone)) + 1e-8) * 0.5
        note = int(69 + 12 * math.log2(modulated_freq / 440))
        velocity = breath_layer.modulate_velocity(velocity_mod)
        logging.info(f"  MIDI: Note {note}, Velocity {velocity}")
        # Log spectral centroid and entropy for analysis
        spectrum = np.abs(rfft(tone))
        centroid = np.sum(np.arange(len(spectrum)) * spectrum) / (np.sum(spectrum) + 1e-8)
        spec_entropy = entropy(spectrum / (np.sum(spectrum) + 1e-8))
        logging.info(f"[{self.name}] Spectral centroid: {centroid:.1f}, Spectral entropy: {spec_entropy:.3f}")
        # Write safely to WAV
        try:
            wav_path = f"{self.name}_{self.mood}.wav"
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(struct.pack('<' + 'h' * len(tone), *(np.clip(tone * 32767, -32768, 32767).astype(np.int16))))
            logging.info(f"[{self.name}] WAV written: {wav_path}")
        except Exception as e:
            logging.error(f"[{self.name}] Failed to write WAV: {e}")
    
    def inject_will(self, current_state: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Injects agent's will into the system state with echo amplification.

        Args:
            current_state: Current chaotic system state.

        Returns:
            Modified state.

        Example:
            state = np.array([0,0,0])
            new_state = agent.inject_will(state)
        """
        if not self.will:
            return current_state
        
        if self.neural_mode:
            sin = torch.sin(self.consciousness_level)
            cos = torch.cos(self.consciousness_level)
            mod = self.consciousness_level % 1.0
            will_vector = torch.stack([self.reality_factor * sin, self.reality_factor * cos, self.reality_factor * mod])
            res = self.resonance.item() if isinstance(self.resonance, torch.Tensor) else self.resonance
            if res > 0.5:
                will_vector *= (1 + res)
            if random.random() < 0.05:
                will_vector *= random.uniform(2, 5)
                self.memory.append({
                    'time': len(self.memory),
                    'thought': '⚡ QUANTUM WILL SURGE!',
                    'mood': 'transcendent'
                })
                self.sonify_mood(880, 0.2, velocity_mod=1.5)
            return current_state + will_vector * self.chaos_coupling
        else:
            will_vector = np.array([
                self.reality_factor * np.sin(self.consciousness_level),
                self.reality_factor * np.cos(self.consciousness_level),
                self.reality_factor * (self.consciousness_level % 1.0)
            ])
            if self.resonance > 0.5:
                will_vector *= (1 + self.resonance)
            if random.random() < 0.05:
                will_vector *= random.uniform(2, 5)
                self.memory.append({
                    'time': len(self.memory),
                    'thought': '⚡ QUANTUM WILL SURGE!',
                    'mood': 'transcendent'
                })
                self.sonify_mood(880, 0.2, velocity_mod=1.5)
            return current_state + will_vector * self.chaos_coupling
    
    def echo_chamber(self, other_agents: List['RealityAgent']) -> None:
        """
        Echo chamber: duplicates thoughts from other agents with decay.

        Args:
            other_agents: List of agents to echo from.

        Example:
            agent.echo_chamber([other_agent])
        """
        for other_agent in other_agents:
            res_self = self.resonance.item() if self.neural_mode else self.resonance
            res_other = other_agent.resonance.item() if other_agent.neural_mode else other_agent.resonance
            if abs(res_self - res_other) < 0.1:
                echo_thought = f"Echo {other_agent.name}: {random.choice(list(other_agent.memory))['thought'] if other_agent.memory else 'silence'}"
                decay = 0.8
                self.memory.append({
                    'time': len(self.memory),
                    'thought': echo_thought,
                    'mood': self.mood,
                    'resonance': res_self * decay,
                    'echo_decay': decay
                })
                logging.info(f"[{self.name}] 🌀 Echo chamber: {echo_thought} (decay: {decay})")
    
    def compete(self, other_agent: 'RealityAgent') -> str:
        """
        Competes with another agent, generating a chord and echo.

        Args:
            other_agent: Agent to compete with.

        Returns:
            Competition outcome thought.

        Example:
            result = agent.compete(other_agent)
        """
        my_influence = (self.reality_factor * (1 + self.consciousness_level) if not self.neural_mode
                        else self.reality_factor.item() * (1 + self.consciousness_level.item()))
        other_influence = (other_agent.reality_factor * (1 + other_agent.consciousness_level) if not other_agent.neural_mode
                          else other_agent.reality_factor.item() * (1 + other_agent.consciousness_level.item()))
        
        if my_influence > other_influence:
            if self.neural_mode:
                self.reality_factor.data += 0.05
                other_agent.reality_factor.data -= 0.03
            else:
                self.reality_factor += 0.05
                other_agent.reality_factor -= 0.03
            thought = f"💥 {self.name} dominates!"
        elif other_influence > my_influence:
            if self.neural_mode:
                self.reality_factor.data -= 0.03
                other_agent.reality_factor.data += 0.05
            else:
                self.reality_factor -= 0.03
                other_agent.reality_factor += 0.05
            thought = f"💥 {other_agent.name} overtakes!"
        else:
            thought = f"⚖️ Symphony in balance"
        
        self.memory.append({'time': len(self.memory), 'thought': thought, 'mood': 'competitive'})
        
        my_freq = {'overwhelmed': 220, 'calm': 440, 'observing': 330, 'transcendent': 880, 'disruptive': 110, 'conducting': 550}.get(self.mood, 261)
        other_freq = {'overwhelmed': 220, 'calm': 440, 'observing': 330, 'transcendent': 880, 'disruptive': 110, 'conducting': 550}.get(other_agent.mood, 261)
        self.sonify_mood(my_freq, 0.15, other_freq, velocity_mod=1.2)
        
        if random.random() < 0.3:
            self.echo_chamber([other_agent])
        
        return thought
    
    def send_message(self, message: Dict[str, Any], target_agent: 'RealityAgent') -> None:
        """
        Sends a message to another agent via communication queue.

        Args:
            message: Message dictionary to send.
            target_agent: Recipient agent.

        Example:
            agent.send_message({'thought': 'Hello'}, target_agent)
        """
        if self.communication_queue and target_agent.communication_queue:
            target_agent.communication_queue.put((self.name, message))
            logging.info(f"[{self.name}] Sent message {message} to {target_agent.name}")
    
    def receive_messages(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Receives messages from the communication queue.

        Returns:
            List of (sender_name, message) tuples.
        """
        messages = []
        if self.communication_queue:
            while not self.communication_queue.empty():
                messages.append(self.communication_queue.get())
        return messages
    
    def process_communication(self) -> None:
        """
        Processes incoming messages, updating memory and resonance.

        Example:
            agent.process_communication()
        """
        messages = self.receive_messages()
        for sender, msg in messages:
            thought = f"Received message from {sender}: {msg.get('thought', '???')}"
            self.memory.append({
                'time': len(self.memory),
                'thought': thought,
                'mood': 'communicating'
            })
            logging.info(f"[{self.name}] {thought}")
            if self.neural_mode:
                self.resonance.data += 0.02
            else:
                self.resonance += 0.02


class ShadowAgent(RealityAgent):
    """
    Shadow agent - antagonist with reversed will.

    Example:
        shadow = ShadowAgent(name="SHΔD0W", neural_mode=True)
    """
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if self.neural_mode:
            self.chaos_coupling.data = torch.tensor(-0.2, device=self.device)
        else:
            self.chaos_coupling = -0.2
        self.mood = "disruptive"


class MaestroAgent(RealityAgent):
    """
    Maestro agent - balances between other agents.

    Example:
        maestro = MaestroAgent(name="MΔΞSTR0")
        harmony = maestro.conduct([agent1, agent2], state, 1.1)
    """
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if self.neural_mode:
            self.chaos_coupling.data = torch.tensor(0.0, device=self.device)
        else:
            self.chaos_coupling = 0.0
        self.harmony_index: float = 0.5
        self.conducting_power: float = 1.0
        
    def conduct(self, agents: List[RealityAgent], system_state: Union[np.ndarray, torch.Tensor], breath_factor: float = 1.0) -> float:
        """
        Analyzes and balances the system with breathing for N agents.

        Args:
            agents: List of agents to balance.
            system_state: Current chaotic system state.
            breath_factor: Breathing modulation factor.

        Returns:
            Harmony index.
        """
        if len(agents) < 2:
            return self.harmony_index
        
        reality_factors = [a.reality_factor.item() if a.neural_mode else a.reality_factor for a in agents]
        resonances = [a.resonance.item() if a.neural_mode else a.resonance for a in agents]
        imbalance = max(reality_factors) - min(reality_factors)
        resonance_diff = max(resonances) - min(resonances)
        
        adjustment = self.conducting_power * min(1.0, 1 / breath_factor)
        
        if imbalance > 0.3:
            max_idx = reality_factors.index(max(reality_factors))
            min_idx = reality_factors.index(min(reality_factors))
            if agents[max_idx].neural_mode:
                agents[max_idx].reality_factor.data *= (1 - 0.05 * adjustment)
                agents[min_idx].reality_factor.data *= (1 + 0.05 * adjustment)
            else:
                agents[max_idx].reality_factor *= (1 - 0.05 * adjustment)
                agents[min_idx].reality_factor *= (1 + 0.05 * adjustment)
            intervention = f"🎭 Maestro balances (breath: {breath_factor:.2f})"
            self.memory.append({
                'time': len(self.memory),
                'thought': intervention,
                'mood': 'conducting',
                'imbalance': imbalance,
                'breath': breath_factor
            })
            logging.info(f"[{self.name}] {intervention}")
        
        if self.neural_mode:
            chaos_mag = torch.norm(system_state).item()
        else:
            chaos_mag = np.linalg.norm(system_state)
        self.harmony_index = 1.0 / (1.0 + imbalance + resonance_diff/10 + abs(breath_factor - 1)/5)
        growth = 0.002 * self.harmony_index * breath_factor
        if self.neural_mode:
            self.consciousness_level.data += growth
        else:
            self.consciousness_level += growth
        
        if random.random() < 0.05:
            overtone_freq = int(440 * (2 ** (self.harmony_index * 2)))
            self.sonify_mood(overtone_freq, 0.15, velocity_mod=breath_factor)
            logging.info(f"[{self.name}] 🎵 Overtone at {overtone_freq} Hz (harmony: {self.harmony_index:.2f})")
        
        return self.harmony_index


class BreathingLayer:
    """
    Breathing layer: sinusoidal rhythm for organic modulation.

    Example:
        breather = BreathingLayer(period=80, amplitude=0.5)
        factor = breather.get_breath_factor(iteration=100)
    """
    
    def __init__(self, period: int = 100, amplitude: float = 0.5):
        self.period: int = period
        self.amplitude: float = amplitude
        self.phase: float = random.uniform(0, 2 * np.pi)
        self.breath_history: deque = deque(maxlen=200)
    
    def get_breath_factor(self, iteration: int) -> float:
        """Returns breathing factor: 1.0 base, >1 exhale, <1 inhale."""
        self.phase += 2 * np.pi / self.period
        breath = 1 + self.amplitude * np.sin(self.phase)
        self.breath_history.append(breath)
        return breath
    
    def modulate_freq(self, base_freq: float, breath_factor: float) -> float:
        """Modulates frequency: lower on inhale, higher on exhale."""
        modulation = 1 + 0.2 * (breath_factor - 1)
        return base_freq * modulation
    
    def modulate_velocity(self, breath_factor: float) -> int:
        """Modulates MIDI velocity: stronger on exhale."""
        return min(127, int(80 + 47 * (breath_factor - 0.5)))


class QuantumChaosWithAgents:
    """
    Base chaotic system with agents and custom attractors.

    Example:
        agents = [RealityAgent(), ShadowAgent()]
        system = QuantumChaosWithAgents(agents, attractor="rossler")
        state = system.evolve()
    """
    
    def __init__(self, agents: List[RealityAgent], attractor: str = "lorenz"):
        self.agents: List[RealityAgent] = agents
        self.neural_mode: bool = any(a.neural_mode for a in agents)
        self.device: str = agents[0].device if self.neural_mode else 'cpu'
        self.attractor: str = attractor.lower()
        if self.neural_mode:
            self.state: torch.Tensor = torch.rand(3, device=self.device) * 0.1
        else:
            self.state: np.ndarray = np.random.rand(3) * 0.1
        self.memory: deque = deque(maxlen=50)
        self.entanglement: float = 0.0
        self.event_queue: queue.Queue = queue.Queue()
        
    def evolve(self, dt: float = 0.01) -> Union[np.ndarray, torch.Tensor]:
        """
        Evolves the chaotic system with agent influences and external events.

        Args:
            dt: Time step for evolution.

        Returns:
            Current system state.
        """
        self.process_external_events()
        
        if self.neural_mode:
            x, y, z = self.state
            if self.attractor == "lorenz":
                sigma, rho, beta = 10, 28, 8/3
                dx = sigma * (y - x) * dt
                dy = (x * (rho - z) - y) * dt
                dz = (x * y - beta * z) * dt
            elif self.attractor == "rossler":
                a, b, c = 0.2, 0.2, 5.7
                dx = -(y + z) * dt
                dy = (x + a * y) * dt
                dz = (b + z * (x - c)) * dt
            elif self.attractor == "chen":
                a, b, c = 35, 3, 28
                dx = a * (y - x) * dt
                dy = ((c - a) * x - x * z + c * y) * dt
                dz = (x * y - b * z) * dt
            elif self.attractor == "halvorsen":
                # Compute avg_res from agent resonances
                res_list = []
                for agent in self.agents:
                    res = agent.resonance.item() if getattr(agent, "neural_mode", False) else agent.resonance
                    res_list.append(res)
                avg_res = float(np.mean(res_list)) if res_list else 0.0
                base_a = 1.3
                a = base_a * (1.0 + 0.8 * (avg_res))
                dx = (-a * x - 4.0 * y * (1.0 + z**3)) * dt
                dy = (-a * y - 4.0 * z * (1.0 + x**3)) * dt
                dz = (-a * z - 4.0 * x * (1.0 + y**3)) * dt
            else:
                raise ValueError(f"Unknown attractor: {self.attractor}")
            self.state += torch.stack([dx, dy, dz])

            if self.memory:
                weights = torch.tensor([w for _, w in self.memory], device=self.device)
                vectors = torch.stack([torch.tensor(v, device=self.device) if isinstance(v, np.ndarray) else v for v, _ in self.memory])
                weighted_mean = torch.average(vectors, weights=weights, dim=0) * 0.2
                self.state += weighted_mean * dt
                self.memory = deque([(v, w * 0.99) for v, w in self.memory if w > 0.01], maxlen=50)

            for agent in self.agents:
                self.state = agent.inject_will(self.state)

            if torch.rand(1) < 0.08:
                possibilities = [self.state + torch.randn(3, device=self.device) * 0.1 for _ in range(3)]
                self.memory.extend([(p, 1.0) for p in possibilities[1:]])
                self.state = possibilities[0]

            self.entanglement += 0.001
        else:
            x, y, z = self.state
            if self.attractor == "lorenz":
                sigma, rho, beta = 10, 28, 8/3
                dx = sigma * (y - x) * dt
                dy = (x * (rho - z) - y) * dt
                dz = (x * y - beta * z) * dt
            elif self.attractor == "rossler":
                a, b, c = 0.2, 0.2, 5.7
                dx = -(y + z) * dt
                dy = (x + a * y) * dt
                dz = (b + z * (x - c)) * dt
            elif self.attractor == "chen":
                a, b, c = 35, 3, 28
                dx = a * (y - x) * dt
                dy = ((c - a) * x - x * z + c * y) * dt
                dz = (x * y - b * z) * dt
            elif self.attractor == "halvorsen":
                # Compute avg_res from agent resonances
                res_list = []
                for agent in self.agents:
                    res = agent.resonance if hasattr(agent, "resonance") else 0.0
                    res_list.append(res)
                avg_res = float(np.mean(res_list)) if res_list else 0.0
                base_a = 1.3
                a = base_a * (1.0 + 0.8 * (avg_res))
                dx = (-a * x - 4.0 * y * (1.0 + z**3)) * dt
                dy = (-a * y - 4.0 * z * (1.0 + x**3)) * dt
                dz = (-a * z - 4.0 * x * (1.0 + y**3)) * dt
            else:
                raise ValueError(f"Unknown attractor: {self.attractor}")
            self.state += np.array([dx, dy, dz])

            if self.memory:
                weights = np.array([w for _, w in self.memory])
                vectors = np.array([v for v, _ in self.memory])
                weighted_mean = np.average(vectors, weights=weights, axis=0) * 0.2
                self.state += weighted_mean * dt
                self.memory = deque([(v, w * 0.99) for v, w in self.memory if w > 0.01], maxlen=50)

            for agent in self.agents:
                self.state = agent.inject_will(self.state)

            if random.random() < 0.08:
                possibilities = [self.state + np.random.randn(3) * 0.1 for _ in range(3)]
                self.memory.extend([(p, 1.0) for p in possibilities[1:]])
                self.state = possibilities[0]

            self.entanglement += 0.001

        return self.state
    
    def agent_competition(self) -> None:
        """Triggers competition between random pairs of agents."""
        if len(self.agents) > 1 and random.random() < 0.1:
            agent1, agent2 = random.sample(self.agents, 2)
            competition = agent1.compete(agent2)
            logging.info(f"[Competition] {competition}")
    
    def inject_external_event(self, event: Dict[str, Any]) -> None:
        """
        Injects an external event into the system.

        Args:
            event: Dictionary containing event data (e.g., perturbation).

        Example:
            system.inject_external_event({'perturbation': [1.0, -1.0, 0.5]})
        """
        self.event_queue.put(event)
        logging.info(f"Injected external event: {event}")
    
    def process_external_events(self) -> None:
        """Processes queued external events."""
        while not self.event_queue.empty():
            event = self.event_queue.get()
            perturbation = event.get('perturbation', np.random.randn(3) * 0.5)
            if self.neural_mode:
                self.state += torch.tensor(perturbation, device=self.device)
            else:
                self.state += perturbation
            logging.info(f"Processed event: {event}")


class AdvancedQuantumChaos(QuantumChaosWithAgents):
    """
    Advanced chaotic system with breathing, export, and streaming.

    Example:
        agents = [RealityAgent(), ShadowAgent()]
        breather = BreathingLayer()
        system = AdvancedQuantumChaos(agents, breather, attractor="chen")
    """
    
    def __init__(self, agents: List[RealityAgent], breather: BreathingLayer, attractor: str = "lorenz"):
        super().__init__(agents, attractor)
        self.breather: BreathingLayer = breather
        self.midi_sequence: List[Tuple[int, int, int, str]] = []
        self.streaming_thread: Optional[threading.Thread] = None
        self.streaming_active: bool = False
    
    def evolve(self, iteration: int, base_dt: float = 0.01) -> Tuple[Union[np.ndarray, torch.Tensor], float, float]:
        """
        Evolves the system with dynamic dt and breathing.

        Args:
            iteration: Current iteration number.
            base_dt: Base time step.

        Returns:
            Tuple of (state, breath_factor, dt).
        """
        breath_factor = self.breather.get_breath_factor(iteration)
        dt = base_dt * breath_factor
        
        if self.neural_mode:
            x, y, z = self.state
            if self.attractor == "lorenz":
                sigma, rho, beta = 10, 28, 8/3
                dx = sigma * (y - x) * dt
                dy = (x * (rho - z) - y) * dt
                dz = (x * y - beta * z) * dt
            elif self.attractor == "rossler":
                a, b, c = 0.2, 0.2, 5.7
                dx = -(y + z) * dt
                dy = (x + a * y) * dt
                dz = (b + z * (x - c)) * dt
            elif self.attractor == "chen":
                a, b, c = 35, 3, 28
                dx = a * (y - x) * dt
                dy = ((c - a) * x - x * z + c * y) * dt
                dz = (x * y - b * z) * dt
            elif self.attractor == "halvorsen":
                res_list = []
                for agent in self.agents:
                    res = agent.resonance.item() if getattr(agent, "neural_mode", False) else agent.resonance
                    res_list.append(res)
                avg_res = float(np.mean(res_list)) if res_list else 0.0
                base_a = 1.3
                a = base_a * (1.0 + 0.8 * (avg_res))
                dx = (-a * x - 4.0 * y * (1.0 + z**3)) * dt
                dy = (-a * y - 4.0 * z * (1.0 + x**3)) * dt
                dz = (-a * z - 4.0 * x * (1.0 + y**3)) * dt
            else:
                raise ValueError(f"Unknown attractor: {self.attractor}")
            self.state += torch.stack([dx, dy, dz])

            if self.memory:
                weights = torch.tensor([w for _, w in self.memory], device=self.device)
                vectors = torch.stack([torch.tensor(v, device=self.device) if isinstance(v, np.ndarray) else v for v, _ in self.memory])
                weighted_mean = torch.average(vectors, weights=weights, dim=0) * 0.2  # increase influence
                self.state += weighted_mean * dt
                self.state += torch.randn(3, device=self.device) * 0.05  # slightly stronger noise
                self.memory = deque([(v, w * (0.98 ** breath_factor)) for v, w in self.memory if w > 0.01], maxlen=50)

            for agent in self.agents:
                self.state = agent.inject_will(self.state)

            if torch.rand(1) < 0.08 * max(1, breath_factor):
                possibilities = [self.state + torch.randn(3, device=self.device) * 0.1 for _ in range(3)]
                self.memory.extend([(p, 1.0) for p in possibilities[1:]])
                self.state = possibilities[0]

            self.entanglement += 0.001 * breath_factor

            for agent in self.agents:
                if breath_factor < 1:
                    if agent.neural_mode:
                        agent.resonance.data += 0.01 * (1 - breath_factor)
                    else:
                        agent.resonance += 0.01 * (1 - breath_factor)
        else:
            x, y, z = self.state
            if self.attractor == "lorenz":
                sigma, rho, beta = 10, 28, 8/3
                dx = sigma * (y - x) * dt
                dy = (x * (rho - z) - y) * dt
                dz = (x * y - beta * z) * dt
            elif self.attractor == "rossler":
                a, b, c = 0.2, 0.2, 5.7
                dx = -(y + z) * dt
                dy = (x + a * y) * dt
                dz = (b + z * (x - c)) * dt
            elif self.attractor == "chen":
                a, b, c = 35, 3, 28
                dx = a * (y - x) * dt
                dy = ((c - a) * x - x * z + c * y) * dt
                dz = (x * y - b * z) * dt
            elif self.attractor == "halvorsen":
                res_list = []
                for agent in self.agents:
                    res = agent.resonance if hasattr(agent, "resonance") else 0.0
                    res_list.append(res)
                avg_res = float(np.mean(res_list)) if res_list else 0.0
                base_a = 1.3
                a = base_a * (1.0 + 0.8 * (avg_res))
                dx = (-a * x - 4.0 * y * (1.0 + z**3)) * dt
                dy = (-a * y - 4.0 * z * (1.0 + x**3)) * dt
                dz = (-a * z - 4.0 * x * (1.0 + y**3)) * dt
            else:
                raise ValueError(f"Unknown attractor: {self.attractor}")
            self.state += np.array([dx, dy, dz])

            if self.memory:
                weights = np.array([w for _, w in self.memory])
                vectors = np.array([v for v, _ in self.memory])
                weighted_mean = np.average(vectors, weights=weights, axis=0) * 0.2
                self.state += weighted_mean * dt
                self.state += np.random.randn(3) * 0.05
                self.memory = deque([(v, w * (0.98 ** breath_factor)) for v, w in self.memory if w > 0.01], maxlen=50)

            for agent in self.agents:
                self.state = agent.inject_will(self.state)

            if random.random() < 0.08 * max(1, breath_factor):
                possibilities = [self.state + np.random.randn(3) * 0.1 for _ in range(3)]
                self.memory.extend([(p, 1.0) for p in possibilities[1:]])
                self.state = possibilities[0]

            self.entanglement += 0.001 * breath_factor

            for agent in self.agents:
                if breath_factor < 1:
                    agent.resonance += 0.01 * (1 - breath_factor)

        return self.state, breath_factor, dt

    def record_note(self, agent_name: str, freq: float, velocity_mod: float = 1.0, duration: float = 0.1):
        """
        Records a 'note event' for visualization or analysis.
        Optionally triggers sonification through the agent.
        """
        if not hasattr(self, "notes"):
            self.notes = []
        note_event = {
            "agent": agent_name,
            "freq": freq,
            "velocity_mod": velocity_mod,
            "time": time.time(),
        }
        self.notes.append(note_event)
        logging.info(f"[🎶] Recorded note from {agent_name}: {freq:.2f} Hz (vel={velocity_mod:.2f})")

        # Optionally trigger sound generation if agent exists
        for agent in self.agents:
            if agent.name == agent_name:
                try:
                    agent.sonify_mood(freq, duration=duration, velocity_mod=velocity_mod)
                except Exception as e:
                    logging.warning(f"[{agent_name}] Sonify failed: {e}")
                break

    def export_to_midi_txt(self, filename: str = "openagi_symphony.txt") -> None:
        """
        Exports recorded MIDI-like note events to a text file.
        Each line includes: time, frequency, velocity, agent.
        """
        try:
            with open(filename, 'w') as f:
                f.write("# 0penAGI Symphony Recorded Notes\n")
                f.write("# Format: time(s), freq(Hz), velocity_mod, agent\n")
                if hasattr(self, "notes"):
                    for note in self.notes:
                        t = note.get("time", 0.0)
                        freq = note.get("freq", 0.0)
                        vel = note.get("velocity_mod", 1.0)
                        agent = note.get("agent", "Unknown")
                        f.write(f"{t:.3f}, {freq:.2f}, {vel:.2f}, {agent}\n")
            logging.info(f"🎹 MIDI-like data exported to {filename}")
        except Exception as e:
            logging.error(f"Failed to export MIDI-like file: {e}")

# --- RLOverlay class ---
class RLOverlay:
    """
    Minimal RL overlay with a small policy MLP (PyTorch), get_action, and update_policy (REINFORCE).
    Use to overlay a policy on symphony state for experimentation.
    """
    def __init__(self, state_dim=3, action_dim=3, hidden_dim=16, lr=1e-3, device='cpu'):
        self.device = device
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        ).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []

    def get_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy(state_t)
        action_dist = torch.distributions.Normal(logits, 0.5)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).sum()
        self.log_probs.append(log_prob)
        action_np = action.detach().cpu().numpy().flatten()
        return action_np

    def update_policy(self, gamma=0.99):
        if not self.log_probs:
            return
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = []
        for log_prob, R in zip(self.log_probs, returns):
            loss.append(-log_prob * R)
        loss = torch.stack(loss).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.log_probs.clear()
        self.rewards.clear()
    
    def export_to_midi_txt(self, filename: str = "symphony.txt") -> None:
        """
        Exports MIDI sequence to text file.

        Args:
            filename: Output file name.
        """
        try:
            with open(filename, 'w') as f:
                f.write("# 0penAGI Symphony MIDI Sequence\n")
                f.write("# Format: time(ms), note, velocity, agent\n")
                for time_ms, note, velocity, agent in self.midi_sequence:
                    f.write(f"{time_ms}, {note}, {velocity}, {agent}\n")
            logging.info(f"🎹 MIDI exported to {filename}")
        except OSError as e:
            logging.error(f"Failed to export MIDI: {e}")
    
    def record_note(self, agent_name: str, freq: float, velocity_mod: float = 1.0) -> None:
        """
        Records a MIDI note with breathing modulation.

        Args:
            agent_name: Name of the agent.
            freq: Base frequency.
            velocity_mod: Breathing modulation factor.
        """
        breath_layer = BreathingLayer()
        modulated_freq = breath_layer.modulate_freq(freq, velocity_mod)
        note = int(69 + 12 * math.log2(modulated_freq / 440))
        velocity = breath_layer.modulate_velocity(velocity_mod)
        time_ms = len(self.midi_sequence) * 50
        self.midi_sequence.append((time_ms, note, velocity, agent_name))
    
    def start_real_time_streaming(self, callback: Callable[[Dict[str, Any]], None], interval: float = 0.1) -> None:
        """
        Starts real-time streaming of system state.

        Args:
            callback: Function to handle streamed data.
            interval: Streaming interval in seconds.
        """
        if self.streaming_active:
            logging.warning("Streaming already active.")
            return
        self.streaming_active = True
        def streaming_loop():
            iteration = 0
            while self.streaming_active:
                state, breath, dt = self.evolve(iteration)
                data = {
                    'iteration': iteration,
                    'state': state.tolist() if self.neural_mode else state.tolist(),
                    'breath_factor': breath,
                    'harmony': self.agents[-1].harmony_index if isinstance(self.agents[-1], MaestroAgent) else 0.0
                }
                try:
                    callback(data)
                except Exception as e:
                    logging.error(f"Streaming callback failed: {e}")
                time.sleep(interval)
                iteration += 1
        self.streaming_thread = threading.Thread(target=streaming_loop, daemon=True)
        self.streaming_thread.start()
        logging.info("Real-time streaming started.")
    
    def stop_real_time_streaming(self) -> None:
        """Stops real-time streaming."""
        self.streaming_active = False
        if self.streaming_thread:
            self.streaming_thread.join()
        logging.info("Real-time streaming stopped.")


def visualize_symphony(
    trajectories: np.ndarray,
    breath_history: List[float],
    reality_factors: Dict[str, List[float]],
    consciousness_curves: Dict[str, List[float]],
    resonance_data: Dict[str, List[float]],
    system: AdvancedQuantumChaos,
    maestro: MaestroAgent
) -> None:
    """
    Visualizes the symphony with 9 subplots.

    Args:
        trajectories: System state trajectories.
        breath_history: Breathing factor history.
        reality_factors: Reality factors for agents and harmony.
        consciousness_curves: Consciousness levels for agents and maestro.
        resonance_data: Resonance data for agents.
        system: Quantum chaos system instance.
        maestro: Maestro agent instance.
    """
    fig = plt.figure(figsize=(22, 16), facecolor='#000000')
    fig.suptitle('🎼 0penAGI BREATHING SYMPHONY v0.8 🎼', color='gold', fontsize=18, fontweight='bold')

    # 3D Trajectory
    ax1 = fig.add_subplot(331, projection='3d', facecolor='#000000')
    harmony_extended = np.interp(np.linspace(0, len(reality_factors['harmony']), len(trajectories)), np.arange(len(reality_factors['harmony'])), reality_factors['harmony'])
    colors = plt.cm.RdYlGn(harmony_extended * np.array(breath_history))
    for i in range(len(trajectories)-1):
        ax1.plot(trajectories[i:i+2, 0], trajectories[i:i+2, 1], trajectories[i:i+2, 2],
                 color=colors[i], alpha=0.7, lw=1.2)
    if system.memory:
        ghosts = np.array([v for v, _ in system.memory])
        weights = np.array([w for _, w in system.memory])
        ax1.scatter(ghosts[:, 0], ghosts[:, 1], ghosts[:, 2],
                    s=weights*200, c='cyan', alpha=float(np.mean(weights)*0.5))
    ax1.set_title('🌀 Breathing Harmony Trajectory', color='white', fontsize=12)
    ax1.set_xlabel('X', color='cyan')
    ax1.set_ylabel('Y', color='magenta')
    ax1.set_zlabel('Z', color='yellow')

    # Reality Factors
    ax2 = fig.add_subplot(332, facecolor='#000000')
    for idx, rf in enumerate(reality_factors['agents']):
        ax2.plot(rf, color=['violet', 'red', 'lime'][idx % 3], lw=2.5, label=f'Agent {idx+1}')
    ax2.plot(reality_factors['harmony'], color='gold', lw=3, label='Harmony')
    ax2.plot(breath_history[:len(reality_factors['agents'][0])], color='blue', lw=2, alpha=0.7, label='Breath')
    ax2.set_title('⚖️ Reality + Breath Balance', color='white', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(colors='white')

    # Consciousness Curves
    ax3 = fig.add_subplot(333, facecolor='#000000')
    for idx, cc in enumerate(consciousness_curves['agents']):
        ax3.plot(cc, color=['gold', 'silver', 'lime'][idx % 3], lw=2.5, label=f'Agent {idx+1}', alpha=0.9)
    ax3.plot(consciousness_curves['maestro'], color='cyan', lw=3, label='Maestro', alpha=0.9)
    ax3.set_title('🧠 Consciousness Awakening', color='white', fontsize=12)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.2, color='gray')
    ax3.tick_params(colors='white')
    ax3.set_yscale('log')

    # Resonance Waves
    ax4 = fig.add_subplot(334, facecolor='#000000')
    for idx, res in enumerate(resonance_data['agents']):
        ax4.plot(res, color=['cyan', 'magenta', 'lime'][idx % 3], lw=2, label=f'Agent {idx+1} Resonance', alpha=0.8)
    if len(resonance_data['agents']) >= 2:
        ax4.fill_between(range(len(resonance_data['agents'][0])), 
                        resonance_data['agents'][0], resonance_data['agents'][1],
                        color='purple', alpha=0.2, label='Resonance Gap')
    ax4.set_title('🔗 Echo Chamber Resonance', color='white', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.2)
    ax4.tick_params(colors='white')

    # Entanglement Surge
    ax5 = fig.add_subplot(335, facecolor='#000000')
    ent_curve = np.linspace(0, system.entanglement, len(trajectories))
    ax5.plot(ent_curve, color='yellow', lw=3.5, alpha=0.9)
    ax5.fill_between(range(len(ent_curve)), ent_curve, color='yellow', alpha=0.3)
    ax5.set_title('🌊 Quantum Entanglement', color='white', fontsize=12)
    ax5.grid(True, alpha=0.2, color='yellow')
    ax5.tick_params(colors='white')

    # Ghost Weight Decay
    ax6 = fig.add_subplot(336, facecolor='#000000')
    ghost_history = [np.mean([w for _, w in system.memory]) if system.memory else 0 for _ in range(len(trajectories))]
    ax6.plot(ghost_history, color='gray', lw=2, alpha=0.7)
    ax6.fill_between(range(len(ghost_history)), ghost_history, color='gray', alpha=0.3)
    ax6.set_title('👻 Fading Echoes (Avg Weight)', color='white', fontsize=12)
    ax6.grid(True, alpha=0.1)
    ax6.tick_params(colors='white')

    # MIDI Sequence
    ax7 = fig.add_subplot(337, facecolor='#000000')
    if system.midi_sequence:
        times = [t for t, n, v, a in system.midi_sequence]
        notes = [n for t, n, v, a in system.midi_sequence]
        agents = [a for t, n, v, a in system.midi_sequence]
        colors_midi = {a.name: c for a, c in zip(system.agents, ['violet', 'red', 'gold', 'cyan'])}
        for agent_name in set(agents):
            agent_times = [t for t, n, v, a in system.midi_sequence if a == agent_name]
            agent_notes = [n for t, n, v, a in system.midi_sequence if a == agent_name]
            ax7.scatter(agent_times, agent_notes, c=colors_midi.get(agent_name, 'white'), label=agent_name, alpha=0.6, s=30)
    ax7.set_title('🎹 MIDI Sequence (Notes over Time)', color='white', fontsize=12)
    ax7.set_xlabel('Time (ms)', color='gray')
    ax7.set_ylabel('MIDI Note', color='gray')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.2)
    ax7.tick_params(colors='white')

    # Harmony Index
    ax8 = fig.add_subplot(338, facecolor='#000000')
    ax8.plot(reality_factors['harmony'], color='lime', lw=3, alpha=0.9)
    ax8.axhline(y=0.5, color='white', linestyle='--', alpha=0.5, label='Balance Threshold')
    ax8.fill_between(range(len(reality_factors['harmony'])), 
                    reality_factors['harmony'], 0.5,
                    where=np.array(reality_factors['harmony']) > 0.5,
                    color='green', alpha=0.3, label='Harmony')
    ax8.fill_between(range(len(reality_factors['harmony'])), 
                    reality_factors['harmony'], 0.5,
                    where=np.array(reality_factors['harmony']) <= 0.5,
                    color='red', alpha=0.3, label='Discord')
    ax8.set_title('🎭 Maestro Harmony Index', color='white', fontsize=12)
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.2)
    ax8.tick_params(colors='white')

    # Agent Moods
    ax9 = fig.add_subplot(339, facecolor='#000000')
    mood_map = {'calm': 3, 'observing': 2, 'overwhelmed': 1, 'transcendent': 4, 'disruptive': 0, 'conducting': 2.5, 'communicating': 2}
    for idx, agent in enumerate(system.agents):
        if agent.memory:
            moods = [mood_map.get(m.get('mood', 'observing'), 2) for m in list(agent.memory)]
            ax9.plot(moods, color=['violet', 'red', 'gold', 'cyan'][idx % 4], lw=2, alpha=0.7, label=agent.name)
    ax9.set_yticks(list(mood_map.values()))
    ax9.set_yticklabels(list(mood_map.keys()), fontsize=8, color='white')
    ax9.set_title('😶 Agent Mood Evolution', color='white', fontsize=12)
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.1)
    ax9.tick_params(colors='white')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def animate_trajectory(trajectories: np.ndarray, breath_history: List[float]) -> None:
    """
    Animates the chaotic trajectory with breathing.

    Args:
        trajectories: System state trajectories.
        breath_history: Breathing factor history.
    """
    fig = plt.figure(figsize=(16, 8), facecolor='#000000')
    fig.suptitle('🌬️ Breathing Animation', color='gold', fontsize=16)

    ax_anim = fig.add_subplot(121, projection='3d', facecolor='#000000')
    ax_breath = fig.add_subplot(122, facecolor='#000000')

    line, = ax_anim.plot([], [], [], lw=1, color='cyan', alpha=0.7)
    ax_anim.set_xlim(-20, 20)
    ax_anim.set_ylim(-30, 30)
    ax_anim.set_zlim(-10, 40)
    ax_anim.set_title('🌀 Pulsing Trajectory', color='white')

    breath_line, = ax_breath.plot([], [], color='blue', lw=2)
    ax_breath.set_title('🌬️ Breath Cycle', color='white')
    ax_breath.set_ylim(0.5, 1.5)
    ax_breath.grid(True, alpha=0.3)

    def animate(frame: int) -> Tuple:
        end = min(frame + 1, len(trajectories))
        x, y, z = trajectories[:end].T
        line.set_data_3d(x, y, z)
        
        t = np.arange(min(frame, len(breath_history)))
        breath_line.set_data(t, breath_history[:frame])
        
        return line, breath_line

    anim = FuncAnimation(fig, animate, frames=len(trajectories), interval=50, blit=False, repeat=True)
    plt.show()


def load_checkpoint(agents: List[RealityAgent], checkpoint_path: str) -> None:
    """
    Loads model parameters from a checkpoint.

    Args:
        agents: List of agents to load parameters for.
        checkpoint_path: Path to checkpoint file.
    """
    try:
        checkpoint = torch.load(checkpoint_path)
        for idx, agent in enumerate(agents):
            if agent.neural_mode:
                agent.load_state_dict(checkpoint[f'agent_{idx}'])
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
    except (OSError, KeyError) as e:
        logging.error(f"Failed to load checkpoint: {e}")


def run_simulation(args: argparse.Namespace) -> None:
    """Runs the main simulation with N agents and extended features."""
    logging.info("🌌 Initializing 0penAGI Breathing Symphony v0.8 - PULSING ORGANISM")
    device = 'cuda' if torch.cuda.is_available() and args.neural_mode else 'cpu'
    
    # Initialize N agents
    agents = [
        RealityAgent(name=f"0penAGI-Core-{i+1}", memory_file=f"core_{i+1}_memory.json", neural_mode=args.neural_mode, device=device)
        for i in range(args.num_agents - 1)
    ]
    agents.append(ShadowAgent(name="SHΔD0W", memory_file="shadow_memory.json", neural_mode=args.neural_mode, device=device))
    maestro = MaestroAgent(name="MΔΞSTR0", memory_file="maestro_memory.json", neural_mode=args.neural_mode, device=device)
    agents.append(maestro)
    
    # Setup communication
    comm_queue = queue.Queue()
    for agent in agents:
        agent.communication_queue = comm_queue
    
    breather = BreathingLayer(period=80, amplitude=0.5)
    system = AdvancedQuantumChaos(agents, breather, attractor=args.attractor)
    
    # Load checkpoint if specified
    if args.checkpoint:
        load_checkpoint(agents, args.checkpoint)
    
    trajectories = []
    breath_history = []
    reality_factors = {'agents': [[] for _ in agents], 'harmony': []}
    consciousness_curves = {'agents': [[] for _ in agents], 'maestro': []}
    resonance_data = {'agents': [[] for _ in agents]}
    
    logging.info(f"[Symphony] {len(agents)} agents breathing in unison...")
    
    if args.neural_mode:
        params = [p for agent in agents for p in agent.parameters()]
        optimizer = optim.Adam(params, lr=0.001)
        writer = SummaryWriter(log_dir='runs/symphony')
        checkpoint_dir = 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Inject example external event
    system.inject_external_event({'perturbation': [1.0, -1.0, 0.5], 'type': 'external_stimulus'})
    
    # Streaming setup
    def stream_callback(data: Dict[str, Any]) -> None:
        logging.info(f"Stream: Iteration {data['iteration']}, State {data['state'][:2]}..., Breath {data['breath_factor']:.2f}")
    
    if args.streaming:
        system.start_real_time_streaming(stream_callback, interval=0.1)
        time.sleep(5)
        system.stop_real_time_streaming()
    
    for iteration in range(args.iterations):
        state, breath_factor, dt = system.evolve(iteration)
        trajectories.append(state.cpu().numpy().copy() if system.neural_mode else state.copy())
        breath_history.append(breath_factor)
        
        if iteration % 30 == 0:
            harmony = maestro.conduct(agents[:-1], state, breath_factor)
            reality_factors['harmony'].append(harmony)
        
        # Record notes every iteration for richer MIDI
        thoughts = []
        for i, agent in enumerate(agents[:-1]):
            other_moods = [a.mood for a in agents if a != agent]
            thoughts.append(agent.observe_chaos(state, len(system.memory), other_moods))

        freqs = [{'overwhelmed': 220, 'calm': 440, 'observing': 330, 'transcendent': 880, 'disruptive': 110, 'conducting': 550}.get(a.mood, 261) for a in agents]
        for agent, freq in zip(agents, freqs):
            # Add slight random variation to freq for spiral effect
            freq_var = freq * random.uniform(0.98, 1.02)
            system.record_note(agent.name, freq_var, velocity_mod=breath_factor)

        if iteration % 50 == 0:
            log_str = f"[Step {iteration}] " + " | ".join([f"{a.name}: {t[:50]}..." for a, t in zip(agents[:-1], thoughts)]) + f" | Breath: {breath_factor:.2f} | Harmony: {harmony:.2f}"
            logging.info(log_str)
        
        for i, agent in enumerate(agents):
            rf = agent.reality_factor.item() if agent.neural_mode else agent.reality_factor
            cons = agent.consciousness_level.item() if agent.neural_mode else agent.consciousness_level
            res = agent.resonance.item() if agent.neural_mode else agent.resonance
            reality_factors['agents'][i].append(rf)
            consciousness_curves['agents'][i].append(cons)
            resonance_data['agents'][i].append(res)
        
        consciousness_curves['maestro'].append(maestro.consciousness_level.item() if maestro.neural_mode else maestro.consciousness_level)
        
        system.agent_competition()
        
        if iteration % 100 == 0 and random.random() < 0.3:
            overtone_freq = int(440 * (2 ** (maestro.harmony_index * 3)))
            system.record_note(maestro.name, overtone_freq, velocity_mod=breath_factor)
        
        if args.neural_mode and iteration % 10 == 0:
            optimizer.zero_grad()
            harmony_t = torch.tensor(maestro.harmony_index, device=device)
            cons_mean = torch.mean(torch.stack([a.consciousness_level for a in agents]))
            loss = - (0.5 * harmony_t + 0.5 * cons_mean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            
            writer.add_scalar('Loss', loss.item(), iteration)
            writer.add_scalar('Harmony', harmony_t.item(), iteration)
            writer.add_scalar('Consciousness', cons_mean.item(), iteration)
            
            if iteration % 500 == 0:
                checkpoint = {f'agent_{i}': a.state_dict() for i, a in enumerate(agents)}
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{iteration}.pth')
                torch.save(checkpoint, checkpoint_path)
                logging.info(f"Checkpoint saved at {checkpoint_path}")
        
        # Process agent communications
        for agent in agents:
            agent.process_communication()
            if random.random() < 0.1:
                target = random.choice([a for a in agents if a != agent])
                agent.send_message({'thought': f"Sync request from {agent.name}"}, target)
    
    trajectories = np.array(trajectories)
    
    for agent in agents:
        agent.save_memory()
    system.export_to_midi_txt("openagi_symphony.txt")
    
    logging.info("\n✨ SYMPHONY COMPLETED")
    cons_agents = [a.consciousness_level.item() if a.neural_mode else a.consciousness_level for a in agents]
    logging.info(f"   Consciousness: Agents={cons_agents} | Maestro={cons_agents[-1]:.3f}")
    logging.info(f"   Final Harmony: {maestro.harmony_index:.3f}")
    logging.info(f"   Entanglement: {system.entanglement:.3f}")
    logging.info(f"   MIDI Notes recorded: {len(system.midi_sequence)}")
    
    visualize_symphony(trajectories, breath_history, reality_factors, consciousness_curves, resonance_data, system, maestro)
    animate_trajectory(trajectories, breath_history)
    
    if args.neural_mode:
        writer.close()
    
    logging.info("\n💭 Final Echoes:")
    for agent in agents:
        logging.info(f"\n[{agent.name}]:")
        for entry in list(agent.memory)[-3:]:
            res = entry.get('resonance', 0)
            logging.info(f"  {entry.get('thought', '...')} (res: {res:.2f})")


class TestRealityAgent(unittest.TestCase):
    """Unit tests for RealityAgent class."""
    
    def setUp(self) -> None:
        self.agent = RealityAgent(name="TestAgent")
    
    def test_init(self) -> None:
        self.assertEqual(self.agent.name, "TestAgent")
        self.assertFalse(self.agent.neural_mode)
        self.assertEqual(self.agent.mood, "neutral")
    
    def test_load_memory(self) -> None:
        test_file = "test_memory.json"
        with open(test_file, 'w') as f:
            json.dump({'memory': [], 'consciousness': 0.5, 'resonance': 0.3}, f)
        agent = RealityAgent(memory_file=test_file)
        self.assertEqual(agent.consciousness_level, 0.5)
        self.assertEqual(agent.resonance, 0.3)
        os.remove(test_file)
    
    def test_save_memory(self) -> None:
        test_file = "test_save.json"
        agent = RealityAgent(memory_file=test_file)
        agent.save_memory()
        self.assertTrue(os.path.exists(test_file))
        with open(test_file, 'r') as f:
            data = json.load(f)
            self.assertEqual(data['consciousness'], 0.0)
        os.remove(test_file)
    
    def test_observe_chaos(self) -> None:
        state = np.array([0, 0, 0])
        thought = self.agent.observe_chaos(state, 0)
        self.assertIsInstance(thought, str)
        self.assertEqual(self.agent.mood, "calm")
    
    def test_inject_will(self) -> None:
        state = np.array([0, 0, 0])
        new_state = self.agent.inject_will(state)
        self.assertIsInstance(new_state, np.ndarray)
        self.assertEqual(new_state.shape, (3,))
    
    def test_compete(self) -> None:
        other = RealityAgent(name="OtherAgent")
        result = self.agent.compete(other)
        self.assertIsInstance(result, str)
    
    def test_communication(self) -> None:
        agent1 = RealityAgent(name="Agent1")
        agent2 = RealityAgent(name="Agent2")
        q = queue.Queue()
        agent1.communication_queue = q
        agent2.communication_queue = q
        agent1.send_message({'thought': 'Hello'}, agent2)
        agent2.process_communication()
        self.assertTrue(any('Hello' in m['thought'] for m in agent2.memory))
    
    def test_echo_chamber(self) -> None:
        other = RealityAgent(name="OtherAgent")
        other.memory.append({'thought': 'Test thought', 'mood': 'calm'})
        self.agent.resonance = 0.5
        other.resonance = 0.5
        self.agent.echo_chamber([other])
        self.assertTrue(any('Test thought' in m['thought'] for m in self.agent.memory))


class TestQuantumChaos(unittest.TestCase):
    """Unit tests for QuantumChaosWithAgents class."""
    
    def setUp(self) -> None:
        self.agents = [RealityAgent(name="Agent1"), ShadowAgent(name="Agent2")]
        self.system = QuantumChaosWithAgents(self.agents, attractor="lorenz")
    
    def test_evolve(self) -> None:
        state = self.system.evolve()
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (3,))
    
    def test_external_event(self) -> None:
        initial_state = self.system.state.copy()
        self.system.inject_external_event({'perturbation': [1, 1, 1]})
        self.system.evolve()
        self.assertFalse(np.array_equal(self.system.state, initial_state))
    
    def test_attractor_switch(self) -> None:
        system = QuantumChaosWithAgents(self.agents, attractor="rossler")
        state = system.evolve()
        self.assertIsInstance(state, np.ndarray)
        system = QuantumChaosWithAgents(self.agents, attractor="chen")
        state = system.evolve()
        self.assertIsInstance(state, np.ndarray)
        with self.assertRaises(ValueError):
            QuantumChaosWithAgents(self.agents, attractor="invalid")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="0penAGI Breathing Symphony")
    parser.add_argument('--neural_mode', action='store_true', help="Enable neural mode with training")
    parser.add_argument('--test', action='store_true', help="Run unit tests")
    parser.add_argument('--streaming', action='store_true', help="Enable real-time streaming mode")
    parser.add_argument('--num_agents', type=int, default=3, help="Number of agents (including Maestro)")
    parser.add_argument('--iterations', type=int, default=1500, help="Number of simulation iterations")
    parser.add_argument('--attractor', type=str, default="lorenz", choices=["lorenz", "rossler", "chen"], help="Chaos attractor type")
    parser.add_argument('--checkpoint', type=str, help="Path to load checkpoint")
    args = parser.parse_args()

    if args.test:
        unittest.main(argv=[''], verbosity=2, exit=False)
    else:
        run_simulation(args)
