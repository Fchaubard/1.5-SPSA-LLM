"""
CNN Policy Network for Atari Breakout.

Standard DQN-style architecture:
- Input: (batch, 4, 84, 84) - 4 stacked grayscale frames
- Output: (batch, num_actions) - action logits

~1.7M parameters, compatible with existing SPSA infrastructure.
"""

import torch
import torch.nn as nn


class AtariCNN(nn.Module):
    """
    Standard DQN-style CNN for Atari.

    Architecture:
    - Conv1: 32 filters, 8x8 kernel, stride 4
    - Conv2: 64 filters, 4x4 kernel, stride 2
    - Conv3: 64 filters, 3x3 kernel, stride 1
    - FC1: 512 units
    - FC2: num_actions units (output)

    Input: (batch, 4, 84, 84) normalized frames [0, 1]
    Output: (batch, num_actions) action logits
    """

    def __init__(self, num_actions: int = 4):
        """
        Initialize the CNN.

        Args:
            num_actions: Number of discrete actions (Breakout has 4: NOOP, FIRE, LEFT, RIGHT)
        """
        super().__init__()

        self.num_actions = num_actions

        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size after convolutions
        # Input: 84x84
        # After conv1: (84 - 8) / 4 + 1 = 20
        # After conv2: (20 - 4) / 2 + 1 = 9
        # After conv3: (9 - 3) / 1 + 1 = 7
        # Final: 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Output layer with smaller gain
        nn.init.orthogonal_(self.fc2.weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 4, 84, 84), normalized to [0, 1]

        Returns:
            Action logits of shape (batch, num_actions)
        """
        # Convolutional layers with ReLU
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def get_action(self, state: torch.Tensor, deterministic: bool = True) -> int:
        """
        Get action for a single state.

        Args:
            state: State tensor of shape (4, 84, 84) or (1, 4, 84, 84)
            deterministic: If True, return argmax action; otherwise sample

        Returns:
            Action index
        """
        if state.dim() == 3:
            state = state.unsqueeze(0)

        with torch.no_grad():
            logits = self.forward(state)
            if deterministic:
                action = logits.argmax(dim=1).item()
            else:
                probs = torch.softmax(logits, dim=1)
                action = torch.multinomial(probs, 1).item()

        return action

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
