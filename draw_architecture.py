"""Draw the 3-model architecture diagram (all English labels)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# ============ Colors ============
C_ADDER = '#4A90D9'
C_CARRY = '#7B68EE'
C_LEARN = '#E8913A'
C_INPUT = '#5CB85C'
C_OUTPUT = '#D9534F'
C_TEACHER = '#F0AD4E'
C_FA = '#9B59B6'

# ============ Title ============
ax.text(8, 9.6, 'Online Learning Addition System - 3-Model Architecture',
        fontsize=15, fontweight='bold', ha='center', va='center')
ax.text(8, 9.25, 'DigitAdder (inference) + CarryPropagator (inference) + LearningMechanism (learning)',
        fontsize=10, ha='center', va='center', color='#666')

# ============ Model 1: DigitAdder ============
adder_box = FancyBboxPatch((0.5, 4.5), 5.5, 4.2, boxstyle="round,pad=0.15",
                            facecolor='#E8F0FE', edgecolor=C_ADDER, linewidth=2.5)
ax.add_patch(adder_box)
ax.text(3.25, 8.4, 'Model 1: DigitAdder (2,046 params)', fontsize=12, fontweight='bold',
        ha='center', va='center', color=C_ADDER)

layers_adder = [
    ('x [22]\na+b+cin', 1.5, 7.5, 1.8, 0.6, C_INPUT),
    ('W1 [128x22]\n+ ReLU', 1.5, 6.5, 1.8, 0.7, C_ADDER),
    ('W2 [128x128]\n+ ReLU', 1.5, 5.5, 1.8, 0.7, C_ADDER),
    ('digit_head\nWd [10x128]', 4.0, 6.5, 1.8, 0.7, C_OUTPUT),
    ('carry_head\nWc [2x128]', 4.0, 5.5, 1.8, 0.7, C_OUTPUT),
]
for text, x, y, w, h, color in layers_adder:
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                          facecolor='white', edgecolor=color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, fontsize=8, ha='center', va='center',
            family='monospace', color=color)

# Internal arrows
ax.annotate('', xy=(2.4, 7.15), xytext=(2.4, 7.5),
            arrowprops=dict(arrowstyle='->', color=C_ADDER, lw=1.5))
ax.annotate('', xy=(2.4, 6.15), xytext=(2.4, 6.5),
            arrowprops=dict(arrowstyle='->', color=C_ADDER, lw=1.5))
ax.annotate('', xy=(4.0, 6.85), xytext=(3.3, 6.85),
            arrowprops=dict(arrowstyle='->', color=C_ADDER, lw=1.5))
ax.annotate('', xy=(4.0, 5.85), xytext=(3.3, 5.85),
            arrowprops=dict(arrowstyle='->', color=C_ADDER, lw=1.5))

ax.text(5.95, 6.85, 'digit\n[10]', fontsize=8, ha='left', va='center', color=C_OUTPUT, family='monospace')
ax.text(5.95, 5.85, 'cout\n[2]', fontsize=8, ha='left', va='center', color=C_OUTPUT, family='monospace')

# ============ Model 2: CarryPropagator ============
carry_box = FancyBboxPatch((7.5, 4.5), 4.5, 4.2, boxstyle="round,pad=0.15",
                            facecolor='#F0ECFE', edgecolor=C_CARRY, linewidth=2.5)
ax.add_patch(carry_box)
ax.text(9.75, 8.4, 'Model 2: CarryPropagator (42 params)', fontsize=12, fontweight='bold',
        ha='center', va='center', color=C_CARRY)

# GRU Cell
gru_box = FancyBboxPatch((8.2, 5.8), 2.0, 1.5, boxstyle="round,pad=0.08",
                           facecolor='white', edgecolor=C_CARRY, linewidth=1.5)
ax.add_patch(gru_box)
ax.text(9.2, 6.55, 'GRU Cell\n[2 -> 2]', fontsize=9, ha='center', va='center',
        color=C_CARRY, family='monospace')

# cin_head
cin_box = FancyBboxPatch((8.2, 5.0), 2.0, 0.6, boxstyle="round,pad=0.05",
                           facecolor='white', edgecolor=C_CARRY, linewidth=1.5)
ax.add_patch(cin_box)
ax.text(9.2, 5.3, 'cin_head [2x2]', fontsize=8, ha='center', va='center',
        color=C_CARRY, family='monospace')

# h state
ax.text(10.5, 6.55, 'h [2]', fontsize=9, ha='left', va='center',
        color=C_CARRY, family='monospace', fontweight='bold')

# Recurrent arrow
ax.annotate('', xy=(10.3, 7.0), xytext=(10.3, 6.0),
            arrowprops=dict(arrowstyle='->', color=C_CARRY, lw=1.5,
                           connectionstyle='arc3,rad=-0.5'))
ax.text(10.9, 6.55, 'recur', fontsize=7, ha='left', va='center', color=C_CARRY)

# cin_head arrow
ax.annotate('', xy=(9.2, 5.6), xytext=(9.2, 5.8),
            arrowprops=dict(arrowstyle='->', color=C_CARRY, lw=1.5))

# I/O labels
ax.text(8.0, 6.55, 'cout\n[2]', fontsize=8, ha='right', va='center',
        color=C_INPUT, family='monospace')
ax.text(8.0, 5.3, 'cin\n[2]', fontsize=8, ha='right', va='center',
        color=C_OUTPUT, family='monospace')

# ============ Model 3: LearningMechanism ============
learn_box = FancyBboxPatch((0.5, 0.3), 15, 3.8, boxstyle="round,pad=0.15",
                            facecolor='#FEF3E8', edgecolor=C_LEARN, linewidth=2.5)
ax.add_patch(learn_box)
ax.text(8, 3.8, 'Model 3: LearningMechanism (~17,932 params, independent from inference models)',
        fontsize=12, fontweight='bold', ha='center', va='center', color=C_LEARN)

# FA feedback matrices
fa_items = [
    ('B_digit\n[128x10]', 1.5, 2.5, C_FA),
    ('B_carry\n[128x2]', 3.5, 2.5, C_FA),
    ('B2\n[128x128]', 5.5, 2.5, C_FA),
]
for text, x, y, color in fa_items:
    box = FancyBboxPatch((x, y), 1.6, 0.9, boxstyle="round,pad=0.05",
                          facecolor='white', edgecolor=color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + 0.8, y + 0.45, text, fontsize=8, ha='center', va='center',
            color=color, family='monospace')

ax.text(3.5, 3.4, 'FA Feedback Matrices (fixed random, replace W^T for error propagation)',
        fontsize=9, ha='center', va='center', color=C_FA)

# Learning rates / decay
param_items = [
    ('eta_adder [4]\nlearning rate', 8.0, 2.5, C_LEARN),
    ('eta_carry [2]\nlearning rate', 9.8, 2.5, C_LEARN),
    ('gamma_adder [4]\ndecay', 11.6, 2.5, C_LEARN),
    ('gamma_carry [2]\ndecay', 13.4, 2.5, C_LEARN),
]
for text, x, y, color in param_items:
    box = FancyBboxPatch((x, y), 1.5, 0.9, boxstyle="round,pad=0.05",
                          facecolor='white', edgecolor=color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + 0.75, y + 0.45, text, fontsize=7.5, ha='center', va='center',
            color=color, family='monospace')

# Delta rule formulas
ax.text(8, 1.5, 'Output layer:  dW = eta * gate * (target - softmax(output)) * pre^T  -  gamma * W',
        fontsize=9, ha='center', va='center', color='#333', family='monospace',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9E6', edgecolor='#CCC'))
ax.text(8, 0.8, 'Hidden layer:  dW = eta * gate * (B @ output_error) * ReLU\' * pre^T  -  gamma * W   (FA approx.)',
        fontsize=9, ha='center', va='center', color='#333', family='monospace',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9E6', edgecolor='#CCC'))

# ============ Data flow arrows ============
# DigitAdder -> CarryPropagator: cout
ax.annotate('', xy=(7.5, 5.85), xytext=(6.1, 5.85),
            arrowprops=dict(arrowstyle='->', color='#333', lw=2.0))
ax.text(6.8, 6.15, 'cout', fontsize=9, ha='center', va='center',
        color='#333', fontweight='bold', family='monospace')

# CarryPropagator -> DigitAdder: cin
ax.annotate('', xy=(6.1, 7.0), xytext=(7.5, 7.0),
            arrowprops=dict(arrowstyle='->', color='#333', lw=2.0))
ax.text(6.8, 7.3, 'cin', fontsize=9, ha='center', va='center',
        color='#333', fontweight='bold', family='monospace')

# ============ Teacher ============
teacher_box = FancyBboxPatch((12.5, 7.0), 2.8, 1.5, boxstyle="round,pad=0.1",
                              facecolor='#FFF8E1', edgecolor=C_TEACHER, linewidth=2)
ax.add_patch(teacher_box)
ax.text(13.9, 7.75, 'Teacher\nCorrection\n-----------\ndigit_true\ncout_true\ncin_true\nreward (+/-1)',
        fontsize=8, ha='center', va='center', color=C_TEACHER, family='monospace')

ax.annotate('', xy=(13.9, 4.1), xytext=(13.9, 7.0),
            arrowprops=dict(arrowstyle='->', color=C_TEACHER, lw=2.0, linestyle='dashed'))
ax.text(14.3, 5.5, 'correction\nsignal', fontsize=8, ha='left', va='center', color=C_TEACHER)

# LearningMechanism -> DigitAdder (weight update)
ax.annotate('', xy=(3.25, 4.5), xytext=(3.25, 4.1),
            arrowprops=dict(arrowstyle='->', color=C_LEARN, lw=2.5, linestyle='dashed'))
ax.text(3.8, 4.3, 'dW\nweight\nupdate', fontsize=7, ha='left', va='center',
        color=C_LEARN, fontweight='bold')

# LearningMechanism -> CarryPropagator (weight update)
ax.annotate('', xy=(9.75, 4.5), xytext=(9.75, 4.1),
            arrowprops=dict(arrowstyle='->', color=C_LEARN, lw=2.5, linestyle='dashed'))
ax.text(10.3, 4.3, 'dW\nweight\nupdate', fontsize=7, ha='left', va='center',
        color=C_LEARN, fontweight='bold')

# ============ Loop annotation ============
loop_box = FancyBboxPatch((0.3, 9.0), 6, 0.5, boxstyle="round,pad=0.05",
                           facecolor='#E8F5E9', edgecolor='#4CAF50', linewidth=1.5)
ax.add_patch(loop_box)
ax.text(3.3, 9.25, '<-- Per-digit loop (LSB->MSB): cin -> DigitAdder -> digit+cout -> CarryPropagator -> cin -> ...',
        fontsize=9, ha='center', va='center', color='#2E7D32')

# ============ Three learning modes ============
modes_box = FancyBboxPatch((7.0, 0.5), 8.5, 0.7, boxstyle="round,pad=0.05",
                            facecolor='white', edgecolor='#999', linewidth=1)
ax.add_patch(modes_box)
ax.text(11.25, 0.85, 'Hebbian: gate=1.0  |  Reward: gate=2.0(wrong)/1.0(right)  |  Correction: gate=2.0(wrong)/1.0(right, per-digit)',
        fontsize=8, ha='center', va='center', color='#555', family='monospace')

fig.tight_layout()
fig.savefig('outputs/model_architecture.png', dpi=150, bbox_inches='tight')
print("Architecture diagram saved to outputs/model_architecture.png")
