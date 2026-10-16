"""Pure reward model for the RL-IDS MDP (used by the Gymnasium environment).

R(s,a) = R_detection - C_action(a) - P_latency(a):

    R_detection : confusion-matrix payoff of the decision (TN +1, TP +18,
                  FP -12, FN -40; 'NONE' = 0 for pure-monitor steps).
    C_action    : operating cost of the action.
    P_latency   : latency penalty alpha*max(0, queue-1) for *overuse* of
                  DPI/Throttle (billed from the 2nd consecutive use so that a
                  single Throttle step in the slides' worked example carries
                  no extra term).  Depends on episode history, so it is added
                  by the environment step, not baked into the pure net values.

The reward is a *pure function* of (state, action, next_state): given the
compromise state the agent is really in, the action it took, and the state the
attack evolution leads to next, exactly one net reward is returned.  There is
no hidden state inside the reward: any trainer can reproduce any episode's
returns offline from the (state, action, next_state) trace alone.

Net values per (state, action, outcome):

    Monitor (any state)          -> no detection claim, reward 0
    DPI    on Suspicious, benign -> TN   (+1 - cost 1  = 0)
    DPI    on Suspicious, attack -> TP   (+18 - cost 1 = 17)
    Throttle that lets a real
    attack escalate              -> FN   (-40 - cost 1.5 = -41.5)
    Honeypot snaring an attack   -> TP   (+18 - cost 3 = 15)
    Block  that only persists    -> no event yet, just the cost (-2)
    Block  that resolves         -> TP payoff (+18, incident closed;
                                   slide net value folds the cost in)
    Isolate (any Compromised)    -> TP, charged full quarantine cost (+3)
"""
from __future__ import annotations

from rl_ids import config as C

# net reward by (state, action_name, next_state); next_state may be
# TERMINAL_MARKER / None for terminal actions. Deterministic pure function.
_NET: dict[tuple[str, str, str], float] = {
    # --- Normal ----------------------------------------------------------- #
    ("Normal", "Monitor", "Normal"):        0.0,
    ("Normal", "Monitor", "Suspicious"):    0.0,   # attack prelude - not yet detected
    ("Normal", "Monitor", "Compromised"):   0.0,   # sudden compromise - not yet detected
    # --- Suspicious ------------------------------------------------------- #
    ("Suspicious", "Monitor", "Suspicious"):    0.0,
    ("Suspicious", "Monitor", "Compromised"):   0.0,
    ("Suspicious", "Monitor", "Normal"):        0.0,
    ("Suspicious", "DPI", "Normal"):            C.DETECTION_REWARDS["TN"] - C.ACTION_COSTS[C.ACTION_INDEX["DPI"]],        # 0.0
    ("Suspicious", "DPI", "Suspicious"):       -C.ACTION_COSTS[C.ACTION_INDEX["DPI"]],                                 # -1.0
    ("Suspicious", "DPI", "Compromised"):       C.DETECTION_REWARDS["TP"] - C.ACTION_COSTS[C.ACTION_INDEX["DPI"]],        # 17.0
    ("Suspicious", "Throttle", "Normal"):       C.DETECTION_REWARDS["TN"] - C.ACTION_COSTS[C.ACTION_INDEX["Throttle"]],   # -0.5
    ("Suspicious", "Throttle", "Suspicious"):  -C.ACTION_COSTS[C.ACTION_INDEX["Throttle"]],                             # -1.5
    ("Suspicious", "Throttle", "Compromised"):  C.DETECTION_REWARDS["FN"] - C.ACTION_COSTS[C.ACTION_INDEX["Throttle"]],   # -41.5
    ("Suspicious", "Honeypot", "Normal"):       C.DETECTION_REWARDS["TN"] - C.ACTION_COSTS[C.ACTION_INDEX["Honeypot"]],   # -2.0
    ("Suspicious", "Honeypot", "Suspicious"):   C.DETECTION_REWARDS["TP"] - C.ACTION_COSTS[C.ACTION_INDEX["Honeypot"]],   # 15.0
    ("Suspicious", "Honeypot", "Compromised"):  C.DETECTION_REWARDS["TP"] - C.ACTION_COSTS[C.ACTION_INDEX["Honeypot"]],   # 15.0
    # --- Compromised ------------------------------------------------------ #
    ("Compromised", "Monitor", "Compromised"):  0.0,
    ("Compromised", "Monitor", "Normal"):       0.0,
    # Block that merely persists: cost only (attack still active).
    ("Compromised", "Block", "Compromised"):   -C.ACTION_COSTS[C.ACTION_INDEX["Block"]],                                # -2.0
    # Block that resolves: TP payoff, terminal (per-slide net value 18).
    ("Compromised", "Block", "Normal"):          C.DETECTION_REWARDS["TP"],                                            # 18.0
    # Isolate: TP but full quarantine cost.
    ("Compromised", "Isolate", C.TERMINAL_MARKER): C.DETECTION_REWARDS["TP"] - C.ACTION_COSTS[C.ACTION_INDEX["Isolate"]],   # 3.0
    ("Compromised", "Isolate", None):            C.DETECTION_REWARDS["TP"] - C.ACTION_COSTS[C.ACTION_INDEX["Isolate"]],
    # --------------------------------------------------------------------- #
    # Mask-free (Path 2) entries: every playbook-illegal action is now
    # reachable.  Semantics: inspecting benign traffic is a wasted-cost no-op
    # (NONE), mitigation on a benign state is a false positive, and a soft action
    # on a confirmed compromise is a false negative.  Design goal: the playbook
    # action stays the strict argmax in every state (verified by the mask-free
    # Bellman solution), so the task is unchanged - only the mask is gone.
    # --------------------------------------------------------------------- #
    # --- Normal (true state benign) --------------------------------------- #
    ("Normal", "DPI", "Normal"):        -C.ACTION_COSTS[C.ACTION_INDEX["DPI"]],              # -1.0  (NONE)
    ("Normal", "DPI", "Suspicious"):    -C.ACTION_COSTS[C.ACTION_INDEX["DPI"]],
    ("Normal", "DPI", "Compromised"):   -C.ACTION_COSTS[C.ACTION_INDEX["DPI"]],
    ("Normal", "Throttle", "Normal"):   -C.ACTION_COSTS[C.ACTION_INDEX["Throttle"]],         # -1.5
    ("Normal", "Throttle", "Suspicious"): -C.ACTION_COSTS[C.ACTION_INDEX["Throttle"]],
    ("Normal", "Throttle", "Compromised"): -C.ACTION_COSTS[C.ACTION_INDEX["Throttle"]],
    ("Normal", "Honeypot", "Normal"):   C.DETECTION_REWARDS["FP"] - C.ACTION_COSTS[C.ACTION_INDEX["Honeypot"]],   # -15.0
    ("Normal", "Honeypot", "Suspicious"): C.DETECTION_REWARDS["FP"] - C.ACTION_COSTS[C.ACTION_INDEX["Honeypot"]],
    ("Normal", "Honeypot", "Compromised"): C.DETECTION_REWARDS["FP"] - C.ACTION_COSTS[C.ACTION_INDEX["Honeypot"]],
    ("Normal", "Block", "Normal"):      C.DETECTION_REWARDS["FP"] - C.ACTION_COSTS[C.ACTION_INDEX["Block"]],      # -14.0
    ("Normal", "Block", "Suspicious"):  C.DETECTION_REWARDS["FP"] - C.ACTION_COSTS[C.ACTION_INDEX["Block"]],
    ("Normal", "Block", "Compromised"): C.DETECTION_REWARDS["FP"] - C.ACTION_COSTS[C.ACTION_INDEX["Block"]],
    ("Normal", "Isolate", C.TERMINAL_MARKER): C.DETECTION_REWARDS["FP"] - C.ACTION_COSTS[C.ACTION_INDEX["Isolate"]],  # -27.0
    ("Normal", "Isolate", None):        C.DETECTION_REWARDS["FP"] - C.ACTION_COSTS[C.ACTION_INDEX["Isolate"]],
    # --- Suspicious: over-reaction on a merely-suspicious flow ------------ #
    ("Suspicious", "Block", "Normal"):      C.DETECTION_REWARDS["FP"] - C.ACTION_COSTS[C.ACTION_INDEX["Block"]],   # -14.0
    ("Suspicious", "Block", "Suspicious"):  C.DETECTION_REWARDS["TP"] - C.ACTION_COSTS[C.ACTION_INDEX["Block"]],   # 16.0
    ("Suspicious", "Block", "Compromised"): C.DETECTION_REWARDS["TP"] - C.ACTION_COSTS[C.ACTION_INDEX["Block"]],   # 16.0
    ("Suspicious", "Isolate", C.TERMINAL_MARKER): C.DETECTION_REWARDS["FP"] - C.ACTION_COSTS[C.ACTION_INDEX["Isolate"]],  # -27.0
    ("Suspicious", "Isolate", None):        C.DETECTION_REWARDS["FP"] - C.ACTION_COSTS[C.ACTION_INDEX["Isolate"]],
    # --- Compromised: too-soft actions let a confirmed compromise persist -- #
    ("Compromised", "DPI", "Compromised"):      C.DETECTION_REWARDS["FN"] - C.ACTION_COSTS[C.ACTION_INDEX["DPI"]],      # -41.0
    ("Compromised", "DPI", "Normal"):           C.DETECTION_REWARDS["TP"] - C.ACTION_COSTS[C.ACTION_INDEX["DPI"]],      # 17.0
    ("Compromised", "Throttle", "Compromised"): C.DETECTION_REWARDS["FN"] - C.ACTION_COSTS[C.ACTION_INDEX["Throttle"]], # -41.5
    ("Compromised", "Throttle", "Normal"):      C.DETECTION_REWARDS["TP"] - C.ACTION_COSTS[C.ACTION_INDEX["Throttle"]], # 16.5
    ("Compromised", "Honeypot", "Compromised"): C.DETECTION_REWARDS["FN"] - C.ACTION_COSTS[C.ACTION_INDEX["Honeypot"]], # -43.0
    ("Compromised", "Honeypot", "Normal"):      C.DETECTION_REWARDS["TP"] - C.ACTION_COSTS[C.ACTION_INDEX["Honeypot"]], # 15.0
}


def _normalize_action(action: int | str) -> str:
    if isinstance(action, str):
        name = action
    else:
        name = C.ACTIONS[int(action)]
    if name not in C.ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    return name


def classification(state: str, action: int | str, next_state: str | None) -> str:
    """Return the confusion-matrix event ('TN','TP','FP','FN','NONE') for a step."""
    name = _normalize_action(action)
    if name == "Monitor":
        return "NONE"
    key = (state, name, next_state)
    net = _NET.get(key)
    if net is None:
        return "NONE"
    costs = C.ACTION_COSTS[C.ACTION_INDEX[name]]
    det = net + costs if next_state not in (None, C.TERMINAL_MARKER) else net
    # map back to the detection payoff that produced the net value (best effort)
    for event, value in C.DETECTION_REWARDS.items():
        if next_state in (None, C.TERMINAL_MARKER):
            if abs(net - value) < 1e-9 or (name == "Isolate" and net == value - costs):
                return event
        elif abs(det - value) < 1e-9 or abs(net - value) < 1e-9:
            return event
    return "NONE"


def reward_function(state: str, action: int | str, next_state: str | None,
                    enforce_legality: bool = True) -> float:
    """Net immediate reward for (state, action) leading to next_state.

    ``next_state`` is the MDP state the transition landed on; for terminal
    actions (Isolate) pass ``None`` or C.TERMINAL_MARKER.  With
    ``enforce_legality=True`` (default) masked (illegal) actions return
    C.INVALID_ACTION_PENALTY - the environment never offers them, so this is just
    a guard rail.  Mask-free mode (Path 2) passes ``enforce_legality=False`` so
    the reward is total over all 18 (state, action) pairs.
    """
    name = _normalize_action(action)
    if enforce_legality and C.ACTION_INDEX[name] not in C.VALID_ACTIONS[state]:
        return C.INVALID_ACTION_PENALTY
    key = (state, name, next_state)
    if key not in _NET:
        # Unknown-but-legal combination: reward is the negative action cost
        # (no detection event). Keeps the function total and cheap to test.
        return -C.ACTION_COSTS[C.ACTION_INDEX[name]]
    return _NET[key]


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """G_t for each step t of an episode: G_t = sum_{k>=t} gamma^(k-t) r_k."""
    g = 0.0
    out = [0.0] * len(rewards)
    for t in range(len(rewards) - 1, -1, -1):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


def scripted_detection_events(
    trace: list[tuple[str, str, str | None]]
) -> list[str]:
    """Events for a (state, action, next_state) trace (for reports/tests)."""
    return [classification(s, a, sp) for s, a, sp in trace]


# --------------------------------------------------------------------------- #
# Explicit R(s,a) = R_detection - C_action(a) - P_latency(a) components
# --------------------------------------------------------------------------- #
def detection_payoff(event: str) -> float:
    """R_detection for a confusion-matrix event ('TN','TP','FP','FN','NONE')."""
    if event == "NONE":
        return 0.0
    return C.DETECTION_REWARDS[event]


def action_cost(action: int | str) -> float:
    """C_action(a): operating cost of executing the action once."""
    return C.ACTION_COSTS[C.ACTION_INDEX[_normalize_action(action)]]


def latency_penalty(queue_length: int, alpha: float | None = None) -> float:
    """P_latency: alpha*max(0, queue-1) on DPI/Throttle overuse.

    ``queue_length`` is the number of *consecutive* DPI/Throttle actions.  Billing
    starts at the 2nd consecutive use, so a lone Throttle step is cost-free
    (keeps the slides' worked example exact) while sustained overuse accrues
    alpha per extra queued step.
    """
    alpha = C.LATENCY_ALPHA if alpha is None else float(alpha)
    return alpha * max(0, int(queue_length) - 1)


def step_breakdown(
    state: str, action: int | str, next_state: str | None,
    enforce_legality: bool = True,
) -> dict[str, float | str]:
    """Decompose one step into its reward components (latency excluded).

    Returns {event, detection (R_detection), cost (C_action), net}.  ``net`` is
    the exact value reward_function() returns, i.e. R(s,a) before P_latency.
    Pass ``enforce_legality=False`` to decompose a mask-free (Path 2) step.
    """
    name = _normalize_action(action)
    net = reward_function(state, name, next_state, enforce_legality=enforce_legality)
    return {
        "event": classification(state, name, next_state),
        "detection": detection_payoff(classification(state, name, next_state)),
        "cost": action_cost(name),
        "net": net,
    }


def step_reward(
    state: str,
    action: int | str,
    next_state: str | None,
    latency_run: int = 0,
    enforce_legality: bool = True,
) -> float:
    """Full per-step reward R(s,a) = net(state, action, next) - P_latency(run).

    ``latency_run`` must mirror the environment's consecutive DPI/Throttle
    counter *after* this action was executed (this step's queue length).
    """
    return (reward_function(state, action, next_state,
                            enforce_legality=enforce_legality)
            - latency_penalty(latency_run))


def returns_from_trace(
    trace: list[tuple[str, str, str | None]],
    gamma: float = C.GAMMA,
    enforce_legality: bool = True,
) -> list[float]:
    """Discounted returns G_t for an offline (state, action, next_state) trace."""
    return discounted_returns(
        [reward_function(s, a, sp, enforce_legality=enforce_legality)
         for s, a, sp in trace], gamma
    )