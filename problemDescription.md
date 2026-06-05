# MVP Product Specification: RL Delivery Robot for Warehouses 


## 1. Problem Description
The goal is to train an autonomous robot to navigate a continuous 2D warehouse environment from a starting point to a delivery goal. The agent must optimize path efficiency (minimizing travel time and path length) while avoiding static obstacles. 

To bridge the gap between simulation stability and realistic operation, this framework uses Procedural World Generation on an underlying discrete grid to guarantee environment solvability, which is then mapped into a Continuous State Space for reinforcement learning training. The action space remains discrete


## 2. Procedural World Generation & Solvability Verification

To ensure that the RL agent only trains on environments that contain a valid, traversable path, worlds are generated procedurally using a discrete-to-continuous pipeline.

```
+------------------+     Verify Path     +-----------------------+     Convert to     +----------------------------+
|  Discrete Grid   | ------------------> | BFS Connectivity Test | -----------------> | Continuous World Simulator |
| Obstacle/Free    |      (4-Way)        | (Pass -> Accept)      |     Coordinates    | (AABBs, Lidar, Circle Bot) |
+------------------+                     +-----------------------+                    +----------------------------+
```

### Step-by-Step Generation Pipeline

1. Grid Parameterization & Clearance Allocation:
   * The world is represented as a grid of $M \times M$ cells (e.g., $16 \times 16$).
   * To prevent "exploration paralysis" and ensure the robot has sufficient clearance to turn without grazing corners, the grid cell width ($W_{cell}$) is set to **twice the robot's diameter**:
     $$W_{cell} = 2 \cdot (2r) = 4r$$
     With a robot radius $r = 0.2$, the cell size is exactly $0.8 \times 0.8$. The continuous world dimensions are thus $12.8 \times 12.8$ units.

2. Random Obstacle Generation:
   * Rectangular obstacles are procedurally generated and snapped to whole grid cells. 
   * Cells covered by these rectangles are marked as blocked; all other cells are marked as free.

3. Start and Goal Placement:
   * A start cell and a goal cell are randomly selected from the pool of free cells (ensuring they are not the same cell).

4. Validity Check via BFS:
   * A Breadth-First Search (BFS) is executed on the discrete grid starting from the robot's cell using 4-connectivity (Up, Down, Left, Right). 
   * Reject: If the BFS cannot find a path to the goal cell, the layout is discarded, and the pipeline restarts from Step 2.
   * Accept: If a path is found, the layout is accepted and compiled into the continuous simulator.

5. Compilation to Continuous Space:
   * Obstacles: The coordinates of the corners of the blocked cells are stored as continuous Axis-Aligned Bounding Boxes (AABBs).
   * Robot Initial Position: The robot is placed at the exact center of its designated start cell.
   * Target Placement with Clearance: The target is placed within its designated cell. If placed randomly inside the cell (rather than at the exact center), a safety margin is enforced: the target's center must remain at a distance of at least $r_{robot} + r_{goal}$ away from any adjacent blocked cell boundaries to prevent the robot from colliding with a wall while attempting to reach it.
   * Once accepted, the discrete grid is discarded, and all state transitions and collision checks are calculated in the continuous coordinate space.

---

## 3. State Definition

### Robot & Goal Representations
* **Robot:** Modeled as a circular rigid body with radius $r = 0.2$.
* **Goal:** Modeled as a circular acceptance region with radius $r_{goal} = 0.3$.

### Continuous State Vector
At each time step $t$, the agent receives a continuous state vector $\mathbf{s}_t$:
$$ \mathbf{s}_t = \left[ 
    \cos(\theta_t), \, 
    \sin(\theta_t), \, 
    v_t, \, 
    L_1, \, L_2, \, \dots, \, L_N 
\right]^T $$

Where:
* $\theta_t$ is the heading angle of the robot's velocity vector.
* $\cos(\theta_t), \sin(\theta_t)$ represent the heading orientation to avoid angular boundary discontinuities [1].
* $v_t = \|\mathbf{v}_t\|$ is the current speed of the robot, where $0 \le v_t \le v_{max}$.
* $L_i$ represents the normalized inverse proximity measurements from the Lidar array.

### Lidar Inputs
The robot is equipped with $N$ virtual Lidar rangefinder rays (e.g., $N = 8$) evenly distributed over a $360^\circ$ field of view. For each ray $i \in \{1, \dots, N\}$:
1. Compute the distance $d_i$ along the ray angle $\alpha_i = \theta_t + i \cdot \frac{2\pi}{N}$ to the nearest obstacle boundary up to a maximum range $d_{max}$ (e.g., $3.0$ units) [1].
2. Normalize the reading to an inverse proximity value [1]:
   $$L_i = 1.0 - \frac{d_i}{d_{max}} \in [0, 1]$$
   *An input of $0.0$ represents a clear path along that ray, while $1.0$ indicates that an obstacle is in immediate contact with the robot's boundary.*

---

## 4. Actions
The action space $\mathcal{A}$ is a discrete set of $6$ commands designed to adjust the robot's kinematics over a control step of duration $\Delta t$:

1. **Acceleration:** Increase linear velocity: $v_{t+1} = \min(v_t + \Delta v, v_{max})$.
2. **Braking:** Decrease linear velocity: $v_{t+1} = \max(v_t - \Delta v, 0)$.
3. **Small Clockwise Rotation:** Rotate heading angle: $\theta_{t+1} = \theta_t - \Delta \theta_{small}$.
4. **Small Anti-Clockwise Rotation:** Rotate heading angle: $\theta_{t+1} = \theta_t + \Delta \theta_{small}$.
5. **Big Clockwise Rotation:** Rotate heading angle: $\theta_{t+1} = \theta_t - \Delta \theta_{big}$.
6. **Big Anti-Clockwise Rotation:** Rotate heading angle: $\theta_{t+1} = \theta_t + \Delta \theta_{big}$.

---

## 5. Rewards
The reward function $R(s_t, a_t, s_{t+1})$ is composed of:

* **Goal Reach:** 
  $$R_{goal} = +100.0 \quad \text{(terminal state, if } dist_{goal,t} \le r_{goal}\text{)}$$
* **Collision:** If the continuous distance to any obstacle boundary or map outer wall is $\le r_{robot}$ ($0.2$):
  $$R_{collision} = -50.0 \quad \text{(terminal state)}$$
* **Step Penalty:** 
  $$R_{step} = -0.1 \quad \text{per simulation step (penalizes idling and circuitous paths)}$$
* **Progress Reward:** 
  $$R_{progress} = c \cdot (dist_{goal,t-1} - dist_{goal,t})$$
  Where $c > 0$ is a scaling coefficient.
