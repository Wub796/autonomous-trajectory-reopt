# Mathematical Foundations: From AP Calculus AB to Autonomous Spaceflight

This primer derives and explains every mathematical equation used in this research project. Every derivation builds directly on topics covered in **AP Calculus AB**, **AP Statistics**, and **introductory Physics / Linear Algebra**.

---

## 1. Derivatives and Rates of Change: Numerical Orbit Propagation

### 1.1 The Definition of the Derivative
In AP Calculus AB, the instantaneous derivative of a function $f(t)$ is defined as the limit of the average rate of change:

$$f'(t) = \frac{df}{dt} = \lim_{\Delta t \to 0} \frac{f(t + \Delta t) - f(t)}{\Delta t}$$

When modeling a spacecraft moving through space:
- Position is a vector $\mathbf{r}(t) = \langle x(t), y(t), z(t) \rangle$.
- Velocity is the rate of change of position: $\mathbf{v}(t) = \frac{d\mathbf{r}}{dt}$.
- Acceleration is the rate of change of velocity: $\mathbf{a}(t) = \frac{d\mathbf{v}}{dt} = \frac{d^2\mathbf{r}}{dt^2}$.

### 1.2 The Tangent Line (Linear Approximation)
For a small time interval $\Delta t$, the tangent line gives an approximation for the function at $t + \Delta t$:

$$f(t + \Delta t) \approx f(t) + f'(t) \cdot \Delta t$$

Applying this to velocity and position yields **Euler's Method**:

$$\mathbf{v}(t + \Delta t) \approx \mathbf{v}(t) + \mathbf{a}(t) \cdot \Delta t$$
$$\mathbf{r}(t + \Delta t) \approx \mathbf{r}(t) + \mathbf{v}(t + \Delta t) \cdot \Delta t$$

In orbital mechanics, using the updated velocity $\mathbf{v}(t + \Delta t)$ to compute the new position is known as **semi-implicit Euler integration**. This prevents artificial energy buildup and ensures orbital stability over multi-month simulations.

---

## 2. Universal Gravitation in Three Dimensions

Newton's Law of Universal Gravitation states that the gravitational attraction between the Sun (mass $M_\odot$) and a spacecraft (mass $m$) separated by distance $r = \|\mathbf{r}\|$ has magnitude:

$$F_g = \frac{G M_\odot m}{r^2}$$

Dividing by spacecraft mass $m$ ($a = F/m$) gives the gravitational acceleration magnitude:

$$a_g = \frac{\mu}{r^2}, \quad \text{where } \mu = G M_\odot = 1.3271 \times 10^{11} \text{ km}^3/\text{s}^2$$

### Direction in 3D Space
Gravity pulls radially inward toward the Sun (the origin). The unit vector pointing toward the Sun is $-\frac{\mathbf{r}}{\|\mathbf{r}\|}$. Combining magnitude and direction:

$$\mathbf{a}_g = \left(\frac{\mu}{\|\mathbf{r}\|^2}\right) \left(-\frac{\mathbf{r}}{\|\mathbf{r}\|}\right) = -\frac{\mu \mathbf{r}}{\|\mathbf{r}\|^3}$$

In Cartesian components:

$$a_{gx} = -\frac{\mu x}{(x^2 + y^2 + z^2)^{3/2}}, \quad a_{gy} = -\frac{\mu y}{(x^2 + y^2 + z^2)^{3/2}}, \quad a_{gz} = -\frac{\mu z}{(x^2 + y^2 + z^2)^{3/2}}$$

---

## 3. Rocket Propulsion: The Natural Logarithm and Mass Depletion

### 3.1 Propellant Mass Flow Rate
An electric ion thruster expels propellant at an effective exhaust velocity $c = I_{sp} \cdot g_0$, where $I_{sp}$ is the specific impulse (in seconds) and $g_0 = 9.80665 \text{ m/s}^2$.

By Newton's Third Law, thrust $T$ is the rate of momentum ejection:

$$T = -\frac{dm}{dt} \cdot c = -\frac{dm}{dt} \cdot (I_{sp} g_0)$$

Rearranging gives the differential equation for propellant depletion:

$$\frac{dm}{dt} = -\frac{T}{I_{sp} g_0}$$

### 3.2 Derivation of the Tsiolkovsky Rocket Equation
In gravity-free space, $m \frac{dv}{dt} = -c \frac{dm}{dt}$. Dividing both sides by $m$:

$$dv = -c \frac{1}{m} dm$$

Integrating both sides from initial mass $m_0$ to final mass $m_f$:

$$\int_{v_0}^{v_f} dv = -c \int_{m_0}^{m_f} \frac{1}{m} dm$$

Using the fundamental integral $\int \frac{1}{x} dx = \ln|x|$:

$$\Delta v = -c \Big( \ln(m_f) - \ln(m_0) \Big) = c \Big( \ln(m_0) - \ln(m_f) \Big) = I_{sp} g_0 \ln\left(\frac{m_0}{m_f}\right)$$

---

## 4. Thruster Degradation: Logarithmic Aging & Derivatives

### 4.1 The Degradation Model
Experimental wear tests on Hall thrusters (e.g., SPT-140) demonstrate that erosion rates are highest during initial operation and decelerate over time. This physical behavior is captured by a logarithmic curve:

$$P(h) = P_0 - k \ln\left(1 + \frac{h}{\tau}\right)$$

- $h$: cumulative operating hours
- $P_0$: initial nominal performance (thrust $T_0$ or efficiency $\eta_0$)
- $k$: degradation sensitivity coefficient
- $\tau$: characteristic wear deceleration time constant

### 4.2 Wear Rate as a Derivative
Using the Chain Rule from AP Calculus AB:

$$\frac{d}{dh} \left[ \ln\left(1 + \frac{h}{\tau}\right) \right] = \frac{1}{1 + \frac{h}{\tau}} \cdot \frac{1}{\tau} = \frac{1}{\tau + h}$$

Therefore, the instantaneous rate of degradation is:

$$\frac{dP}{dh} = -\frac{k}{\tau + h}$$

- At $h = 0$, the wear rate is maximum: $\left.\frac{dP}{dh}\right|_{0} = -\frac{k}{\tau}$.
- As $h \to \infty$, the wear rate asymptotically approaches zero: $\lim_{h \to \infty} \frac{dP}{dh} = 0$.

---

## 5. Least-Squares Regression via Calculus Optimization

Given $N$ empirical wear data points $(h_i, y_i)$, we want to find $P_0$, $k$, and $\tau$ that minimize the Sum of Squared Errors (SSE):

$$SSE(P_0, k, \tau) = \sum_{i=1}^N \Big( y_i - \big[P_0 - k \ln(1 + h_i/\tau)\big] \Big)^2$$

### 5.1 Linear Transformation with Fixed $\tau$
If $\tau$ is temporarily fixed, let $x_i = \ln(1 + h_i/\tau)$. The model becomes linear:

$$y_i \approx a + b x_i, \quad \text{where } a = P_0 \text{ and } b = -k$$

The error function is:

$$S(a, b) = \sum_{i=1}^N (y_i - a - b x_i)^2$$

### 5.2 Finding the Minimum via Single-Variable Derivatives (AP Calculus AB & AP Statistics)
In AP Statistics, every least-squares regression line passes through the point of averages $(\bar{x}, \bar{y})$, which gives the intercept directly in terms of the slope $b$:

$$\bar{y} = a + b \bar{x} \implies a = \bar{y} - b \bar{x}$$

Substituting $a = \bar{y} - b \bar{x}$ into the error function turns $S$ into a **single-variable quadratic function of the slope $b$**:

$$S(b) = \sum_{i=1}^N \big( (y_i - \bar{y}) - b(x_i - \bar{x}) \big)^2$$

Now we use the **First Derivative Test** and the **Chain Rule** from AP Calculus AB to find the slope $b$ that minimizes $S(b)$:

$$\frac{dS}{db} = -2 \sum_{i=1}^N (x_i - \bar{x}) \big( (y_i - \bar{y}) - b(x_i - \bar{x}) \big) = 0$$

Solving for $b$ gives the exact AP Statistics least-squares slope formula:

$$b = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}$$

We evaluate this analytic solution across a 1D grid of candidate $\tau$ values to find the global minimum $\tau^*$.

---

## 6. Statistical Anomaly Detection: Z-Score & Central Limit Theorem

### 6.1 Noise Reduction via Rolling Averages
Let hourly sensor measurements be modeled as:

$$Y_t = \mu_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2)$$

A rolling average of $w$ consecutive measurements computes:

$$\bar{Y}_w = \frac{1}{w} \sum_{i=0}^{w-1} Y_{t-i}$$

By the properties of variance for independent random variables:

$$\text{Var}(\bar{Y}_w) = \frac{1}{w^2} \sum_{i=0}^{w-1} \text{Var}(\epsilon_{t-i}) = \frac{1}{w^2} (w \sigma^2) = \frac{\sigma^2}{w}$$

The standard deviation of the average (Standard Error) is:

$$\sigma_{\bar{Y}} = \frac{\sigma}{\sqrt{w}}$$

Averaging $w = 24$ hours of measurements reduces sensor noise by $\sqrt{24} \approx 4.9\times$.

### 6.2 Physics-Informed Residual and Z-Score
The detector computes the residual between measured thrust $T_t$ and the expected degradation curve $T_{\text{expected}}(t)$:

$$e_t = \frac{T_t}{T_{\text{expected}}(t)} - 1$$

During nominal flight, $\mathbb{E}[e_t] = 0$. The standardized Z-score is:

$$z_t = \frac{\bar{e}_w}{\sigma_e / \sqrt{w}}$$

Under healthy conditions, $z_t \sim \mathcal{N}(0, 1)$. By the normal distribution:
- $P(z_t < -2.0) \approx 2.28\%$
- $P(z_t < -3.5) \approx 0.023\%$

Requiring $z_t < -3.5$ for $3$ consecutive hours virtually eliminates false alarms during nominal operations while detecting true failures within hours.

### 6.3 Collective Anomaly Feature Extraction & Isolation Forest (#20, #26, #31)
In their anomaly detection survey, Chandola et al. (2009) [#31] define a **collective anomaly** as a sequence of data points that is anomalous together even when individual points look normal in isolation. To turn hourly readings into collective features for an **Isolation Forest** (Liu et al., 2008 [#26]; Fu et al., 2022 [#20]), we compute:
1. **24-hour Rolling Residual Mean** $\bar{e}_w(t) = \frac{1}{w}\sum_{i=0}^{w-1} e(t-i)$ (reduces noise by $\sqrt{w}$).
2. **24-hour Secant Slope (Average Rate of Change from AP Calculus AB):**

$$\text{slope}_w(t) = \frac{\bar{e}_w(t) - \bar{e}_w(t - w)}{w} \approx \frac{d\bar{e}}{dt}$$

An Isolation Forest (#26) builds $M = 50$ random binary decision trees by picking a random feature and a random split value between the minimum and maximum of that feature. Because collective degradation points sit outside the healthy cluster, they are isolated near the root of the tree (short path length $h(\mathbf{x})$), whereas normal points require many splits (deep path length).

---

## 7. Machine Learning: Finite-Difference Policy Gradients & Proximal Clipping

### 7.1 Neural Network Architecture (#5, #18, #22)
The pilot policy $\pi_\theta(\mathbf{o})$ maps an 11-dimensional observation vector $\mathbf{o}$ to 3 continuous control outputs through a single hidden layer:

$$\mathbf{h} = \tanh(\mathbf{W}_1 \mathbf{o} + \mathbf{b}_1) \quad (\mathbf{h} \in \mathbb{R}^{16})$$
$$\mathbf{u} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2 \quad (\mathbf{u} \in \mathbb{R}^3)$$

Control actions are bounded using standard functions:
- Throttle: $\delta = \text{sigmoid}(u_0) = \frac{1}{1 + e^{-u_0}} \in (0, 1)$
- Pitch angle: $\alpha = \frac{\pi}{2} \tanh(u_1) \in (-\pi/2, \pi/2)$
- Yaw angle: $\beta = \frac{\pi}{2} \tanh(u_2) \in (-\pi/2, \pi/2)$

All weights and biases form a parameter vector $\boldsymbol{\theta} \in \mathbb{R}^{243}$.

### 7.2 The Symmetric Difference Quotient (AP Calculus AB)
In AP Calculus AB, the derivative $f'(x)$ of a function can be approximated by testing a small step $+h$ and $-h$ on either side of $x$ (**symmetric difference quotient**):

$$f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}$$

Applying this exact idea to the neural network weights $\boldsymbol{\theta}$ along a random perturbation direction $\boldsymbol{\epsilon}$ with step size $\sigma$ gives the rate of change of the mission score $J$:

$$\frac{\Delta J}{\Delta \sigma} \approx \frac{J(\boldsymbol{\theta} + \sigma \boldsymbol{\epsilon}) - J(\boldsymbol{\theta} - \sigma \boldsymbol{\epsilon})}{2\sigma}$$

### 7.3 Advantage Baseline Subtraction (#27, #30) & Finite-Difference Policy Gradient
To reduce trial-and-error noise over long 460-day trajectories (#27 Schulman et al., 2016; #30 Mnih et al., 2016; Salimans et al., 2017), raw scores are converted into baseline-centered rank advantages $A_i \in [-0.5, +0.5]$ with mean $0$:

$$\hat{\mathbf{g}} = \frac{1}{2 P \sigma} \sum_{i=1}^P \Big[ A(\boldsymbol{\theta} + \sigma \boldsymbol{\epsilon}_i) - A(\boldsymbol{\theta} - \sigma \boldsymbol{\epsilon}_i) \Big] \boldsymbol{\epsilon}_i$$

### 7.4 Proximal Safety Cap / Clipped Policy Update (#19 Schulman et al., 2017)
In AP Calculus AB, a tangent line only approximates a curve accurately *near* the point of tangency. Taking too large a parameter step can push the neural network into unstable steering behavior. Following Proximal Policy Optimization (#19), we apply a piecewise clipping function to cap the maximum change per update:

$$\text{clip}(\Delta \theta, -\epsilon_{\text{clip}}, +\epsilon_{\text{clip}}) = \begin{cases} -\epsilon_{\text{clip}}, & \Delta \theta < -\epsilon_{\text{clip}} \\ \Delta \theta, & -\epsilon_{\text{clip}} \le \Delta \theta \le +\epsilon_{\text{clip}} \\ +\epsilon_{\text{clip}}, & \Delta \theta > +\epsilon_{\text{clip}} \end{cases}$$

$$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k + \text{clip}\big(\eta \hat{\mathbf{g}},\; -\epsilon_{\text{clip}},\; +\epsilon_{\text{clip}}\big)$$

---

## 8. Experimental Design: Controlled & Two-Way Factorial Analysis (#28, #29)

### 8.1 Fixed-Control Comparison (#28 Creswell, 2012)
To establish causal validity (#28), all controller architectures are evaluated on the **exact same random seeds**, initial Earth launch vectors, and thruster degradation onset hours. Because every extraneous variable is held constant, any difference in terminal Mars miss distance $d_{\min}$ or fuel consumed $\Delta m$ is strictly caused by the controller architecture.

### 8.2 Two-Way Factorial Interaction (#29 Leedy & Ormrod, 2015)
In a two-way factorial experiment with Factor A (Controller Setup, 3 levels) and Factor B (Degradation Severity $s \in \{10\%, 15\%, 20\%, 25\%\}$, 4 levels), the **interaction effect** measures whether the benefit of autonomous adaptation changes as hardware wear worsens:

$$\Delta d_{\text{saved}}(s) = \bar{d}_{\text{Pre-Planned}}(s) - \bar{d}_{\text{Onboard}}(s)$$

If $\Delta d_{\text{saved}}(s)$ increases monotonically with $s$, the two factors interact positively: the more severely the thruster degrades, the larger the trajectory error prevented by onboard autonomy.
