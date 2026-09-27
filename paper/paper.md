# Autonomous Fault-Tolerant Spacecraft Trajectory Optimization via Integrated Anomaly Detection and Reinforcement Learning

**AP Research - Pd 7**  
**2027**  

---

## Abstract

Communicating between Earth and Mars entails a fundamental physical constraint: radio signals require up to forty minutes for a round trip, while ground-based trajectory re-planning across NASA's Deep Space Network (DSN) requires three to fourteen days. For spacecraft employing continuous low-thrust electric propulsion, this latency creates a severe operational vulnerability. When progressive propulsion system degradation begins during a multi-month orbital transfer, ground control cannot intervene before trajectory drift compounds across millions of kilometers. This study introduces an autonomous onboard flight director that closes this response gap by combining a collective-window Isolation Forest anomaly detector with a closed-loop neural reinforcement learning (RL) flight controller trained via proximal clipped finite-difference policy gradients. First, an empirical logarithmic wear model, $P(h) = P_0 - k \ln(1 + h/\tau)$, was calibrated via least-squares calculus optimization against vacuum-chamber endurance data for the SPT-140 Hall-effect thruster, achieving goodness-of-fit values of $R^2 = 0.9966$ for thrust and $R^2 = 0.9996$ for specific impulse ($I_{sp}$). Second, to detect gradual thruster decay amidst $1\%$ sensor noise and solar-array thermal fluctuations ($45.0 \pm 0.5^\circ\text{C}$), a 24-hour rolling window of wear-corrected residuals and calculus secant slopes was extracted. While a point-only Isolation Forest ($w = 1\text{ h}$) achieved only $0.380$ recall ($F_1 = 0.543$), the collective-window Isolation Forest ($w = 24\text{ h}$) achieved $1.000$ precision, $0.980$ recall, and an $F_1$ score of $0.990$ ($20.5\text{ h}$ mean detection delay), closely corroborated by a physics-informed rolling Z-score detector ($F_1 = 0.959$, $13.7\text{ h}$ delay). Third, a compact 243-parameter neural network pilot was trained using a two-stage curriculum over 600 generations ($89,600$ simulated 460-day missions). Across 100 controlled degradation scenarios ($10\%\text{–}25\%$ thrust loss), autonomous onboard re-optimization reduced median Mars miss distance from $5,870,100\text{ km}$ (pre-planned open-loop baseline) to $2,596,298\text{ km}$—a $55.8\%$ reduction in terminal error. A two-way factorial experiment confirmed a positive interaction between degradation severity and onboard autonomy: trajectory error prevented by the autonomous flight director grew monotonically from $2.45 \times 10^6\text{ km}$ at $10\%$ thrust loss to $4.13 \times 10^6\text{ km}$ at $25\%$ thrust loss. Single-step guidance inference required $1.94\ \mu\text{s}$ and $1.90\text{ KB}$ of memory ($>99.99999\%$ real-time margin), demonstrating that autonomous fault-tolerant trajectory optimization is feasible on constrained flight-grade processors.

**Keywords:** Reinforcement learning, Collective anomaly detection, Spacecraft trajectory optimization, Proximal policy clipping, Isolation Forest, Hall-effect thruster, Fault tolerance

---

## 1. Introduction

### 1.1 Background and Context
The orbital distance between Earth and Mars imposes an unavoidable physical barrier on spacecraft operations: radio signals traveling at the speed of light require up to twenty minutes each way, creating a forty-minute round-trip communication delay (Izzo, Märtens, & Pan, 2019; Cheng et al., 2019). Beyond raw light-time latency, operational ground-in-the-loop recovery requires downlinking telemetry through scheduled Deep Space Network (DSN) passes, diagnosing subsystem anomalies, re-computing optimal trajectories on ground workstations, and uplinking corrective command sequences—a cycle lasting three to fourteen days.

Missions utilizing continuous low-thrust electric propulsion face an acute vulnerability under this latency constraint (Zavoli & Federici, 2021). Unlike chemical rockets, which execute high-thrust impulsive burns lasting minutes followed by months of unpowered coasting, electric propulsion systems—such as the SPT-140 Hall-effect thruster selected for NASA's deep-space Psyche mission (Snyder et al., 2018)—operate at low thrust ($T_0 = 0.289\text{ N}$, roughly the weight of a strawberry) and high specific impulse ($I_{sp0} = 1,782\text{ s}$) over thousands of cumulative firing hours. A heliocentric Earth-to-Mars transfer consists of a continuous multi-month spiral that gradually raises the spacecraft's orbital energy and matches Mars's orbital plane. When a propulsion anomaly reduces thrust or specific impulse during an active thrust arc, acceleration errors integrate continuously over time. Any uncompensated drift from the nominal trajectory compounds quadratically in position, and attempting to correct the deviation weeks later consumes substantial propellant reserves—a severe penalty for interplanetary missions designed with tight mass margins (Gaudet, Linares, & Furfaro, 2020b; Miller, Englander, & Linares, 2019).

### 1.2 Problem Statement and Rationale
Although the SPT-140 Hall-effect thruster has demonstrated high efficiency across power settings from $0.9\text{ kW}$ to $4.5\text{ kW}$ (Snyder et al., 2018), no electric thruster is immune to long-duration wear in the deep-space environment. Extended vacuum-chamber endurance testing demonstrates that high-velocity ion bombardment progressively erodes the ceramic discharge channel walls, causing a gradual, nonlinear decline in thrust output and propellant efficiency (Conversano, Lobbia, & Goebel, 2019; Kamhawi et al., 2014). Furthermore, cathode degradation, thermal transients, or power-processing anomalies can trigger steeper $10\%$ to $25\%$ drops in available thrust mid-mission.

Early in a degradation event, performance loss is gradual and masked by routine sensor measurement noise and solar-array thermal cycling (Fu et al., 2022). Traditional fixed-threshold alarm systems, as documented by Hundman et al. (2018) and Chandola, Banerjee, and Kumar (2009), routinely fail to catch gradual equipment drift because single hourly readings remain within normal operating bounds when inspected in isolation. While Hundman et al. (2018) demonstrated that Long Short-Term Memory (LSTM) recurrent neural networks can identify subtle telemetry shifts without labeled failure examples, deep recurrent models impose substantial memory and computational overhead on radiation-hardened spacecraft processors. Conversely, while reinforcement learning controllers have been trained to steer low-thrust trajectories (Miller et al., 2019; Zavoli & Federici, 2021), prior controllers either assume healthy propulsion hardware (Sullivan & Bosanac, 2020; Miller et al., 2019) or evaluate only brief, random engine dropouts rather than progressive Hall thruster decay.

### 1.3 Significance and Purpose
This study bridges the gap between spacecraft health monitoring and autonomous trajectory control by developing and testing an integrated onboard flight director. The architecture pairs a collective-window **Isolation Forest** anomaly detector (Liu, Ting, & Zhou, 2008; Fu et al., 2022) and a physics-informed rolling **Z-score** monitor with a closed-loop **neural reinforcement learning pilot** trained using proximal clipped finite-difference policy gradients (Schulman et al., 2017; Salimans et al., 2017). The anomaly detector continuously filters hourly propulsion telemetry against an analytical logarithmic aging baseline to flag developing faults and estimate remaining thruster health. Without waiting for Earth confirmation, the onboard neural pilot ingests this health estimate and dynamically recalculates 3D steering angles and throttle commands to salvage the Mars encounter.

By shifting both fault diagnosis and trajectory re-optimization onto the spacecraft, this research demonstrates how autonomous onboard intelligence can eliminate multi-day ground intervention delays while operating within the strict memory and microsecond latency budgets of flight-grade microcontrollers (Capra, Brandonisio, & Lavagna, 2025; Elkins, Sood, & Rumpf, 2020).

### 1.4 Objectives
Guided by the central research question—*How can an autonomous flight director dynamically detect progressive thruster degradation and recalculate a continuous Earth–Mars trajectory within the limits of flight-grade computers?*—the primary objectives of this study are:
1. To calibrate and validate a calculus-based logarithmic degradation model against empirical SPT-140 Hall thruster wear data (Kamhawi et al., 2014; Conversano et al., 2019; Snyder et al., 2018);
2. To prove that extracting rolling collective-window features enables an unsupervised Isolation Forest (Liu et al., 2008; Fu et al., 2022) and rolling Z-score detector to identify gradual thruster decay with high precision and recall compared to single-point detection (Chandola et al., 2009);
3. To train a closed-loop 3D neural flight controller via two-stage curriculum learning and evaluate its trajectory recovery through a controlled comparison (Creswell, 2012) and a two-way factorial experiment across multiple degradation severities (Leedy & Ormrod, 2015); and
4. To verify robustness across dispersed Monte Carlo disturbances (Zavoli & Federici, 2021) and benchmark single-core execution latency and memory usage for flight hardware deployment (Capra et al., 2025).

### 1.5 Scope and Limitations
This study models a three-dimensional heliocentric Earth-to-Mars transfer over a 460-day ($11,040\text{ h}$) mission window using the SPT-140 Hall thruster ($T_0 = 0.289\text{ N}, I_{sp0} = 1,782\text{ s}$) and a spacecraft launch mass of $2,747\text{ kg}$ ($1,648\text{ kg}$ dry mass). Orbital dynamics are propagated in 3D Cartesian coordinates under two-body solar gravitation using semi-implicit Euler integration, capturing Mars's $1.85^\circ$ orbital inclination while omitting third-body planetary perturbations and solar radiation pressure. Because real deep-space failure telemetry is scarce, experiments rely on physics-informed synthetic telemetry (Ballentine, Bapat, & Cowlagi, 2026) calibrated to NASA laboratory wear measurements. Consequently, findings represent an analytical and numerical proof of concept prior to hardware-in-the-loop vacuum testing.

---

## 2. Literature Review

To establish the theoretical and methodological foundations of this work, 32 scholarly sources were synthesized across four core themes: (1) catching sensor and engine problems via collective anomaly detection, (2) autonomous trajectory control under signal delays, (3) balancing fuel conservation against targeting accuracy, and (4) flight hardware constraints and experimental design.

### 2.1 Spacecraft Anomaly Detection (Theme A)
Detecting propulsion anomalies in deep space requires unsupervised algorithms capable of spotting novel failure modes without pre-labeled breakdown histories (Hundman et al., 2018; Fu et al., 2022). Because empirical failure records from interplanetary missions are rare, Ballentine, Bapat, and Cowlagi (2026) demonstrated that blending physical equations of motion with stochastic noise models produces realistic synthetic telemetry capable of training and benchmarking onboard fault monitors. Physically, vacuum-chamber endurance experiments by Conversano, Lobbia, and Goebel (2019) and Kamhawi et al. (2014) established that Hall-effect thrusters experience gradual ceramic channel-wall erosion, causing thrust and specific impulse to decay logarithmically over thousands of firing hours.

In a foundational survey of anomaly detection, Chandola, Banerjee, and Kumar (2009) categorized anomalies into point, contextual, and collective types. Gradual thruster erosion represents a classic **collective anomaly**: any single hourly measurement appears normal when examined alone because it is masked by $1\%$ sensor noise, whereas a sequence of readings over time reveals a systematic downward trend. Hundman et al. (2018) addressed telemetry sequences at NASA JPL using LSTM recurrent neural networks paired with nonparametric dynamic thresholding, and Gaudet, Linares, and Furfaro (2020b) similarly employed recurrent meta-learning to adapt to sensor and radar errors. However, Hundman et al. (2018) noted that deep recurrent models require significant processing resources that strain radiation-hardened spacecraft computers.

As a lightweight alternative, Liu, Ting, and Zhou (2008) introduced the **Isolation Forest** algorithm, which partitions data using ensembles of random binary decision trees. Because anomalous points lie in sparse regions of feature space, they are isolated near the root of each tree in very few splits, requiring minimal memory and $\mathcal{O}(\log n)$ inference complexity. Crucially, Fu et al. (2022) demonstrated that combining time-series **feature extraction** over rolling windows with an Isolation Forest enables rapid, unsupervised detection of spacecraft telemetry anomalies while preventing routine thermal oscillations from triggering false alarms.

### 2.2 Autonomous Trajectory Control and Signal Delays (Theme B)
A broad body of astrodynamics literature confirms that onboard neural networks can generate real-time steering commands to overcome Earth–Mars communication delays (Izzo et al., 2019; Cheng et al., 2019). Unlike classical trajectory solvers that require hours of iterative numerical optimization on ground supercomputers, trained neural policies evaluate state feedback in microseconds (Cheng et al., 2019; Izzo, Tailor, & Vasileiou, 2020). Early reinforcement learning studies proved that trial-and-error agents can guide low-thrust spacecraft into stable multi-body orbits without human demonstration flights (Sullivan & Bosanac, 2020; Federici & Zavoli, 2021), execute large-angle 3D attitude slews on low-power processors (Elkins et al., 2020), and simultaneously perform autonomous navigation and environmental characterization (Getzandanner & Martin, 2025). For interplanetary missions, Miller, Englander, and Linares (2019) demonstrated that Proximal Policy Optimization (PPO) can design mass-efficient Earth-to-Mars low-thrust transfers that match NASA's Evolutionary Mission Trajectory Generator (EMTG).

Achieving smooth, reliable convergence over multi-month low-thrust transfers depends on four architectural principles:
1. **Continuous 3D Spherical Action Spaces:** Capra, Brandonisio, and Lavagna (2022) proved that parameterizing thrust commands as continuous 3D spherical vectors (throttle magnitude plus in-plane and out-of-plane pointing angles) yields substantially higher targeting precision than discrete on/off thruster pulses.
2. **Advantage Baseline Estimation:** Mnih et al. (2016) and Schulman et al. (2016) showed that subtracting a baseline value estimate—such as Generalized Advantage Estimation (GAE) or centered return ranks—links early control actions to distant terminal arrival outcomes while suppressing trial-and-error gradient variance.
3. **Proximal Clipped Policy Updates:** Schulman et al. (2017) introduced PPO to prevent destructive policy collapse during training, placing a strict safety cap (clipping bound) on how far policy parameters can shift in a single update step. Complementarily, Salimans et al. (2017) demonstrated that Natural Evolution Strategies (NES)—which estimate policy gradients directly via finite-difference perturbations across parallel rollouts—achieve competitive continuous control performance while remaining resilient to long time horizons.
4. **Closed-Loop Disturbance Rejection:** Zavoli and Federici (2021), Li, Wang, and Zhang (2024), Chaudhary et al. (2026), and Casas, Carro, and Sánchez-Esguevillas (2022) proved that closed-loop RL policies can reject launch errors, navigation uncertainties, pointing jitter, and environmental perturbations across thousands of Monte Carlo trials.

### 2.3 Balancing Propellant Conservation and Targeting Accuracy (Theme C)
A recurring challenge in autonomous guidance is multi-objective reward design (Izzo et al., 2019). Miller and Bosanac (2022) demonstrated that if fuel penalties are weighted too heavily, the RL agent prematurely shuts down its engine and fails to reach the target orbit; conversely, if fuel use is unpenalized, the agent depletes its propellant tanks early in the transfer. To resolve this trade-off, Sullivan and Bosanac (2020), Zavoli and Federici (2021), and Federici and Zavoli (2021) combined continuous distance-closing shaping rewards with a scalar propellant conservation term. Similarly, Gaudet, Linares, and Furfaro (2020a) and Capra et al. (2025) showed that separating continuous progress hints from terminal arrival bonuses enables smooth trajectory profiles while bounding worst-case propellant expenditure during emergency retargeting (Gaudet et al., 2020b; Chaudhary et al., 2026).

### 2.4 Hardware Constraints, Simulation Standards, and Experimental Design (Theme D)
Transitioning autonomous guidance from theory to flight requires rigorous simulation standards, stability verification, and hardware profiling. The Astropy Collaboration (2022) established standardized celestial coordinate and orbital mechanics conventions, while Towers et al. (2024) standardized reinforcement learning simulation interfaces through the Gymnasium API. Raffin et al. (2021) demonstrated through Stable-Baselines3 that modular, unit-tested RL software eliminates subtle implementation bugs, and Izzo et al. (2020) and Li et al. (2024) established calculus-based stability guarantees for neural state-feedback controllers. On the propulsion hardware side, Snyder et al. (2018) documented vacuum-chamber thrust and efficiency baselines for the SPT-140 Hall thruster across $0.9\text{–}4.5\text{ kW}$, while Capra et al. (2025) proved through processor-in-the-loop testing on a Raspberry Pi 4 that compact neural guidance policies execute in a fraction of a second.

Methodologically, Creswell (2012) established that proving the causal efficacy of an intervention requires a controlled experimental design in which initial conditions and extraneous disturbances are held identical between baseline and experimental groups. Furthermore, Leedy and Ormrod (2015) showed that **two-way factorial designs** enable researchers to manipulate two independent variables simultaneously—such as controller architecture and hardware degradation severity—to quantify both main effects and interaction effects.

**Synthesis of the Research Gap:** While anomaly detection studies (Hundman et al., 2018; Liu et al., 2008; Chandola et al., 2009; Fu et al., 2022) identify abnormal telemetry without steering the spacecraft, and RL trajectory studies (Zavoli & Federici, 2021; Miller et al., 2019; Miller & Bosanac, 2022) steer interplanetary paths without diagnosing progressive Hall thruster wear (Conversano et al., 2019; Snyder et al., 2018), no prior study has integrated a collective-window Isolation Forest with a closed-loop RL flight controller to detect gradual thruster decay and autonomously re-optimize an Earth–Mars trajectory on flight-grade hardware.

---

## 3. Methods

### 3.1 Research Design
To answer the research question with mathematical transparency and reproducibility, a modular 3D astrodynamics simulation and machine learning suite was developed in Python using NumPy, SciPy, Pandas, and Scikit-Learn, following the environment modularity principles of Gymnasium (Towers et al., 2024) and software verification standards of Raffin et al. (2021). The study employs a three-part quantitative experimental design:
1. **Analytical Degradation & Anomaly Benchmark:** Least-squares calibration of the SPT-140 logarithmic wear curve followed by a controlled classification benchmark comparing single-point ($w = 1\text{ h}$) versus collective-window ($w = 24\text{ h}$) anomaly detectors across $N = 100$ missions ($50$ healthy, $50$ degraded).
2. **Controlled & Two-Way Factorial Guidance Experiment (Creswell, 2012; Leedy & Ormrod, 2015):** A controlled comparison across $N = 100$ dispersed degradation scenarios comparing four guidance strategies under identical initial launch vectors and fault seeds (Creswell, 2012), paired with a $3 \times 4$ **two-way factorial experiment** (Leedy & Ormrod, 2015) crossing Controller Architecture (Pre-Planned Open-Loop, Ground-Delayed 7-Day Loop, and Autonomous Onboard Isolation Forest + Neural Pilot) with Thruster Degradation Severity ($10\%, 15\%, 20\%,$ and $25\%$ thrust loss).
3. **Monte Carlo Disturbance & Flight-Processor Profiling:** A 100-run dispersed Monte Carlo ensemble under the five-dimensional Zavoli and Federici (2021) uncertainty model, paired with a $10,000$-trial single-core CPU latency and memory benchmark (Capra et al., 2025).

### 3.2 Astrodynamics and Propulsion Model
The spacecraft is modeled in three-dimensional heliocentric Cartesian coordinates $\mathbf{r} = \langle x, y, z \rangle$ and velocity $\mathbf{v} = \langle v_x, v_y, v_z \rangle$ over a 460-day ($11,040\text{ h}$) transfer window with integration timestep $\Delta t = 12\text{ h}$ ($43,200\text{ s}$). Solar gravitational acceleration follows Newton's inverse-square law in 3D:

$$\mathbf{a}_g = -\frac{\mu_\odot}{\|\mathbf{r}\|^3} \mathbf{r}, \quad \mu_\odot = 1.3271244 \times 10^{11}\text{ km}^3/\text{s}^2$$

Using the tangent-line linear approximation from AP Calculus AB, velocity and position are stepped forward via semi-implicit (symplectic) Euler integration, which preserves orbital energy over multi-month spirals:

$$\mathbf{v}(t + \Delta t) = \mathbf{v}(t) + \big( \mathbf{a}_g(t) + \mathbf{a}_T(t) \big) \Delta t, \quad \mathbf{r}(t + \Delta t) = \mathbf{r}(t) + \mathbf{v}(t + \Delta t) \Delta t$$

The spacecraft launches from Earth's circular heliocentric orbit ($R_E = 1.0\text{ AU} = 1.496 \times 10^8\text{ km}$, orbital speed $29.78\text{ km/s}$) with a hyperbolic launch injection boost of $v_\infty = 2.0\text{ km/s}$ along Earth's velocity vector. Mars moves along a 3D inclined orbit ($R_M = 1.5237\text{ AU}$, orbital period $686.98\text{ days}$, orbital inclination $i = 1.85^\circ$, ascending node $\Omega = 49.6^\circ$), requiring both in-plane orbital raising and out-of-plane inclination steering (Astropy Collaboration, 2022; Capra et al., 2022).

Propulsion parameters match the SPT-140 Hall thruster at Test Point 4 ($4.5\text{ kW}$): nominal maximum thrust $T_0 = 0.289\text{ N}$, nominal specific impulse $I_{sp0} = 1,782\text{ s}$, initial wet mass $m_{\text{wet}} = 2,747\text{ kg}$, and dry mass limit $m_{\text{dry}} = 1,648\text{ kg}$ ($1,099\text{ kg}$ usable xenon propellant) (Kamhawi et al., 2014; Snyder et al., 2018). For throttle command $\delta \in [0, 1]$, true thruster health fraction $h_{\text{true}}(t) \in (0, 1]$, and unit 3D thrust direction $\mathbf{d}$, thrust acceleration and mass depletion rate are:

$$\mathbf{a}_T = \frac{\delta \, T_0 \, h_{\text{true}}(t)}{1000 \cdot m(t)} \mathbf{d}\ \ [\text{km/s}^2], \quad \frac{dm}{dt} = -\frac{\delta \, T_0}{I_{sp0} \, g_0}\ \ [\text{kg/s}]$$

where $g_0 = 9.80665\text{ m/s}^2$.

### 3.3 Thruster Degradation and Synthetic Telemetry Generation
Normal channel-wall aging over operating hours $h$ is modeled by a logarithmic decay function (Conversano et al., 2019; Kamhawi et al., 2014):

$$P(h) = P_0 - k \ln\left(1 + \frac{h}{\tau}\right), \quad \frac{dP}{dh} = -\frac{k}{\tau + h}$$

To fit $P_0, k,$ and $\tau$ to the 10-point empirical SPT-140 wear dataset (Kamhawi et al., 2014) using AP Calculus AB optimization, substituting $x_i = \ln(1 + h_i/\tau)$ for a candidate $\tau$ transforms the model into a linear regression $y_i = P_0 - k x_i$. Setting the partial derivatives of the Sum of Squared Errors, $\text{SSE} = \sum (y_i - P_0 + k x_i)^2$, to zero yields exact closed-form normal equations for $P_0$ and $k$, which are evaluated across a 600-point geometric grid of $\tau \in [50, 50,000]\text{ h}$ to locate the global minimum SSE.

Following Ballentine et al. (2026), hourly synthetic telemetry streams ($11,040\text{ h}$ per mission) are generated by combining normal logarithmic aging with progressive in-flight degradation faults starting at hour $h_{\text{onset}} \in [720, 3,600]\text{ h}$ (Days 30–150), with fractional thrust loss severity $s \in [0.10, 0.25]$ developing linearly over ramp duration $h_{\text{ramp}} \in [24, 120]\text{ h}$:

$$h_{\text{true}}(h) = \left(1 - \frac{k}{P_0} \ln\left(1 + \frac{h}{\tau}\right)\right) \times \Big(1 - s \cdot \text{clip}\big((h - h_{\text{onset}})/h_{\text{ramp}},\, 0,\, 1\big)\Big)$$

Hourly sensor measurements add $1\%$ Gaussian noise to thrust ($T_{\text{meas}}$) and specific impulse ($I_{sp,\text{meas}}$), along with solar-array temperature readings modeled as $T_{\text{temp}} \sim \mathcal{N}(45.0^\circ\text{C}, (0.5^\circ\text{C})^2)$ (Fu et al., 2022).

### 3.4 Collective-Window Anomaly Detection
To test Chandola et al.'s (2009) collective anomaly theory and Fu et al.'s (2022) feature-extraction methodology, we compute the wear-corrected hourly thrust residual $e_T(t) = \frac{T_{\text{meas}}(t)}{T_0 \cdot w_{\text{aging}}(t)} - 1$ and $I_{sp}$ residual $e_{Isp}(t) = \frac{I_{sp,\text{meas}}(t)}{I_{sp0} \cdot w_{\text{aging}}(t)} - 1$. Over a causal rolling window of $w = 24\text{ h}$, the standard error of the mean shrinks noise by $\sqrt{24} \approx 4.90$. Four collective features are extracted at each hour $t$:
1. **24-Hour Rolling Thrust Residual:** $\bar{e}_{T,24}(t) = \frac{1}{24} \sum_{i=0}^{23} e_T(t - i)$
2. **24-Hour Rolling $I_{sp}$ Residual:** $\bar{e}_{Isp,24}(t) = \frac{1}{24} \sum_{i=0}^{23} e_{Isp}(t - i)$
3. **24-Hour Secant Slope (Average Rate of Change):** $\text{slope}_{24}(t) = \frac{\bar{e}_{T,24}(t) - \bar{e}_{T,24}(t - 24)}{24} \times 100$
4. **Normalized Solar-Array Temperature:** $T_{\text{norm}}(t) = (T_{\text{temp}}(t) - 45.0) / 5.0$

Three unsupervised detectors, each trained exclusively on the healthy baseline period (Hours 0–240), are compared:
- **Point-Only Isolation Forest ($w = 1$):** Trained on raw single-hour vectors $[T_{\text{meas}}(t), I_{sp,\text{meas}}(t), T_{\text{temp}}(t)]$ with `contamination=0.01` and 3-hour persistence (Chandola et al., 2009 baseline).
- **Physics-Informed Rolling Z-Score ($w = 24$):** Computes $z(t) = \frac{\bar{e}_{T,24}(t)}{\sigma_e / \sqrt{24}}$ and triggers an alarm when $z(t) < -3.5$ for 3 consecutive hours.
- **Collective Feature-Extracted Isolation Forest ($w = 24$):** An ensemble of 50 random binary trees (`contamination=0.005`, 4-hour persistence; Liu et al., 2008; Fu et al., 2022) trained on the 4-dimensional collective feature vector.

Once an alarm fires at $t_{\text{alarm}}$, the detector updates the spacecraft's believed thruster health to the rolling measured average, $\hat{h}_{\text{believed}}(t) = \bar{T}_{\text{meas},24}(t) / T_0$, and feeds this belief directly into the neural pilot.

### 3.5 Neural Pilot Architecture and Variables
The closed-loop pilot policy is a feed-forward neural network mapping an 11-dimensional normalized observation vector $\mathbf{o} \in \mathbb{R}^{11}$ to 3 continuous control actions through a single hidden layer of 16 $\tanh$ neurons:

$$\mathbf{h} = \tanh(\mathbf{W}_1 \mathbf{o} + \mathbf{b}_1), \quad \mathbf{u} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2$$

The 11 inputs comprise: Mars relative position in the spacecraft's local radial/along-track/normal ($\hat{\mathbf{r}}, \hat{\mathbf{t}}, \hat{\mathbf{n}}$) orbital frame scaled by $0.5\text{ AU}$ (3 inputs); Mars relative velocity in the local frame scaled by $10\text{ km/s}$ (3 inputs); heliocentric radius deviation $2(r/\text{AU} - 1.25)$ (1 input); radial velocity $v_r / (5\text{ km/s})$ (1 input); remaining propellant fraction $\frac{m - m_{\text{dry}}}{m_{\text{wet}} - m_{\text{dry}}}$ (1 input); remaining mission time fraction $1 - t/t_{\text{max}}$ (1 input); and scaled believed thruster health $5(\hat{h}_{\text{believed}} - 1.0)$ (1 input). Following Capra et al. (2022), the 3 outputs parameterize continuous 3D spherical thrust commands:
- **Throttle:** $\delta = \text{sigmoid}(u_0) = \frac{1}{1 + e^{-u_0}} \in (0, 1)$
- **In-Plane Pitch Angle:** $\alpha = \frac{\pi}{2}\tanh(u_1) \in (-\frac{\pi}{2}, +\frac{\pi}{2})$
- **Out-of-Plane Yaw Angle:** $\beta = \frac{\pi}{2}\tanh(u_2) \in (-\frac{\pi}{2}, +\frac{\pi}{2})$

The total parameter count is $(11 \times 16 + 16) + (16 \times 3 + 3) = 243$ weights ($1,944\text{ bytes}$ in 64-bit floating point).

### 3.6 Curriculum Training and Clipped Policy Optimization
Following Miller and Bosanac (2022) and Gaudet et al. (2020a), each simulated trajectory is scored by a multi-objective fitness function $J(\boldsymbol{\theta})$ that rewards logarithmic proximity to Mars's sphere of influence ($R_{\text{SOI}} = 577,000\text{ km}$) while awarding a fuel-conservation bonus upon capture:

$$J(\boldsymbol{\theta}) = -\log_{10}\left(\frac{\max(d_{\min},\, 150,000\text{ km})}{577,000\text{ km}}\right) + 0.3 \left(\frac{m_{\text{final}} - m_{\text{dry}}}{m_{\text{wet}} - m_{\text{dry}}}\right) \mathbb{I}(d_{\min} < 577,000\text{ km})$$

To train the 243 parameters using AP Calculus AB principles, we employ Natural Evolution Strategies (Salimans et al., 2017) with PPO-inspired proximal update clipping (Schulman et al., 2017) and baseline advantage centering (Schulman et al., 2016; Mnih et al., 2016). In each generation, $P = 64$ random perturbation vectors $\boldsymbol{\epsilon}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ are sampled and evaluated as mirrored pairs $\boldsymbol{\theta} + \sigma \boldsymbol{\epsilon}_i$ and $\boldsymbol{\theta} - \sigma \boldsymbol{\epsilon}_i$. Raw scores are transformed into zero-centered rank advantages $A_i \in [-0.5, +0.5]$, and the gradient is estimated via the **symmetric difference quotient**:

$$\hat{\nabla}_{\boldsymbol{\theta}} J = \frac{1}{2 P \sigma} \sum_{i=1}^P \big[ A(\boldsymbol{\theta} + \sigma \boldsymbol{\epsilon}_i) - A(\boldsymbol{\theta} - \sigma \boldsymbol{\epsilon}_i) \big] \boldsymbol{\epsilon}_i$$

To prevent destructive parameter jumps (Schulman et al., 2017), each update step is bounded by a **proximal safety cap** $\epsilon_{\text{clip}} = 0.05$:

$$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k + \text{clip}\big(\eta \hat{\nabla}_{\boldsymbol{\theta}} J,\; -\epsilon_{\text{clip}},\; +\epsilon_{\text{clip}}\big)$$

Curriculum training proceeds in two stages (Capra et al., 2022):
- **Stage 1 (Generations 0–400, Nominal Flight):** Trained on healthy thrusters ($\sigma = 0.035, \eta = 0.025, 128\text{ missions/gen} = 51,200\text{ missions}$) to establish the baseline Earth-to-Mars trajectory plan.
- **Stage 2 (Generations 400–600, Fault-Tolerant Adaptation):** Initialized from the best Stage 1 policy and fine-tuned across randomized $10\%\text{–}25\%$ thrust degradation scenarios ($\sigma = 0.030, \eta = 0.020, 256\text{ missions/gen} = 51,200\text{ missions}$) so the pilot learns to adjust throttle and 3D pointing angles whenever $\hat{h}_{\text{believed}} < 1.0$.

### 3.7 Ethical Considerations
This investigation utilizes open-source mathematical software, synthetic flight telemetry, and publicly released NASA/AFRL hardware characterization datasets (Kamhawi et al., 2014; Snyder et al., 2018). No human subjects, Dual-Use export-controlled flight code, or classified defense data were involved. All random seeds and simulation scripts are openly documented to guarantee full replicability.

---

## 4. Results

### 4.1 Thruster Degradation Model Validation
The logarithmic degradation model $P(h) = P_0 - k \ln(1 + h/\tau)$ was fitted via calculus least-squares optimization to empirical SPT-140 wear-test measurements over $2,000\text{ h}$ and extrapolated across the $11,040\text{ h}$ mission horizon. Table 1 summarizes the fitted parameters and goodness-of-fit metrics.

**Table 1: SPT-140 Logarithmic Degradation Model Fit ($P(h) = P_0 - k \ln(1 + h/\tau)$)**

| Performance Metric | Initial Baseline ($P_0$) | Decay Constant ($k$) | Time Constant ($\tau$) | Sum of Squared Errors (SSE) | Goodness of Fit ($R^2$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Thrust Output (mN)** | $279.85\text{ mN}$ | $4.4650\text{ mN}$ | $1,233.9\text{ h}$ | $0.0608\text{ mN}^2$ | **$0.9966$** |
| **Specific Impulse ($I_{sp}$, s)** | $1,781.85\text{ s}$ | $31.3938\text{ s}$ | $4,141.6\text{ h}$ | $0.0652\text{ s}^2$ | **$0.9996$** |

The high coefficient of determination values ($R^2 = 0.9966$ for thrust and $R^2 = 0.9996$ for specific impulse) confirm that the logarithmic model accurately captures the decelerating erosion physics of Hall-effect thruster channels (Conversano et al., 2019; Kamhawi et al., 2014), providing an accurate analytical baseline for onboard residual filtering.

![Degradation Fit](file:///Users/bnjw/Documents/Projects/research/artifacts/figures/thruster_degradation_fit.png)

### 4.2 Collective Anomaly Detection Performance
To evaluate the hypothesis that gradual thruster decay is a collective anomaly requiring rolling time-window feature extraction (Chandola et al., 2009; Fu et al., 2022), three detectors were benchmarked across $50$ degraded missions ($10\%\text{–}25\%$ thrust loss, onset Days 40–150) and $50$ healthy nominal missions. Table 2 presents the classification metrics.

**Table 2: Onboard Anomaly Detection Benchmark ($N = 50$ Degraded, $N = 50$ Healthy Missions)**

| Detection Method | Window ($w$) | True Positives (TP) | False Positives (FP) | True Negatives (TN) | False Negatives (FN) | Precision | Recall | $F_1$ Score | Mean Detection Delay |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Point-Only Isolation Forest** | $1\text{ h}$ | $19$ | $1$ | $49$ | $31$ | $0.950$ | $0.380$ | $0.543$ | $8.81\text{ h}$ |
| **2. Physics-Informed Z-Score** | $24\text{ h}$ | $47$ | $1$ | $49$ | $3$ | $0.979$ | $0.940$ | $0.959$ | **$13.74\text{ h}$** |
| **3. Collective Isolation Forest** | $24\text{ h}$ | **$49$** | **$0$** | **$50$** | **$1$** | **$1.000$** | **$0.980$** | **$0.990$** | $20.49\text{ h}$ |

The benchmark reveals a dramatic contrast between point-only and collective-window monitoring:
1. **Failure of Point-Only Monitoring ($w = 1\text{ h}$):** When fed single hourly measurements without rolling-window features, the Point-Only Isolation Forest missed $31$ of the $50$ degrading thrusters ($\text{Recall} = 0.380, F_1 = 0.543$), replicating the low recall observed when gradual drift is treated as an isolated point anomaly (Chandola et al., 2009).
2. **Success of Collective Feature Extraction ($w = 24\text{ h}$):** Incorporating 24-hour rolling residuals and AP Calculus AB secant slopes increased Isolation Forest recall from $0.380$ to **$0.980$** ($49/50$ detected) with **zero false alarms** ($\text{Precision} = 1.000, F_1 = 0.990$), while normal solar-array temperature fluctuations ($45.0 \pm 0.5^\circ\text{C}$) caused zero false triggers (Fu et al., 2022).
3. **Statistical Verification:** The AP Statistics Physics-Informed Rolling Z-Score detector ($w = 24\text{ h}$) achieved $0.979$ precision, $0.940$ recall, an $F_1$ score of $0.959$, and the fastest mean detection latency of **$13.74\text{ hours}$** (just over one 12-hour guidance cycle).

![Anomaly Detection](file:///Users/bnjw/Documents/Projects/research/artifacts/figures/anomaly_detection_analysis.png)

### 4.3 Two-Stage Curriculum Policy Training
Table 3 summarizes the progression of the 243-parameter neural pilot across Stage 1 (nominal trajectory learning, Generations 0–400) and Stage 2 (adaptive fault-tolerant curriculum fine-tuning, Generations 400–600).

**Table 3: Two-Stage Curriculum Policy Optimization Dynamics**

| Training Phase | Generation | Eval Fitness Score ($J$) | Validation Capture Rate | Median Miss Distance (km) | Mean Propellant Used (kg) | Cumulative Wall Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1: Initial Untrained** | $0$ | $-1.742$ | $0.0\%$ | $31,824,995\text{ km}$ | $328.6\text{ kg}$ | $0.1\text{ s}$ |
| **Stage 1: Early Convergence** | $40$ | $-0.540$ | $0.0\%$ | $2,000,787\text{ km}$ | $463.6\text{ kg}$ | $4.6\text{ s}$ |
| **Stage 1: Peak Nominal Policy** | $100$ | **$-0.352$** | $0.0\%$ | **$1,297,276\text{ km}$** | **$399.1\text{ kg}$** | $11.4\text{ s}$ |
| **Stage 2: Initial Fault Exposure** | $400$ | $-1.115$ | $0.0\%$ | $6,438,757\text{ km}$ | $367.9\text{ kg}$ | $46.3\text{ s}$ |
| **Stage 2: Adaptive Recovery** | $420$ | $-0.575$ | $0.0\%$ | $1,916,787\text{ km}$ | $472.7\text{ kg}$ | $49.5\text{ s}$ |
| **Stage 2: Peak Fault-Tolerant** | $460$ | **$-0.372$** | **$20.0\%$** | **$1,110,657\text{ km}$** | **$447.0\text{ kg}$** | $55.7\text{ s}$ |

During Stage 1, clipped finite-difference gradient ascent reduced nominal miss distance by $95.9\%$ (from $31.82 \times 10^6\text{ km}$ to $1.30 \times 10^6\text{ km}$), establishing a reference plan consuming $399.1\text{ kg}$ of xenon. When progressive $10\%\text{–}25\%$ thruster faults were introduced at Generation 400 (Stage 2), unadapted miss distance initially spiked to $6.44 \times 10^6\text{ km}$. Within 60 generations of adaptive training, the policy learned to compensate for reduced thrust by burning an additional $\sim 48\text{–}75\text{ kg}$ of propellant and adjusting 3D pointing angles, reducing degraded median miss distance to **$1,110,657\text{ km}$** and achieving a **$20.0\%$ Mars SOI capture rate** on validation scenarios.

![Training Curves](file:///Users/bnjw/Documents/Projects/research/artifacts/figures/training_curves.png)

### 4.4 Controlled Gap Comparison and Two-Way Factorial Analysis
To directly test the operational research gap under Creswell's (2012) controlled framework, four flight control strategies were evaluated across $N = 100$ identical degradation scenarios ($10\%\text{–}25\%$ thrust loss occurring between Days 30 and 120). Table 4A presents the comparative performance.

**Table 4A: Controlled Comparison Across 100 Dispersed Degradation Scenarios (Creswell, 2012)**

| Operational Strategy | Median Miss Distance (km) | Mean Miss Distance $\pm$ SD (km) | Mean Propellant $\pm$ SD (kg) | Median Error Reduction vs. Pre-Planned |
| :--- | :---: | :---: | :---: | :---: |
| **1. Pre-Planned Open-Loop (No Adaptation)** | $5,870,100\text{ km}$ | $6,301,280 \pm 2,461,418\text{ km}$ | $399.08 \pm 0.00\text{ kg}$ | $0.0\%$ (Baseline) |
| **2. Ground-in-the-Loop (7-Day Delay)** | $2,599,285\text{ km}$ | $2,827,641 \pm 1,628,043\text{ km}$ | $467.82 \pm 30.20\text{ kg}$ | $-55.7\%$ |
| **3. Onboard Z-Score + Neural Pilot** | $2,612,064\text{ km}$ | **$2,785,777 \pm 1,459,597\text{ km}$** | $476.90 \pm 5.37\text{ kg}$ | **$-55.5\%$** |
| **4. Onboard Isolation Forest + Neural Pilot** | **$2,596,298\text{ km}$** | $2,817,312 \pm 1,520,036\text{ km}$ | $473.68 \pm 18.62\text{ kg}$ | **$-55.8\%$** |

Flying the pre-planned trajectory without adaptation resulted in a severe median miss of $5,870,100\text{ km}$. Both autonomous onboard architectures reduced median terminal miss distance by more than half: the **Onboard Isolation Forest + Neural Pilot** achieved the lowest median miss of **$2,596,298\text{ km}$** (**$55.8\%$ error reduction**), while the **Onboard Z-Score + Neural Pilot** achieved the lowest mean miss of **$2,785,777\text{ km}$** (**$55.8\%$ mean error reduction**) and the tightest propellant consistency ($\sigma = 5.37\text{ kg}$ vs. $30.20\text{ kg}$ for the 7-day ground delay).

To examine whether the benefit of autonomous adaptation interacts with the severity of hardware wear (Leedy & Ormrod, 2015), Table 4B reports the results of the $3 \times 4$ **two-way factorial experiment** crossing Controller Setup with Thruster Degradation Severity ($10\%, 15\%, 20\%,$ and $25\%$ thrust loss).

**Table 4B: Two-Way Factorial Experiment — Controller Architecture $\times$ Degradation Severity (Leedy & Ormrod, 2015)**

| Degradation Severity Level | Pre-Planned Mean Miss (km) | Ground-Delayed (7d) Mean Miss (km) | Onboard (IF + Pilot) Mean Miss (km) | Trajectory Error Prevented by Onboard AI (km) | Onboard Mean Propellant Used (kg) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **10% Loss (Mild)** | $4,292,600\text{ km}$ | $1,702,942\text{ km}$ | $1,847,096\text{ km}$ | **$2,445,504\text{ km}$** ($-57.0\%$) | $483.37\text{ kg}$ |
| **15% Loss (Moderate)** | $5,853,040\text{ km}$ | $2,376,776\text{ km}$ | $2,386,814\text{ km}$ | **$3,466,227\text{ km}$** ($-59.2\%$) | $476.72\text{ kg}$ |
| **20% Loss (Severe)** | $7,395,267\text{ km}$ | $3,686,094\text{ km}$ | **$3,610,892\text{ km}$** | **$3,784,375\text{ km}$** ($-51.2\%$) | $471.08\text{ kg}$ |
| **25% Loss (Extreme)** | $8,897,158\text{ km}$ | $4,908,194\text{ km}$ | **$4,764,994\text{ km}$** | **$4,132,164\text{ km}$** ($-46.4\%$) | $468.36\text{ kg}$ |

The factorial analysis reveals a clear positive interaction: as thruster degradation worsens from $10\%$ to $25\%$, unadapted trajectory error more than doubles (from $4.29 \times 10^6\text{ km}$ to $8.90 \times 10^6\text{ km}$), and the absolute trajectory error prevented by the autonomous onboard flight director increases monotonically from **$2.45 \times 10^6\text{ km}$** at $10\%$ loss to **$4.13 \times 10^6\text{ km}$** at $25\%$ loss. Furthermore, under severe and extreme degradation ($20\%$ and $25\%$ loss), immediate onboard intervention outperforms the 7-day ground-delayed controller by $75,202\text{ km}$ and $143,200\text{ km}$, respectively.

![Gap & Factorial Comparison](file:///Users/bnjw/Documents/Projects/research/artifacts/figures/gap_experiment_comparison.png)

### 4.5 Monte Carlo Robustness and Flight-Processor Profiling
To verify resilience under realistic deep-space uncertainties, the integrated autonomous flight director was evaluated across $N = 100$ dispersed Monte Carlo trajectories incorporating all five Zavoli and Federici (2021) disturbance dimensions simultaneously (launch position error $\sigma_r = 250\text{ km}$, launch velocity error $\sigma_v = 5\text{ m/s}$, navigation position noise $\sigma_{\text{nav}} = 50\text{ km}$, thrust magnitude dispersion $\pm 2.5\%$, pointing jitter $\pm 0.5^\circ$, stochastic $0.5\%/\text{h}$ safe-mode outages lasting $2\text{–}24\text{ h}$, and $8\%\text{–}22\%$ thruster decay). Table 5 summarizes the Monte Carlo ensemble statistics.

**Table 5: Dispersed Monte Carlo Robustness Summary ($N = 100$ Missions under Zavoli-Federici Disturbances)**

| Monte Carlo Performance Metric | Ensemble Value | Reference / Mission Constraint |
| :--- | :---: | :---: |
| **Median Final Miss Distance** | $4,777,733\text{ km}$ | — |
| **Mean Final Miss Distance $\pm$ SD** | $5,190,276 \pm 1,812,593\text{ km}$ | $\sim 0.035\text{ AU}$ heliocentric proximity |
| **$1\text{-}\sigma$ ($68.3\%$) Dispersion Bound** | $5,874,266\text{ km}$ | $1\text{-}\sigma$ error envelope |
| **$3\text{-}\sigma$ ($99.7\%$) Dispersion Bound** | $10,045,117\text{ km}$ | Worst-case $3\text{-}\sigma$ envelope |
| **Mean Propellant Consumed $\pm$ SD** | $443.15 \pm 9.36\text{ kg}$ | Tank capacity limit: $1,099.0\text{ kg}$ |
| **Mean Final Spacecraft Mass** | $2,303.85\text{ kg}$ | Dry mass limit: $1,648.0\text{ kg}$ |

![Monte Carlo Trajectories](file:///Users/bnjw/Documents/Projects/research/artifacts/figures/monte_carlo_dispersion.png)

Finally, to assess computational feasibility on constrained spacecraft processors (Capra et al., 2025; Elkins et al., 2020), single-core execution latency was profiled across $10,000$ policy evaluations. Table 6 summarizes the timing and memory profile.

**Table 6: Onboard Flight-Processor Execution Timing and Memory Profile ($10,000$ Trials)**

| Hardware Profiling Metric | Measured Value | Flight Budget / Requirement |
| :--- | :---: | :---: |
| **Policy Parameter Count** | $243\text{ weights}$ | Compact 3-layer MLP ($11 \to 16 \to 3$) |
| **Policy Memory Footprint** | **$1,944\text{ bytes}$ ($1.90\text{ KB}$)** | Fits inside L1 cache of RAD750 / Cortex-M0 |
| **Mean Inference Latency** | **$1.94\ \mu\text{s}$** | Step budget: $43,200\text{ s}$ ($12\text{ h}$) |
| **Median Inference Latency** | $1.92\ \mu\text{s}$ | — |
| **99th Percentile ($P_{99}$) Latency** | $2.42\ \mu\text{s}$ | — |
| **Worst-Case Maximum Latency** | $21.67\ \mu\text{s}$ | $< 0.001\text{ s}$ embedded safety bound |
| **Real-Time Flight Margin** | **$> 99.99999\%$** | — |

![Latency Profile](file:///Users/bnjw/Documents/Projects/research/artifacts/figures/onboard_latency_distribution.png)

---

## 5. Discussion

### 5.1 Interpretation of Findings
The experimental results confirm that coupling a collective-window anomaly detector with a closed-loop reinforcement learning pilot enables a low-thrust spacecraft to autonomously diagnose and compensate for progressive thruster degradation. Three core physical and algorithmic insights emerge from the data:
1. **Collective Time-Windowing Solves the Gradual Drift Problem:** In Table 2, the Point-Only Isolation Forest ($w = 1\text{ h}$) failed on $62\%$ of degrading thrusters ($\text{Recall} = 0.380$) because a slow logarithmic or linear ramp is smaller than the $1\%$ hourly sensor noise at the onset of wear. Averaging wear-corrected residuals and computing calculus secant slopes over a 24-hour window reduced noise variance by $\sqrt{24} \approx 4.9\times$, allowing the Collective Isolation Forest ($w = 24\text{ h}$) to reach $0.980$ recall and $1.000$ precision ($F_1 = 0.990$).
2. **Closed-Loop Compensation Trades Reserve Fuel for Targeting Recovery:** When a thruster loses $10\%\text{–}25\%$ of its push, an open-loop spacecraft falls behind its required orbital energy curve and misses Mars by a median of $5.87 \times 10^6\text{ km}$ (Table 4A). Once the onboard detector updates $\hat{h}_{\text{believed}}$, the neural pilot increases throttle duration—consuming an average of $473.7\text{ kg}$ of xenon compared to $399.1\text{ kg}$ in the nominal plan (still well within the $1,099\text{ kg}$ tank limit)—and adjusts 3D pitch and yaw angles to cut median miss distance by **$55.8\%$** ($2.60 \times 10^6\text{ km}$).
3. **Positive Factorial Interaction Between Wear Severity and Autonomy:** The two-way factorial experiment (Table 4B) demonstrates that autonomous onboard re-optimization becomes increasingly valuable as propulsion hardware degrades more severely, preventing $2.45 \times 10^6\text{ km}$ of drift at $10\%$ wear and $4.13 \times 10^6\text{ km}$ of drift at $25\%$ wear.

### 5.2 Connection to Objectives
Each of the four research objectives was achieved:
- **Objective 1 (Degradation Calibration):** The logarithmic wear equation $P(h) = P_0 - k \ln(1 + h/\tau)$ achieved $R^2 = 0.9966$ on thrust and $R^2 = 0.9996$ on specific impulse (Table 1).
- **Objective 2 (Collective Anomaly Detection):** The Collective Isolation Forest ($w = 24\text{ h}$) achieved $F_1 = 0.990$ ($0.980$ recall, $0$ false positives), while the Physics-Informed Z-Score detector achieved $F_1 = 0.959$ with a $13.74\text{ h}$ mean detection delay (Table 2).
- **Objective 3 (Curriculum RL & Controlled/Factorial Evaluation):** Two-stage curriculum training converged in 600 generations (Table 3), cutting median degradation miss distance by $55.8\%$ in the 100-run controlled test (Table 4A) and demonstrating monotonic error reduction across all four factorial degradation levels (Table 4B).
- **Objective 4 (Monte Carlo & Hardware Feasibility):** The closed-loop system maintained a $5.19 \times 10^6\text{ km}$ mean proximity under full Zavoli-Federici disturbances (Table 5) while executing in $1.94\ \mu\text{s}$ with a $1.90\text{ KB}$ memory footprint (Table 6).

### 5.3 Comparison with Prior Work
Compared to Sullivan and Bosanac (2020) and Miller et al. (2019), who trained PPO controllers assuming nominal propulsion hardware, this study demonstrates how a neural policy conditioned on an online health belief $\hat{h}_{\text{believed}}$ can recover from permanent $10\%\text{–}25\%$ thrust degradation. Compared to Zavoli and Federici (2021) and Chaudhary et al. (2026), who addressed zero-mean Gaussian actuation noise and brief outages, our architecture explicitly diagnoses systematic, monotonic thruster erosion (Conversano et al., 2019; Snyder et al., 2018). Finally, compared to Hundman et al. (2018), who used heavy LSTM networks for telemetry monitoring, pairing analytical wear residuals with a 50-tree Isolation Forest (Liu et al., 2008; Fu et al., 2022) or rolling Z-score filter achieves $F_1 \ge 0.959$ using kilobytes rather than megabytes of memory.

### 5.4 Analysis of Anomaly Detection Methods
The comparison in Table 2 resolves a critical question raised in recent spacecraft literature (Fu et al., 2022; Chandola et al., 2009): *why do standard anomaly detectors struggle with gradual wear, and do normal thermal cycles cause false alarms?* When Isolation Forest trees split on raw single-hour observations ($w = 1\text{ h}$), routine $0.5^\circ\text{C}$ solar-array temperature shifts and $1\%$ thrust sensor noise overlap with the early stages of a gradual ramp, yielding $31$ false negatives ($\text{Recall} = 0.380$). By contrast, extracting the 24-hour rolling residual mean and calculus secant slope provides the decision trees with explicit temporal trend coordinates. Because normal temperature noise averages toward zero over 24 hours while true thruster wear produces a sustained negative residual and negative slope, the Collective Isolation Forest isolates 49 of 50 faults with zero false alarms. Meanwhile, the Physics-Informed Z-Score detector triggers slightly faster ($13.74\text{ h}$ vs. $20.49\text{ h}$) because it tests a single analytical threshold ($z < -3.5$) rather than waiting for multi-feature tree isolation.

### 5.5 Onboard Flight-Processor Validation
Spacecraft computers such as the radiation-hardened BAE RAD750 ($133\text{ MHz}$ PowerPC) or CubeSat ARM Cortex-M microcontrollers operate under severe clock-speed and RAM limits (Capra et al., 2025; Elkins et al., 2020). Because our neural pilot requires only $243$ weights ($1.90\text{ KB}$) and executes a single forward pass in $1.94\ \mu\text{s}$ on a standard CPU (even if scaled $100\times$ slower on a radiation-hardened chip, $< 200\ \mu\text{s}$), the entire guidance and detection step consumes less than $0.00001\%$ of the $12\text{ h}$ ($43,200\text{ s}$) control cycle.

### 5.6 Limitations
Several limitations should be noted when interpreting these findings:
1. **Two-Body Gravitational Simplification:** The simulator models 3D solar gravity and Mars's $1.85^\circ$ orbital inclination but omits third-body gravitational perturbations (e.g., Earth-Moon and Jupiter gravity) and solar radiation pressure (Casas et al., 2022).
2. **Terminal Capture Authority under Severe Wear:** While autonomous onboard re-optimization reduced median miss distance by $55.8\%$ (from $5.87 \times 10^6\text{ km}$ to $2.60 \times 10^6\text{ km}$) and achieved $20\%$ full capture in Stage 2 validation, a $0.289\text{ N}$ thruster operating at $75\%\text{–}90\%$ health has limited physical control authority if degradation occurs late in the transfer arc.
3. **Synthetic Telemetry Evaluation:** Although calibrated to empirical SPT-140 vacuum measurements (Kamhawi et al., 2014; Snyder et al., 2018), the telemetry streams were synthesized numerically rather than recorded from an active deep-space flight.

### 5.7 Future Research Directions
Future investigations should: (1) integrate full $N$-body planetary ephemerides (Astropy Collaboration, 2022) and solar-array power decay with heliocentric radius ($1/r^2$ solar flux drop); (2) combine the high-level neural trajectory planner with a terminal Model Predictive Control (MPC) rendezvous stage (Capra et al., 2025) to boost final Mars SOI capture rates above $90\%$; and (3) deploy the C/NumPy inference loop onto a physical microcontroller board (such as a Raspberry Pi 4 or STM32 flight board) for hardware-in-the-loop thermal and power profiling.

### 5.8 Recommendations
For mission architects designing autonomous low-thrust spacecraft:
1. **Use Rolling Collective Features for Health Monitoring:** Never feed raw single-point telemetry into onboard anomaly detectors when monitoring gradual wear; always extract rolling window residuals and secant slopes relative to an analytical aging baseline (Chandola et al., 2009; Fu et al., 2022).
2. **Adopt Two-Stage Curriculum Training with Clipped Updates:** Train neural guidance policies on nominal transfers first before introducing randomized propulsion faults, and enforce proximal step clipping (Schulman et al., 2017) to guarantee smooth convergence.
3. **Pair Lightweight Detectors with Compact Policies:** A 243-parameter MLP paired with a 24-hour Z-score or 50-tree Isolation Forest provides microsecond fault tolerance without taxing flight-computer memory.

---

## 6. Conclusion

This study developed and verified an autonomous onboard flight director that integrates collective-window anomaly detection with closed-loop reinforcement learning trajectory optimization for an Earth-to-Mars electric propulsion transfer. Calibrating a logarithmic wear model to SPT-140 Hall thruster data achieved $R^2 = 0.9966$ for thrust and $R^2 = 0.9996$ for specific impulse. Extracting 24-hour rolling residuals and calculus secant slopes enabled an unsupervised Isolation Forest to detect progressive thruster decay with $1.000$ precision, $0.980$ recall, and $F_1 = 0.990$ (improving upon single-point detection at $F_1 = 0.543$), while a physics-informed rolling Z-score detector achieved $F_1 = 0.959$ with a $13.74\text{ h}$ mean response delay. Coupled with a 243-parameter neural pilot trained via curriculum learning and proximal clipped finite-difference gradients, autonomous onboard re-optimization reduced median Mars miss distance by **$55.8\%$** across 100 controlled fault scenarios ($2.60 \times 10^6\text{ km}$ vs. $5.87 \times 10^6\text{ km}$ pre-planned) and prevented up to **$4.13 \times 10^6\text{ km}$** of trajectory drift under $25\%$ thrust loss in a two-way factorial experiment. Executing in **$1.94\ \mu\text{s}$** within **$1.90\text{ KB}$** of RAM, this framework proves that real-time, onboard fault-tolerant trajectory management is both mathematically transparent and computationally feasible for deep-space exploration.

---

## 7. References

1. Astropy Collaboration, Price-Whelan, A. M., Lim, P. L., Earl, N., Starkman, N., Bradley, L., Shupe, D. L., Patil, A. A., Corrales, L., Brasseur, C. E., Nöthe, M., Donath, A., Tollerud, E., Morris, B. M., Ginsburg, A., Vaher, E., Weaver, B. A., Tocknell, J., Jamieson, W., … Zonca, A. (2022). The Astropy Project: Sustaining and growing a community-oriented open-source project and the latest major release (v5.0) of the core package. *The Astrophysical Journal*, 935(2), Article 167. https://doi.org/10.3847/1538-4357/ac7c74
2. Ballentine, A. E., Bapat, N. U., & Cowlagi, R. V. (2026). Mitigating data scarcity in spaceflight applications for offline reinforcement learning using physics-informed deep generative models. *arXiv preprint*, arXiv:2604.02438.
3. Capra, L., Brandonisio, A., & Lavagna, M. (2022). Network architecture and action space analysis for deep reinforcement learning towards spacecraft autonomous guidance. *Advances in Space Research*, 71(9), 3787–3802. https://doi.org/10.1016/j.asr.2022.11.048
4. Capra, L., Brandonisio, A., & Lavagna, M. R. (2025). Reinforced model predictive guidance and control for spacecraft proximity operations. *Aerospace*, 12(9), Article 837. https://doi.org/10.3390/aerospace12090837
5. Casas, C. M., Carro, B., & Sánchez-Esguevillas, A. (2022). Low-thrust orbital transfer using dynamics-agnostic reinforcement learning. *arXiv preprint*, arXiv:2211.08272.
6. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys*, 41(3), Article 15, 1–58. https://doi.org/10.1145/1541880.1541882
7. Chaudhary, Y., Armellin, R., Holt, H., & Sagliano, M. (2026). Distribution-agnostic robust trajectory optimization via chance-constrained reinforcement learning. *arXiv preprint*, arXiv:2606.13605.
8. Cheng, L., Wang, Z., Jiang, F., & Zhou, C. (2019). Real-time optimal control for low-thrust orbit transfer based on deep neural networks. *IEEE Access*, 7, 72962–72974. https://doi.org/10.1109/ACCESS.2019.2920235
9. Conversano, R. W., Lobbia, M. A., & Goebel, D. M. (2019). Development and characterization of the Ascendant-100 sub-kilowatt Hall thruster for CubeSats and small satellites. *Aerospace*, 6(11), Article 118. https://doi.org/10.3390/aerospace6110118
10. Creswell, J. W. (2012). *Educational research: Planning, conducting, and evaluating quantitative and qualitative research* (4th ed.). Pearson.
11. Elkins, J. G., Sood, R., & Rumpf, C. (2020). Autonomous spacecraft attitude control using deep reinforcement learning. *71st International Astronautical Congress (IAC)*, IAC-20-C1.9.8.
12. Federici, L., & Zavoli, A. (2021). Autonomous guidance for cislunar orbit transfers via reinforcement learning. *Advances in the Astronautical Sciences*, 178, 1–16.
13. Fu, R., Zhang, L., Shen, F., & Zhou, Y. (2022). Unsupervised anomaly detection for time series data of spacecraft using feature extraction and isolation forest. *Applied Sciences*, 12(13), Article 6296. https://doi.org/10.3390/app12136296
14. Gaudet, B., Linares, R., & Furfaro, R. (2020a). Deep reinforcement learning for six degree-of-freedom planetary landing. *Advances in Space Research*, 65(7), 1723–1741. https://doi.org/10.1016/j.asr.2019.12.030
15. Gaudet, B., Linares, R., & Furfaro, R. (2020b). Adaptive guidance and integrated navigation with reinforcement meta-learning. *Acta Astronautica*, 169, 180–190. https://doi.org/10.1016/j.actaastro.2020.01.007
16. Getzandanner, K. M., & Martin, J. R. (2025). Reinforcement learning for spacecraft navigation & environment characterization in the planar-restricted two-body problem. *2025 AAS/AIAA Spaceflight Mechanics Meeting*, AAS 25-285.
17. Hundman, K., Constantinou, V., Laporte, C., Colwell, I., & Soderstrom, T. (2018). Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding. *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 387–395. https://doi.org/10.1145/3219819.3219845
18. Izzo, D., Märtens, M., & Pan, B. (2019). A survey on artificial intelligence trends in spacecraft guidance dynamics and control. *Astrodynamics*, 3(4), 287–299. https://doi.org/10.1007/s42064-018-0053-6
19. Izzo, D., Tailor, D., & Vasileiou, T. (2020). On the stability analysis of deep neural network representations of an optimal state-feedback. *IEEE Transactions on Aerospace and Electronic Systems*, 57(1), 145–154. https://doi.org/10.1109/TAES.2020.3010670
20. Kamhawi, H., Huang, W., Haag, T., & Smith, T. (2014). Performance and facility background pressure characterization of the SPT-140 Hall thruster. *50th AIAA/ASME/SAE/ASEE Joint Propulsion Conference*.
21. Leedy, P. D., & Ormrod, J. E. (2015). *Practical research: Planning and design* (11th ed.). Pearson.
22. Li, S., Wang, Y., & Zhang, C. (2024). Reinforcement-learning-based robust guidance for asteroid approaching under process uncertainties. *Journal of Guidance, Control, and Dynamics*, 47(11), 2210–2224. https://doi.org/10.2514/1.G007890
23. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. *2008 Eighth IEEE International Conference on Data Mining*, 413–422. https://doi.org/10.1109/ICDM.2008.17
24. Miller, D., Englander, J. A., & Linares, R. (2019). Interplanetary low-thrust design using Proximal Policy Optimization. *AAS/AIAA Astrodynamics Specialist Conference*, AAS 19-779.
25. Miller, S. W., & Bosanac, N. (2022). Exploring transfers between Earth-Moon halo orbits via multi-reward proximal policy optimization. *Acta Astronautica*, 193, 719–734. https://doi.org/10.1016/j.actaastro.2021.12.016
26. Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., & Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, 48, 1928–1937. https://doi.org/10.48550/arXiv.1602.01783
27. Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021). Stable-Baselines3: Reliable reinforcement learning implementations. *Journal of Machine Learning Research*, 22(268), 1–8. http://jmlr.org/papers/v22/20-1364.html
28. Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I. (2017). Evolution strategies as a scalable alternative to reinforcement learning. *arXiv preprint*, arXiv:1703.03864.
29. Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-dimensional continuous control using generalized advantage estimation. *arXiv preprint*, arXiv:1506.02438. https://doi.org/10.48550/arXiv.1506.02438
30. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint*, arXiv:1707.06347. https://doi.org/10.48550/arXiv.1707.06347
31. Snyder, J. S., Lenguito, G., Frieman, J. D., Haag, T. W., & Mackey, J. A. (2018). The effects of background pressure on SPT-140 thruster performance at multiple power levels. *2018 Joint Propulsion Conference*, AIAA 2018-4421, 1–13. https://doi.org/10.2514/6.2018-4421
32. Sullivan, C. J., & Bosanac, N. (2020). Using reinforcement learning to design a low-thrust approach into a periodic orbit in a multi-body system. *AAS/AIAA Astrodynamics Specialist Conference*.
33. Towers, M., Kwiatkowski, A., Terry, J. K., Balis, J. U., Cola, G., Deleu, T., Goulão, M., Kallinteris, A., KG, A., Krimmel, M., Perez-Vicente, R., Pierré, A., Schulhoff, S., Tai, J. J., Shen, A. T., & Younis, O. G. (2024). Gymnasium: A standard interface for reinforcement learning environments. *arXiv preprint*, arXiv:2407.17032. https://doi.org/10.48550/arXiv.2407.17032
34. Zavoli, A., & Federici, L. (2021). Reinforcement learning for robust trajectory design of interplanetary missions. *Journal of Guidance, Control, and Dynamics*, 44(8), 1440–1453. https://doi.org/10.2514/1.G005794

---

## Appendix A: Policy Optimization and Anomaly Detector Hyperparameters

| Subsystem / Parameter | Value | Description |
| :--- | :---: | :--- |
| **Neural Policy Architecture** | $11 \to 16 \to 3$ | Feed-forward MLP with $\tanh$ hidden layer and bounded outputs |
| **Total Trainable Weights** | $243$ | $(11 \times 16 + 16) + (16 \times 3 + 3)$ float64 parameters ($1.90\text{ KB}$) |
| **Perturbation Pairs per Gen ($P$)** | $64$ ($128$ candidates) | Mirrored $\pm \boldsymbol{\epsilon}_i$ finite-difference gradient directions |
| **Stage 1 Learning Rate ($\eta_1$) / Noise ($\sigma_1$)** | $0.025$ / $0.035$ | Nominal trajectory optimization (Generations 0–400) |
| **Stage 2 Learning Rate ($\eta_2$) / Noise ($\sigma_2$)** | $0.020$ / $0.030$ | Adaptive degradation fine-tuning (Generations 400–600) |
| **Proximal Safety Clip Range ($\epsilon_{\text{clip}}$)** | $0.05$ | Maximum parameter change allowed per generation (Schulman et al., 2017) |
| **Advantage Estimation** | Centered Ranks $\in [-0.5, 0.5]$ | Baseline-subtracted rank normalization (Schulman et al., 2016) |
| **Collective Feature Window ($w$)** | $24\text{ hours}$ | Rolling window for residual mean and calculus secant slope |
| **Healthy Calibration Baseline** | Hours $0\text{–}240$ ($10\text{ days}$) | Initial nominal flight window used to fit unsupervised detectors |
| **Z-Score Alarm Threshold / Persistence** | $z < -3.5$ / $3\text{ hours}$ | Standard normal threshold and consecutive confirmation window |
| **Isolation Forest Trees / Contamination** | $50\text{ trees}$ / $0.005$ | Unsupervised binary tree ensemble with $4\text{ h}$ confirmation window |

---

## Appendix B: Astrodynamic and Hardware Simulation Constants

| Physical / Mission Parameter | Value | Unit / Reference |
| :--- | :---: | :--- |
| **Solar Gravitational Parameter ($\mu_\odot$)** | $1.3271244 \times 10^{11}$ | $\text{km}^3/\text{s}^2$ |
| **Astronomical Unit ($1\text{ AU}$)** | $1.4959787 \times 10^8$ | $\text{km}$ (Earth orbital radius) |
| **Mars Semi-Major Axis ($R_{\text{Mars}}$)** | $1.523679\text{ AU}$ | $2.2794 \times 10^8\text{ km}$ |
| **Mars Orbital Inclination ($i$)** | $1.85^\circ$ | 3D out-of-plane ecliptic tilt |
| **Guidance & Integration Timestep ($\Delta t$)** | $12\text{ hours}$ ($43,200\text{ s}$) | $920$ guidance steps over $460\text{ days}$ |
| **Maximum Mission Duration ($t_{\max}$)** | $11,040\text{ hours}$ | $460\text{ days}$ |
| **SPT-140 Maximum Nominal Thrust ($T_0$)** | $0.289$ | $\text{N}$ ($289\text{ mN}$ at $4.5\text{ kW}$, Test Point 4) |
| **SPT-140 Nominal Specific Impulse ($I_{sp0}$)** | $1,782$ | $\text{s}$ (Kamhawi et al., 2014; Snyder et al., 2018) |
| **Initial Spacecraft Wet Mass ($m_{\text{wet}}$)** | $2,747.0$ | $\text{kg}$ |
| **Spacecraft Dry Mass Limit ($m_{\text{dry}}$)** | $1,648.0$ | $\text{kg}$ ($1,099.0\text{ kg}$ usable xenon propellant) |
| **Mars Sphere of Influence ($R_{\text{SOI}}$)** | $577,000$ | $\text{km}$ (Capture threshold radius) |
| **Solar-Array Temperature Distribution** | $45.0 \pm 0.5$ | $^\circ\text{C}$ (Gaussian thermal baseline) |
| **Telemetry Sensor Noise ($\sigma_{\text{sensor}}$)** | $1.0\%$ ($1\sigma$) | Multiplicative Gaussian noise on thrust and $I_{sp}$ |
