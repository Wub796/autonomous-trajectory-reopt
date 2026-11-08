# Annotated Bibliography & Synthesis Matrix (Entries #1–#32)

**Topic:** Optimization of an onboard autonomous flight director using anomaly detection and reinforcement learning models  
**Gap:** Anomaly detection studies (#1, #26, #31) spot bad sensor data without steering the spacecraft, while PPO flight studies (#3, #4, #21) steer Earth–Mars paths without a detector for slow engine wear. No study has combined an Isolation Forest detector with a PPO flight controller to catch slow thruster decay and adjust an Earth–Mars flight path on small spacecraft computers.  
**Research Question:** *How can an autonomous flight director dynamically detect progressive thruster degradation and recalculate a continuous Earth–Mars trajectory within the limits of flight-grade computers?*

---

## Part 1 · Quick Summary of All 32 Sources

| # | Author (Year) | Question Asked | Key Finding |
| :-: | :--- | :--- | :--- |
| **#1** | Hundman et al. (2018) | How can a spacecraft detect sensor glitches on its own without examples of past breakdowns? | AI can automatically spot unusual sensor readings without needing labeled examples of past failures, though heavy LSTMs tax flight chips. |
| **#2** | Sullivan & Bosanac (2020) | Can trial-and-error AI steer a low-power spacecraft into a steady orbit? | An AI autopilot can learn to guide low-thrust spacecraft into stable orbits while balancing targeting and fuel. |
| **#3** | Zavoli & Federici (2021) | Can an AI autopilot keep a spacecraft on course to Mars even if its engines misfire? | An AI flight controller can keep a spacecraft on target despite engine firing errors across thousands of Monte Carlo runs. |
| **#4** | Miller & Bosanac (2022) | How do you train an AI to reach its destination quickly without wasting fuel? | Penalizing engine use in a multi-reward setup teaches an AI to make steady flight progress while conserving fuel. |
| **#5** | Capra et al. (2022) | What type of AI design gives a spacecraft the smoothest steering control? | Continuous 3D steering controls and memory work much better than simple on-and-off thruster commands. |
| **#6** | Capra et al. (2025) | Can smart AI autopilots follow strict safety rules while running on small spacecraft computers? | Pairing AI with safety rules prevents collisions and runs in under 0.12 seconds on a Raspberry Pi 4. |
| **#7** | Izzo et al. (2019) | How can spacecraft steer themselves instead of waiting for commands sent from Earth? | Onboard neural networks let spacecraft make instant flying decisions to beat deep-space radio delays. |
| **#8** | Astropy Collaboration (2022) | How can flight simulations track real planetary paths instead of assuming perfect circles? | Standardized Astropy tools track exact planetary positions to keep flight simulations realistic. |
| **#9** | Izzo et al. (2020) | How can engineers mathematically prove that an AI flight controller will remain stable? | Calculus-based math tools can prove a neural network will keep a vehicle stable after unexpected bumps. |
| **#10** | Gaudet et al. (2020a) | How can an AI learn to steer smoothly while balancing flying hints against sticking a landing? | Splitting training rewards into steering hints and arrival goals helps an AI land softly on Mars without wasting fuel. |
| **#11** | Gaudet et al. (2020b) | Can an onboard AI quickly recover from a broken engine without waiting for help from Earth? | Giving an AI adaptive feedback helps it adjust in real time to surprise engine problems and broken sensors. |
| **#12** | Chaudhary et al. (2026) | How can a spacecraft correct flight mistakes without relying on overly simple textbook math about noise? | Testing flight corrections across thousands of simulations keeps spacecraft safely on course under messy space conditions. |
| **#13** | Li et al. (2024) | How can reinforcement learning maintain targeting accuracy under low-thrust propulsion limits? | Embedding stability rules into policy training maintains high targeting accuracy despite unexpected flight disturbances. |
| **#14** | Federici & Zavoli (2021) | Can an AI learn low-thrust orbital transfers without relying on expert demonstration flights? | Trial-and-error AI can independently discover fuel-efficient low-thrust trajectories using distance rewards and fuel penalties. |
| **#15** | Ballentine et al. (2026) | How can spaceflight AI be trained when real flight failure telemetry is unavailable? | Physics-informed generative models produce realistic synthetic flight data that bridges the data scarcity gap. |
| **#16** | Towers et al. (2024) | How can simulated space environments be standardized for reinforcement learning agents? | Gymnasium provides a standard simulation API that manages continuous states and actions for RL algorithms. |
| **#17** | Conversano et al. (2019) | How do electric Hall thrusters degrade during continuous vacuum operation? | Documented empirical thrust loss and wall-erosion signatures in Hall thrusters, providing baselines for hardware wear. |
| **#18** | Cheng et al. (2019) | Can neural networks compute continuous low-thrust orbital maneuvers fast enough to run onboard? | Trained neural networks generate real-time low-thrust commands instantly, eliminating the wait for Earth communications. |
| **#19** | Schulman et al. (2017) | How can an AI flight controller update its steering actions without destabilizing its flight policy? | Clipping policy updates (PPO safety cap) prevents destructive changes and keeps continuous control training stable. |
| **#20** | Fu et al. (2022) | How can spacecraft detect unusual telemetry patterns without pre-labeled failure datasets? | Combining time-series feature extraction with an Isolation Forest flags telemetry anomalies without pre-labeled failure examples. |
| **#21** | Miller et al. (2019) | Can reinforcement learning guide a low-thrust spacecraft from Earth to Mars using realistic planetary positions? | PPO can learn mass-efficient Earth-to-Mars transfers matching NASA trajectory tools. |
| **#22** | Elkins et al. (2020) | Can AI control a spacecraft's orientation accurately without needing a powerful onboard computer? | Deep reinforcement learning performs accurate spacecraft attitude maneuvers while using limited computing power. |
| **#23** | Getzandanner & Martin (2025) | Can an AI navigate a spacecraft while also learning about its surrounding space environment? | Reinforcement learning combines spacecraft navigation with environmental characterization while making autonomous decisions. |
| **#24** | Casas et al. (2022) | Can an AI guide a low-thrust spacecraft even when it does not know the spacecraft's exact dynamics? | A dynamics-agnostic AI learns low-thrust maneuvers and adapts when environmental conditions and thruster performance change. |
| **#25** | Raffin et al. (2021) | How can engineers make sure the reinforcement-learning software used for spacecraft control is reliable? | Standardized, tested RL software (Stable-Baselines3) reduces implementation bugs and provides reliable training. |
| **#26** | Liu et al. (2008) | How can spacecraft detect unusual sensor data quickly without using much onboard memory? | Isolation Forest detects anomalies efficiently by isolating unusual data points with simple decision trees using tiny memory. |
| **#27** | Schulman et al. (2016) | How can reinforcement learning stay stable when learning complicated continuous-control tasks over long periods? | Generalized Advantage Estimation (GAE) reduces noisy trial-and-error updates and links early actions to final arrival. |
| **#28** | Creswell (2012) | How do researchers set up a fair experiment to prove a new method works? | Keeping starting conditions fixed between a baseline and test group proves whether the controller caused the improvement. |
| **#29** | Leedy & Ormrod (2015) | How can an experiment test two changing variables at once? | Factorial designs test every combination of controller setup and engine wear level to see how they interact. |
| **#30** | Mnih et al. (2016) | How can an AI learn steady control by pairing steering choices with progress scores? | Actor-Critic networks running parallel tests learn continuous control tasks steadily on standard CPUs. |
| **#31** | Chandola et al. (2009) | Why do simple warning alarms miss slow, gradual hardware wear? | Gradual equipment drift is a collective anomaly that only shows up across a sequence of readings over time. |
| **#32** | Snyder et al. (2018) | How does the SPT-140 Hall thruster perform across different deep-space power settings? | Measured real thrust and efficiency baselines for the SPT-140 across 0.9 kW to 4.5 kW throttle levels in vacuum tests. |

---

## Part 2 · Synthesis Matrix

**My themes:**
- **Theme A:** Catching sensor & engine problems
- **Theme B:** Autonomous flying & signal delays
- **Theme C:** Saving fuel vs. reaching the target
- **Theme D:** Real hardware & flight safety

| Source (author, year, entry #) | Theme A: Catching sensor & engine problems | Theme B: Autonomous flying & signal delays | Theme C: Saving fuel vs. reaching the target | Theme D: Real hardware & flight safety |
| :--- | :--- | :--- | :--- | :--- |
| **#32 Snyder et al. (2018)** | | | | SPT-140 thruster baselines from 0.9–4.5 kW (#32, p. 4) |
| **#31 Chandola et al. (2009)** | Slow engine wear is a collective anomaly (#31, p. 8) | | | |
| **#30 Mnih et al. (2016)** | | Actor-critic AI learns smooth control (#30, p. 1928) | | Parallel runs keep CPU training steady (#30, p. 1931) |
| **#29 Leedy & Ormrod (2015)** | | | | Factorial design tests two variables at once (#29, p. 192) |
| **#28 Creswell (2012)** | | | | Fixed controls prove controller works (#28, p. 295) |
| **#27 Schulman et al. (2016)** | | Links early burns to final arrival (#27, p. 2) | | Cuts trial-and-error noise on long flights (#27, p. 2) |
| **#26 Liu et al. (2008)** | Random trees isolate bad data in few splits (#26, p. 413) | | | Uses very little memory and runs fast (#26, p. 416) |
| **#25 Raffin et al. (2021)** | | | | Tested PPO code prevents software bugs (#25, p. 1) |
| **#24 Casas et al. (2022)** | | AI adapts to changing engine power (#24, abstract) | | |
| **#23 Getzandanner & Martin (2025)** | | AI navigates and senses space on its own (#23, p. 1) | | |
| **#22 Elkins et al. (2020)** | | AI controls spacecraft pointing angles (#22, p. 1) | | Works without a powerful computer (#22, p. 1) |
| **#21 Miller et al. (2019)** | | PPO steers low-thrust Earth–Mars flights (#21, p. 1) | Matches NASA tool fuel efficiency (#21, p. 9) | |
| **#20 Fu et al. (2022)** | Isolation Forest flags abnormal sensor data (#20, p. 1) | | | |
| **#19 Schulman et al. (2017)** | | Safety cap keeps AI learning smooth (#19, p. 3) | | Stops autopilot from making wild mistakes (#19, p. 3) |
| **#18 Cheng et al. (2019)** | | Neural networks plan paths instantly (#18, p. 72962) | | |
| **#17 Conversano et al. (2019)** | Wall wear causes slow thrust drop (#17, p. 11) | | | Vacuum tests track Hall thruster wear (#17, p. 5) |
| **#16 Towers et al. (2024)** | | | | Standard Gymnasium toolkit for AI tests (#16, p. 2) |
| **#15 Ballentine et al. (2026)** | Physics AI makes realistic fake failure data (#15, abstract) | | | Fixes lack of real space failure records (#15, abstract) |
| **#14 Federici & Zavoli (2021)** | | Trial-and-error AI steers without human examples (#14, p. 1) | Distance rewards and fuel penalties guide flight (#14, p. 6) | |
| **#13 Li et al. (2024)** | | Guides low-thrust approach under bumps (#13, p. 2210) | | Stability rules keep AI within engine limits (#13, p. 2214) |
| **#12 Chaudhary et al. (2026)** | | AI adds real-time fixes to planned route (#12, abstract) | Limits worst-case fuel costs via penalties (#12, p. 5) | 100,000 test flights prove safety under noise (#12, p. 9) |
| **#11 Gaudet et al. (2020b)** | Adapts to radar and sensor errors (#11, p. 180) | Memory AI adapts when thrusters fail (#11, p. 180) | Burns extra fuel during emergency retargeting (#11, p. 186) | Prevents crashes that ruin standard systems (#11, p. 188) |
| **#10 Gaudet et al. (2020a)** | | Autonomously lands on Mars in heavy wind (#10, p. 1723) | Splits rewards into hints vs. touchdown (#10, p. 1728) | Needs very little power on flight chips (#10, p. 1738) |
| **#9 Izzo et al. (2020)** | | Neural networks act as real-time controllers (#9, p. 145) | | Math tools prove AI stays stable after nudges (#9, p. 148) |
| **#8 Astropy Collaboration (2022)** | | | | Real planet math prevents simulation errors (#8, p. 4) |
| **#7 Izzo et al. (2019)** | Reviews AI that learns orbit mistakes (#7, p. 291) | Onboard AI steers spacecraft in real time (#7, p. 287) | Surveys ways to balance flight time and fuel (#7, p. 293) | Warns AI must be proven safe before flight (#7, p. 296) |
| **#6 Capra et al. (2025)** | | Pairs smart AI planning with safety rules (#6, p. 1) | Math rules balance staying on course with fuel (#6, p. 8) | Runs in under 0.12 s on a Raspberry Pi (#6, p. 17) |
| **#5 Capra et al. (2022)** | | AI plans flight paths around space objects (#5, p. 3787) | Smooth steering beats simple on/off bursts (#5, p. 3793) | Networks with memory train more steadily (#5, p. 3798) |
| **#4 Miller & Bosanac (2022)** | | Steers spacecraft through tricky gravity zones (#4, p. 719) | Balances flying forward against burning fuel (#4, p. 723) | Learning fails if fuel penalties are tuned wrong (#4, p. 728) |
| **#3 Zavoli & Federici (2021)** | | AI handles unexpected thruster errors (#3, p. 1440) | Penalizes engine errors to keep paths steady (#3, p. 1443) | Thousands of test flights prove reliability (#3, p. 1448) |
| **#2 Sullivan & Bosanac (2020)** | | AI learns to steer low-thrust satellites (#2, p. 1) | Weighs hitting the target against fuel use (#2, p. 7) | Assumes healthy engines, needs fault testing (#2, p. 14) |
| **#1 Hundman et al. (2018)** | Spots subtle sensor shifts without labeled data (#1, p. 387) | | | Heavy neural networks struggle on flight chips (#1, p. 393) |

### Note on Entry #30 in Your Bibliography Table
In your pasted notes, Entry #30 (*Mnih et al., 2016*) accidentally had the annotation rows from Entry #32 (*Snyder et al., 2018*) pasted beneath it. Here is the corrected annotation block for **Entry #30** so you can paste it directly into your turn-in document:

| Field | Corrected Text for Entry #30 |
| :--- | :--- |
| **Entry # and date added** | Entry #30 · added 11/3 |
| **Citation** | Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., & Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, 48, 1928–1937. https://doi.org/10.48550/arXiv.1602.01783 |
| **What it says** | This study introduces Actor-Critic reinforcement learning methods that run multiple simulation environments in parallel on a standard multi-core CPU instead of requiring specialized GPUs. By pairing an "Actor" network (which chooses steering actions) with a "Critic" baseline (which scores progress), the AI learns continuous control steadily without unstable updates. |
| **How they studied it** | The authors trained parallel Actor-Critic agents across continuous physics simulators and video game benchmarks, comparing training speed and stability on standard CPU cores against GPU-based methods. |
| **Can I trust it?** | It is a landmark peer-reviewed conference paper published at ICML 2016 by senior AI scientists at Google DeepMind. |
| **How it connects to my question** | It explains how my flight controller pairs action selection with a baseline progress score and runs parallel trajectory rollouts on a standard CPU during training. |
| **Questions it raises** | How many parallel trajectory simulations are needed per update step to keep learning smooth when thruster power degrades mid-flight? |

