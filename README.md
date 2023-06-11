# Microbial Growht :microbe: Final project of Laboratory of Computational Physics - Mod. B

**Group members:** Giovanni Zago, Emanuele Sarte, Alessio Saccomani, Fateme <br>
**Supervisor:** Prof. Carlo Albert <br>
**Academic year:** 2022/2023

This project has been carried out by a group of Physics of Data student as final project of the course Laboratory of Computational Physics - Mod. B, held by Prof. Marco Baiesi. The subject of the project and the supervision of the work has been provided by Prof. Carlo Albert. 

## Table of contents
1. [Introduction](#introduction)
    1. [Formalism](#formalism)
    2. [Model 1](#model1)
    3. [Model 1.2](#model1.2)
    4. [Model 2](#model2)
    4. [Model 3](#model3)

## Introduction <a name="introduction"></a>

The aim of this project is analyzing data coming from microbial coltures by performing Bayesian inference of parameters belonging to stochastic models that describe the growth and the lifetime of the microbes themselves. The datasets used in this work are *Tanouchi25c*, *Tanouchi37c* and *Susman18*. 

### Formalism <a name="formalism"></a>
The underlying idea that is shared by all the models considered for the analysis of the data is that a microbe can be described as a *dynamical system* characterized by a few dynamical variables $\vec{x}$ that obey a *"motion" equation*:

$$\dot{\vec{x}} = F(\vec{x}) \tag{1}$$

Clearly, both $\vec{x}$ and $F(\vec{x})$ are specified by the considered model, and thus can vary a lot. For example, a dynamical variable that is common to all the models is the *microbe size* $m$, thus by solving eq. (1) one gets how the size of a microbe grows over time. Another importat aspect that must be embedded in a model is the lifetime of a microbe. As a matter of fact, it is evident that $m$ can not become arbitrairly large, since the microbe would divide at a certain moment, generating a daughter microbe. However, it is reasonable to think that the time at which the microbe divides is not deterministic, and so its lifespan should be drawn from a *probability distribution*, that, again, can vary according to the considered model. The best way to formalize this is to define a function $S(t)$ called *survival probability* that represents the probability of the microbe cell to survive (i.e. not divide) for $\hat{t} \leq t$. In general the survival probability obeys

$$\frac{\dot{S}(t)}{S(t)} = - h(\vec{x}(t)) \tag{2}$$

where $h(\vec{x})$ is a generic function of the dynamic variables. Since $S(t)$ is a cumulative probability then $\dot{S}(t)$ is a probability distribution, and so $t \sim \dot{S}(t)$, with $t \geq 0$, represents, in theory, a possibile lifespan of the cell. This is a key concept because, for example, by collecting data samples of the microbes lifespans we can infer the parameters that describe $\dot{S}(t)$, which are also the parameters that describe $\vec{x}(t)$, allowing us to validate our models. <br>
Next we are going to showcase the models considered for the Bayesian inference. 

### Model 1 <a name="model1"></a>
### Model 1.2 <a name="model1.2"></a>

### Model 2 <a name="model2"></a>
Model 2 aims at correcting the istantaneous division problem that arisen in Model 1.2. To do so we need to introduce another dynamic variable, $p$, which represents the amount of a fictious protein that needs to be accumulated over a certain threshold quantity in order to allow the division process. Thus in this case we have that

$$\vec{x} = \begin{pmatrix} m \\ p \end{pmatrix}$$

and

$$\dot{\vec{x}} = \begin{pmatrix} \omega_1 m \\ \omega_2 c m \end{pmatrix}\tag{3}$$

where $w_1$ and $w_2$ are the growth and the protein accumulation rate respectively and $c$ represents the protein concentration, which we can set equal to 1 without losing generality. The survival probability equation becomes

$$\frac{\dot{S}(t)}{S(t)} = - h(m,p) \tag{4}$$

with

$$h(m,p) = \begin{cases}0 \qquad \qquad \text{for} \quad p \lt u\\ w_2 \frac{p + v}{u + v} \qquad \text{for} \quad p \geq u\end{cases} \tag{5}$$

and the division equation becomes

$$\mathcal{J}(m,p | m',p') = \delta(p)\delta\Big(m - \frac{m'}{2}\Big)$$

meaning that the protein amount drops to zero at each cell division, while the microbe size after the division becomes half the one reached at the end of the previous life cycle. By integrating (3) we get

$$m(t) = m_0 e ^ {w_1 t} \tag{7}$$
$$p(t) = \frac{w_2}{w_1} m_0 (e ^ {w_1 t} - 1) \tag{8}$$

Now it is clear that the condition on the value of $h(m,p)$ reporten in eq. (5) is equivalent to 

$$t \geq \frac{1}{w_1} \log \Big(\frac{w_1 u}{w_2 m_0} + 1\Big) \equiv t^* \tag{9}$$

which is always a positive quantity. Thus by integrating (5) we get

$$S(t) = 1 \qquad t \lt t^* \tag{10a}$$
$$S(t) = \exp{\Bigg\{- \frac{\omega_2^2 m_0}{\omega_1^2 (u + v)} \Big[e ^ {\omega_1 t} + \omega_1 t \Big(\frac{\omega_1 v}{\omega_2 m_0} - 1\Big) - 1\Big]\Bigg\}} \qquad t \geq t^* \tag{10b}$$

from which it is evident that istantaneous division is not allowed. 


### Model 3 <a name="model3"></a>
