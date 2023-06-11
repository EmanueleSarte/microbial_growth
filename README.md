# Microbial Growht :microbe:
## Final project of Laboratory of Computational Physics - Mod. B

**Group members:** Giovanni Zago, Emanuele Sarte, Alessio Saccomani, Fateme <br>
**Supervisor:** Prof. Carlo Albert <br>
**Academic year:** 2022/2023

This project has been carried out by a group of Physics of Data student as final project of the course Laboratory of Computational Physics - Mod. B, held by Prof. Marco Baiesi. The subject of the project and the supervision of the work has been provided by Prof. Carlo Albert. 

## Introduction

The aim of this project is analyzing data coming from microbial coltures by performing Bayesian inference of parameters belonging to stochastic models that describe the growth and the lifetime of the microbes themselves. The datasets used in this work are *Tanouchi25c*, *Tanouchi37c* and *Susman18*. 

### Formalism
The underlying idea that is shared by all the models considered for the analysis of the data is that a microbe can be described as a *dynamical system* characterized by a few dynamical variables $\vec{x}$ that obey a *"motion" equation*:

$$\dot{\vec{x}} = F(\vec{x}) \tag{1}$$

Clearly, both $\vec{x}$ and $F(\vec{x})$ are specified by the considered model, and thus can vary a lot. For example, a dynamical variable that is common to all the models is the *microbe size* $m$, thus by solving eq. (1) one gets how the size of a microbe grows over time. Another importat aspect that must be embedded in a model is the lifetime of a microbe. As a matter of fact, it is evident that $m$ can not become arbitrairly large, since the microbe would divide at a certain moment, generating a daughter microbe. However, it is reasonable to think that the time at which the microbe divides is not deterministic, and so its lifespan should be drawn from a *probability distribution*, that, again, can vary according to the considered model. The best way to formalize this is to define a function $S(t)$ called *survival probability* that represents the probability of the microbe cell to survive (i.e. not divide) for $\hat{t} \leq t$. In general the survival probability obeys

$$\frac{\dot{S}(t)}{S(t)} = - h(\vec{x}(t)) \tag{2}$$

where $h(\vec{x})$ is a generic function of the dynamic variables. Since $S(t)$ is a cumulative probability then $\dot{S}(t)$ is a probability distribution, and so $t \sim \dot{S}(t)$, with $t \geq 0$, represents, in theory, a possibile lifespan of the cell. This is a key concept because, for example, by collecting data samples of the microbes lifespans we can infer the parameters that describe $\dot{S}(t)$, which are also the parameters that describe $\vec{x}(t)$, allowing us to validate our models. <br>
Next we are going to showcase the models considered for the Bayesian inference. 

### Model 1
### Model 1.2
### Model 2
Model 2 aims at correcting the istantaneous division problem that arisend in Model 1.2. To do so we need to introduce another dynamic variable, $p$, which represents the amount of a fictious protein that needs to be accumulated over a certain threshold quantity in order to allow the division process. Thus in this case we have that

$$\vec{x} = \begin{pmatrix} m \\ p \end{pmatrix}$$
### Model 3
