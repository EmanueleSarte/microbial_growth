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

Clearly, both $\vec{x}$ and $F(\vec{x})$ are specified by the considered model, and thus can vary a lot. For example, a dynamical variable that is common to all the models is the *microbe size* $m$, thus by solving eq. (1) one gets how the size of the microbe grows over time. However, another importat aspect that must be embedded in a model is the lifetime of a microbe. As a matter of fact, it is evident that $m$ can not become arbitrairly large, since the microbe would divide at a certain moment, generating
