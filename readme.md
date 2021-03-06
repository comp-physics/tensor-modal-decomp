# Tensor modal decomposition

## To-do

* Look at the distribution of the $z$'s

## Main

* What it is: Modal decomposition
* How it is done
  * Statistics
  * greater-than second-order moments
  * ICA, etc.
  * latent variables
  * greater-than second-order tensors
* What is does
  * recover structures in flows
  * model reduction
    * beyond energy
    * "high-order coupling"
  * rare extreme events

## Problems worth solving

* Spurious modes
  * Arise from
    * artificial boundary conditions used to modulate non-reflecting outflow or inflow
    * sponge regions
  * Turn up in modal decompositions, are discarded *ad hoc*
  * Idea: condition correlation matrix on data?
* Predicting extreme events
* Efficient model reduction
* Identifying coherent structures

## Flow data

* Some canonical, relevant flows
  * Flow past sphere
  * Flow over hydrofoil
  * Turbulent channel flow
  * Lid-driven cavity
  * Kelvin-Helmholtz instability
  * Boundary layer transition
  * Turbulent jet

* Where to get the data
  * Simulate it
  * Collaborators (Oliver, Ethan, etc.)
  * [JHU Turbulence Database](http://turbulence.pha.jhu.edu/) 
  * [More from JHU](https://pages.jh.edu/cmeneve1/datasets.html)
  * etc.

## Software

* [FastICA](https://en.wikipedia.org/wiki/FastICA)
* [Scikit-FastICA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html)
* [Julia](https://juliahub.com/ui/Packages/IndependentComponentAnalysis/NI0LK/0.1.4)
* [More](https://github.com/search?q=independent+component+analysis)




## Literature

[Located here](https://www.zotero.org/groups/4507615/comp-physics/collections/HX6358UM) 
