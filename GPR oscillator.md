Physics-Informed Surrogate Modeling of a Damped Harmonic Oscillator

A high-performance machine learning pipeline that replaces slow numerical computer simulations with a fast data-driven emulator using Gaussian Process Regression. This project maps physical parameters directly to complete motion paths over time while providing built-in reliability and uncertainty estimates.

**The Physics Background**

The system being modeled is a damped harmonic oscillator
the equation of motion for the following is:
m {d^2x/dt^2} + c {dx/dt} + kx = 0

Three main physical properties drive this system:
*Mass(m)
*Damping Coefficient(c): Determines the dissipative losses due to air resistance or friction.
*Spring Stiffness(k):Determines frequency of oscillation based on how strongly the spring pulls the object back to equilibrium

Instead of running heavy step-by-step physics calculations every single time we want to know how the system moves, we train a machine learning model to learn these physical rules from past examples.


**How the Machine Learning Works**

We treat this as a multi-output prediction problem:

1.Random Experiment Design: We simulate the physics model many times using random combinations of mass, damping, and stiffness within realistic boundaries.
2.Inputs: The three physical numbers (mass, damping, spring stiffness).
3.Targets: The entire motion path (displacement over 1,000 distinct time points) stored in a Pandas DataFrame.
4.The Surrogate Engine: A machine learning model called a Gaussian Process Regressor looks at these examples and learns the hidden mathematical relationship between the input settings and the resulting motion curves.


**Key Features**

1.Instant Predictions: Instead of waiting for a differential equation solver to calculate every time step from scratch, the trained model outputs the entire motion path in a few seconds.
2.Confidence Estimates (Uncertainty Quantification): Unlike standard neural networks that just guess an answer blindly, this model calculates its own confidence level. It shades a 95% confidence band around its prediction, warning you when it is operating on unfamiliar parameters.This 95% is determined by taking upper and lower bound as 1.96*(standard devation from mean) which comes out to be 95% of the area under the gaussian curve.
3.Clean Data Management: Uses Pandas DataFrames to structure simulation runs clearly before flattening them into numerical matrices for training.
4.Fully Reproducible: Built with fixed random seeding so anyone cloning your repository gets the exact same training data and results.
