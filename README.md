# CarND-Kidnapped-Vehicle-Project

[Success]: ./picture/Success.png "success"

## Description

In this project, I implemented a Particle Filter to localize current position in closed environment.

## Starting Guied

1. You can use a build.sh script to compile particle filter algorithms. If you want to clean your directory, clean.sh script is useful.
```bash
./build.sh
```
2. Execute run.sh script to execute localization.
```bash
./run.sh
```
3. Select Project3: Kidnapped Vehicle simulator and press Start button. You can get the simulator here <<https://github.com/udacity/self-driving-car-sim/releases>>. 

## Localization Algorithm

### 1. Initialize particle filter 
Initialize metrices related to the particle filter. The metrices to be initialezed are as follows.

* GPS measurement uncertainty
* Landmark measurement uncertainty
* Read map data
* The number of particles
* Particles position x, position y, yaw, and weight

### 2. Prediction
Predict the next step position based on the two-wheel model.
This is done by using control inputs and time elapsed between time steps.
```cpp
double pred_x = c1 * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
double pred_y = c1 * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
double pred_head = yaw_rate * delta_t;
```

### 3. Update weight
After prediction step, update particles' weight.
First, search landmark position in vehicle's sensor range.
Socond, transform observation cordinate to each particle's cordinate.
Third, calculate new particle weight.

### 4. Resample
Resample particles according to the updated weight.

## Project Result
The following picture shows the result of localization of particle filter.
This project passed the project's criteria as the picture show.
The location provided by particle with highest weight is then compared with the ground truth is calculated.

![alt text][success]
