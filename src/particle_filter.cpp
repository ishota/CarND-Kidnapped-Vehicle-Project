/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::default_random_engine;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

    // Set the number of particles
    num_particles = 10;

    // Create normal distribution
    std::normal_distribution<double> init_dist_x(x, std[0]);
    std::normal_distribution<double> init_dist_y(y, std[1]);
    std::normal_distribution<double> init_dist_theta(theta, std[2]);

    // Initialize particles
    for (int i = 0; i < num_particles; i++) {
        Particle init_particle;
        init_particle.id = i;
        init_particle.x  = init_dist_x(gen);
        init_particle.y  = init_dist_y(gen);
        init_particle.theta   = init_dist_theta(gen);
        init_particle.weight  = 1.0;
        
        particles.push_back(init_particle);
        weights.push_back(init_particle.weight);
    }

    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
    /**
     * TODO: Add measurements to each particle and add random Gaussian noise.
     * NOTE: When adding noise you may find std::normal_distribution 
     *   and std::default_random_engine useful.
     *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
     *  http://www.cplusplus.com/reference/random/default_random_engine/
     */

    // Predict all particles
    for (unsigned int i = 0; i < particles.size(); i++) {

        // Hard check for division by zero 
        double c1 = velocity;
        if (abs(yaw_rate) > 0.0001) {
            c1 /= yaw_rate;
        }

        // Predict motion as bicycle model
        double pred_x = c1 * (sin(particles[i].theta) + yaw_rate * delta_t - sin(particles[i].theta));
        double pred_y = c1 * (cos(particles[i].theta) - cos(particles[i].theta) + (yaw_rate * delta_t));
        double pred_head = yaw_rate * delta_t;

        // Create normal distribution
        std::normal_distribution<double> pred_dist_x(pred_x, std_pos[0]);
        std::normal_distribution<double> pred_dist_y(pred_y, std_pos[1]);
        std::normal_distribution<double> pred_dist_theta(pred_head, std_pos[2]);

        // Update particles
        particles[i].x += pred_dist_x(gen);
        particles[i].y += pred_dist_y(gen);
        particles[i].theta += pred_dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
    /**
     * TODO: Find the predicted measurement that is closest to each 
     *   observed measurement and assign the observed measurement to this 
     *   particular landmark.
     * NOTE: this method will NOT be called by the grading code. But you will 
     *   probably find it useful to implement this method and use it as a helper 
     *   during the updateWeights phase.
     */

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
    /**
     * TODO: Update the weights of each particle using a mult-variate Gaussian 
     *   distribution. You can read more about this distribution here: 
     *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
     * NOTE: The observations are given in the VEHICLE'S coordinate system. 
     *   Your particles are located according to the MAP'S coordinate system. 
     *   You will need to transform between the two systems. Keep in mind that
     *   this transformation requires both rotation AND translation (but no scaling).
     *   The following is a good resource for the theory:
     *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
     *   and the following is a good resource for the actual equation to implement
     *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
     */

}

void ParticleFilter::resample() {
    /**
     * TODO: Resample particles with replacement with probability proportional 
     *   to their weight. 
     * NOTE: You may find std::discrete_distribution helpful here.
     *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
     */

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
    // particle: the particle to which assign each listed association, 
    //   and association's (x,y) world coordinates mapping
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
    vector<double> v;

    if (coord == "X") {
        v = best.sense_x;
    } else {
        v = best.sense_y;
    }

    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}