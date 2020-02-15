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

using namespace std; 

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

    // Set the number of particles
    num_particles = 20;

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
        init_particle.theta  = init_dist_theta(gen);
        init_particle.weight = 1.0; 

        particles.push_back(init_particle);
        weights.push_back(1.0);
    }
    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {

    // Predict all particles
    for (unsigned int i = 0; i < particles.size(); i++) {

        // Hard check for division by zero 
        double c1 = velocity;
        if (abs(yaw_rate) > 0.001) {
            c1 /= yaw_rate;
        }

        // Predict motion as bicycle model
        double pred_x = c1 * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
        double pred_y = c1 * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
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

    for (unsigned int obs_idx = 0; obs_idx < observations.size(); obs_idx++) {

        double obs_x = observations[obs_idx].x;
        double obs_y = observations[obs_idx].y;
        int id_of_minimum_predicted_landmark = -1;

        // Calculate particles association
        double min_distance = INFINITY;
        for (unsigned int pred_idx = 0; pred_idx < predicted.size(); pred_idx++) {
            double pred_x = predicted[pred_idx].x;
            double pred_y = predicted[pred_idx].y;
            double distance = dist(obs_x, obs_y, pred_x, pred_y);

            if (distance < min_distance) {
                min_distance = distance;
                id_of_minimum_predicted_landmark = pred_idx;
            }
        }
        observations[obs_idx].id = id_of_minimum_predicted_landmark;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {

    double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
    double sig_x = 2 * pow(std_landmark[0], 2);
    double sig_y = 2* pow(std_landmark[1], 2);

    for (int particle_idx = 0; particle_idx < num_particles; particle_idx++) {
        Particle current_particle = particles[particle_idx];

        vector<LandmarkObs> landmark_vector;
        for (unsigned int landmark_idx = 0; landmark_idx < map_landmarks.landmark_list.size(); landmark_idx++) {
            LandmarkObs currnet_landmark;
            currnet_landmark.x  = map_landmarks.landmark_list[landmark_idx].x_f;
            currnet_landmark.y  = map_landmarks.landmark_list[landmark_idx].y_f;
            currnet_landmark.id = map_landmarks.landmark_list[landmark_idx].id_i;

            if (dist(current_particle.x, current_particle.y, currnet_landmark.x, currnet_landmark.y) <= sensor_range) {
                landmark_vector.push_back(currnet_landmark);
            }
        }

        vector<LandmarkObs> transformed_observations;
        for (unsigned int observ_idx = 0; observ_idx < observations.size(); observ_idx++) {
            LandmarkObs current_observ = observations[observ_idx];

            LandmarkObs transformed_observation;
            transformed_observation.id = current_observ.id;
            transformed_observation.x  = current_observ.x*cos(current_particle.theta) - current_observ.y*sin(current_particle.theta) + current_particle.x;
            transformed_observation.y  = current_observ.x*sin(current_particle.theta) + current_observ.y*cos(current_particle.theta) + current_particle.y;
            transformed_observations.push_back(transformed_observation);

            dataAssociation(landmark_vector, transformed_observations);

            double particle_probability = 1.0;
            for (unsigned int observ_idx = 0; observ_idx < transformed_observations.size(); observ_idx++) {

                LandmarkObs transformed_observation = transformed_observations[observ_idx];
                LandmarkObs associated_landmark = landmark_vector[transformed_observation.id];
                particle_probability *= gauss_norm * exp(-(pow(transformed_observation.x - associated_landmark.x, 2)/sig_x + pow(transformed_observation.y - associated_landmark.y, 2) / sig_y ));
            }
            particles[particle_idx].weight = particle_probability;
            weights[particle_idx] = particle_probability;
        }
    }
}

void ParticleFilter::resample() {

    vector<Particle> resampled_particles;
    int idx = rand() % num_particles;

    double max_weight = 0;
    for (int i = 0; i < num_particles; i++) {
        if (weights[i] > max_weight) {
            max_weight = weights[i];
        }
    } 

    double beta = 0.0;
    for (int i = 0; i < num_particles; i++) {
        bool is_picked = false;
        double rand_number = (double(rand()) / double(RAND_MAX));

        beta += rand_number * max_weight * 2.0;
        while (is_picked == false) {
            if (weights[idx] < beta) {
                beta -= weights[idx];
                idx = (idx+1) % num_particles;
            } else {
                resampled_particles.push_back(particles[idx % num_particles]);
                is_picked = true;
            }
        }
    }
    particles = resampled_particles;  
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