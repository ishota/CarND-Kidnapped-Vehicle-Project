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

    for (unsigned int obs_idx = 0; obs_idx < observations.size(); obs_idx++) {

        double obs_x = observations[obs_idx].x;
        double obs_y = observations[obs_idx].y;

        // Calculate particles association
        double min_distance = INFINITY;
        for (unsigned int pred_idx = 0; pred_idx < predicted.size(); pred_idx++) {
            double pred_x = predicted[pred_idx].x;
            double pred_y = predicted[pred_idx].y;
            double distance = dist(obs_x, obs_y, pred_x, pred_y);

            if (distance < min_distance) {
                min_distance = distance;
                observations[obs_idx].id = pred_idx;
            }
        }
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {

    double weight_normalizer = 0.0;

    for (int i = 0; i < num_particles; i++) {
        double particle_x = particles[i].x;
        double particle_y = particles[i].y;
        double particle_theta = particles[i].theta;

        // Transform observations from VEHICLE'S coordinate system to map one.
        vector<LandmarkObs> transformed_observations;
        for (int j = 0; j < observations.size(); j++) {
            LandmarkObs lnd_obs;
            lnd_obs.id = j;
            lnd_obs.x = particle_x + (cos(particle_theta) * observations[j].x) - (sin(particle_theta) * observations[j].y);
            lnd_obs.y = particle_y + (sin(particle_theta) * observations[j].x) - (cos(particle_theta) * observations[j].y);
            transformed_observations.push_back(lnd_obs);
        }

        // Filter map landmarks
        vector<LandmarkObs> predicted_landmarks;
        for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            Map::single_landmark_s landmark = map_landmarks.landmark_list[j];
            if ((abs(particle_x - landmark.x_f) <= sensor_range) && (abs((particle_y - landmark.y_f)) <= sensor_range)) {
                predicted_landmarks.push_back(LandmarkObs {landmark.id_i, landmark.x_f, landmark.y_f});
            }
        }

        // Associate observations to predicted landmarks
        dataAssociation(predicted_landmarks, transformed_observations);

        // Calculate weight of each particle
        particles[i].weight = 1.0;

        double sigma_x  = std_landmark[0];
        double sigma_y  = std_landmark[1];
        double sigma_x2 = pow(sigma_x, 2);
        double sigma_y2 = pow(sigma_y, 2);
        double normalizer = (1.0/(2.0 * M_PI * sigma_x * sigma_y));

        for (int j = 0; j < transformed_observations.size(); j++) {
            double trans_obs_x  = transformed_observations[j].x;
            double trans_obs_y  = transformed_observations[j].y;
            double trans_obs_id = transformed_observations[j].id;
            double multi_prob = 1.0;

            for (int k = 0; k < predicted_landmarks.size(); k++) {
                double pred_landmark_x  = predicted_landmarks[k].x;
                double pred_landmark_y  = predicted_landmarks[k].y;
                double pred_landmark_id = predicted_landmarks[k].id;

                if (trans_obs_id == pred_landmark_id) {
                    multi_prob = normalizer * exp(-1.0 * ((pow((trans_obs_x - pred_landmark_x), 2)/(2.0 * sigma_x2)) + (pow((trans_obs_y - pred_landmark_y), 2)/(2.0 * sigma_y2))));
                    particles[i].weight *= multi_prob;
                }
            }
        }
        weight_normalizer += particles[i].weight;
    }

    // Normalize the weight of all particles
    for (int i = 0; i < particles.size(); i++) {
        particles[i].weight /= weight_normalizer;
        weights[i] = particles[i].weight;
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