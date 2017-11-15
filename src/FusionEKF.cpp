#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;
  
  /**
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */
  H_laser_ << 1, 0, 
              0, 0,
              0, 1, 
              0, 0;
  
  Hj_<< 1, 1, 0, 0,
        1, 1, 0, 0,
        1, 1, 1, 1;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  float px, py;
  float noise_ax, noise_ay;

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */

    //set the acceleration noise components
    noise_ax = 9;
    noise_ay = 9;
    float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;

    // first measurement
    cout << "EKF: " << endl;
    Eigen::VectorXd x_ini = VectorXd(4);  // state vector
    x_ini << 1, 1, 1, 1;

    Eigen::MatrixXd P_ini = MatrixXd(4, 4);  //state covariance matrix P
    P_ini<< 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1000, 0,
            0, 0, 0, 1000;
    
    Eigen::MatrixXd R_ini = MatrixXd(2, 2); //measurement covariance
    R_ini<< 0.0225, 0,
            0, 0.0225;

    Eigen::MatrixXd H_ini = MatrixXd(2, 4);  //measurement matrix
    H_ini<< 1, 0, 0, 0,
            0, 1, 0, 0;

    Eigen::MatrixXd F_ini = MatrixXd(4, 4); //the initial transition matrix F_
    F_ini<< 1, 0, dt, 0,
            0, 1, 0, dt,
            0, 0, 1, 0,
            0, 0, 0, 1;

    Eigen::MatrixXd Q_ini = MatrixXd(4, 4); // process covariance matrix
    
	  Q_ini<< pow(dt,4)/4*noise_ax, 0, pow(dt,3)/2*noise_ax, 0,
            0, pow(dt,4)/4*noise_ay, 0, pow(dt,3)/2*noise_ay,
            pow(dt,3)/2*noise_ax, 0, pow(dt,2)*noise_ax, 0,
            0, pow(dt,3)/2*noise_ay, 0, pow(dt,2)*noise_ay;

    //update mesurements
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) { /**Convert polar to cartesian coordinates and initialize state.*/
      cout << "INIT RADAR" << endl;
      float rho = measurement_pack.raw_measurements_(0);
      float phi = measurement_pack.raw_measurements_(1);
      px = rho * cos(phi);
      py = rho * sin(phi);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {  /** Initialize state. */
      cout << "INIT LIDAR" << endl;
      px = measurement_pack.raw_measurements_(0);
      py = measurement_pack.raw_measurements_(1);
      
      //x_ini << px, py, 0, 0;
      //H_ini = H_laser_;
      //R_ini = R_laser_;
    }
    
    //init EKF with previous values
    ekf_.Init(x_ini, P_ini, F_ini, H_ini, R_ini, Q_ini); //Eigen::MatrixXd &R_in, Eigen::MatrixXd &Q_in);

    ekf_.x_ << px, py, 0, 0;
    this->previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  /**
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  noise_ax = 9;
  noise_ay = 9;

  //compute the time elapsed between the current and previous measurements
	float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
	this->previous_timestamp_ = measurement_pack.timestamp_;

	float dt_2 = dt * dt;
	float dt_3 = dt_2 * dt;
	float dt_4 = dt_3 * dt;

	//Modify the F matrix so that the time is integrated
	ekf_.F_(0, 2) = dt;
	ekf_.F_(1, 3) = dt;

	//set the process covariance matrix Q
	ekf_.Q_ = MatrixXd(4, 4);
	ekf_.Q_ << dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
			   0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
			   dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
			   0, dt_3/2*noise_ay, 0, dt_2*noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  /**
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    cout << "RADAR" << endl;
    Tools tools;
    Hj_ = tools.CalculateJacobian(ekf_.x_);

    ekf_.R_ = R_radar_;
    ekf_.H_ = Hj_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } 
  else {
    // Laser update
    cout << "LIDAR" << endl;
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  this->previous_timestamp_ = measurement_pack.timestamp_;

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
