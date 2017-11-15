#include "kalman_filter.h"
#include <math.h>
#include <iostream>


using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  ///predict the state
  x_ = F_ * x_;
	P_ = (F_ * P_ * F_.transpose()) + Q_;
}

void KalmanFilter::GenericUpdate(const VectorXd &z, const VectorXd &y){
  MatrixXd PHt = P_ * H_.transpose();  
  MatrixXd S = (H_ * PHt) + R_;
	MatrixXd K = PHt * S.inverse();

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

void KalmanFilter::Update(const VectorXd &z) {
  ///update the state by using Kalman Filter equations
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  this->GenericUpdate(z, y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  //TODO: update the state by using Extended Kalman Filter equations
  float h1, h2, h3;

  //calculate h1, h2 and h3
  h1 = sqrt(pow(x_(0),2) + pow(x_(1),2));
  h2 = atan2(x_(1) , x_(0));
  h3 = (h1 >= 0.01) ? ( ((x_(0)*x_(2)) + (x_(1)*x_(3))) / h1 ) : 0;

  //update H jacobian and y
  VectorXd z_pred(3);
  z_pred << h1, h2, h3;
  VectorXd y = z - z_pred;
  
  y(1) = atan2(sin(y(1)), cos(y(1)));
  while(abs(y(1)) > M_PI)
    y(1) = (y(1) > 0)? (y(1) - (2*M_PI)) : (y(1) + (2*M_PI));

  //update values
  this->GenericUpdate(z, y);
}
