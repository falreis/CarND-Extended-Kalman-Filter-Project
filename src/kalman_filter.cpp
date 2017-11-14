#include "kalman_filter.h"
#include <math.h>


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

void KalmanFilter::GenericUpdate(const VectorXd &z, const VectorXd &z_pred){
  MatrixXd PHt = P_ * H_.transpose();
  VectorXd y = z - z_pred;
  
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
  this->GenericUpdate(z, z_pred);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  //TODO: update the state by using Extended Kalman Filter equations
  float h1, h2, h3;
  h1 = sqrt(pow(x_(0),2) + pow(x_(1),2));
  h2 = atan2(x_(1) , x_(0));

  if(h1 != 0)
    h3 = ((x_(0)*x_(2)) + (x_(1)*x_(3))) / h1;
  else
    h3 = 0;

  //update H jacobian
  VectorXd Hj(3);
  Hj << h1, h2, h3;

  //update values
  this->GenericUpdate(z, Hj);
}
