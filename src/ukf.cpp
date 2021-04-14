#include "ukf.h"
#include "Eigen/Dense"

#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    n_x_ = 5;

    // initial state vector
    x_ = VectorXd(n_x_);
    x_ << 0.2, 0.2, 0.2, 0.2, 0.02;

    // initial covariance matrix
    P_ = MatrixXd(n_x_, n_x_);
    P_ << 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0;

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 30;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 30;

    /**
     * DO NOT MODIFY measurement noise values below.
     * These are provided by the sensor manufacturer.
     */

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    /**
     * End DO NOT MODIFY section for measurement noise values
     */

    /**
     * TODO: Complete the initialization. See ukf.h for other member properties.
     * Hint: one or more values initialized above might be wildly off...
     */

    time_us_ = 0;

    is_initialized_ = false;
    n_aug_ = n_x_ + 2;
    lambda_ = 3 - n_aug_;

    weights_ = VectorXd(2 * n_aug_ + 1);
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 0; i < 2 * n_aug_; i++)
    {
        double weight = 0.5 / (lambda_ + n_aug_);
        weights_(i + 1) = weight;
    }

    Q_ = MatrixXd(2, 2);
    Q_ << std_a_ * std_a_, 0.0,
            0.0, std_yawdd_ * std_yawdd_;

    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
}

UKF::~UKF()
{}

void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
    /**
     * TODO: Complete this function! Make sure you switch between lidar and radar
     * measurements.
     */

    if (!is_initialized_)
    {
        InitUKF(meas_package);
    }
    else
    {
        // Regular flow from assignments
        double delta_t = double(meas_package.timestamp_ - time_us_) / 1000000.0;
        time_us_ = meas_package.timestamp_;

        std::cout << "Delta t: " << delta_t << std::endl;
        Prediction(delta_t);
        Update(meas_package);
    }
}

void UKF::Prediction(double delta_t)
{
    /**
     * TODO: Complete this function! Estimate the object's location.
     * Modify the state vector, x_. Predict sigma points, the state,
     * and the state covariance matrix.
     */

    // 1) Generate sigma points
    // create augmented mean vector
    VectorXd x_aug = VectorXd(n_aug_);
    x_aug.head(5) = x_;
    x_aug(5) = 0.0;
    x_aug(6) = 0.0;

    // create augmented state covariance
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

    P_aug.fill(0.0);
    P_aug.topLeftCorner(P_.rows(), P_.cols()) = P_;
    P_aug.bottomRightCorner(Q_.rows(), Q_.cols()) = Q_;
    MatrixXd A = P_aug.llt().matrixL();

    // create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
    Xsig_aug.col(0) = x_aug;
    double multiplier = sqrt(lambda_ + (double)n_aug_);
    for(int i = 0; i < n_aug_; i++)
    {
        Xsig_aug.col(i + 1) = x_aug + multiplier * A.col(i);
        Xsig_aug.col(n_aug_ + 1 + i) = x_aug - multiplier * A.col(i);
    }

    // 2) Predict sigma points
    for(int i = 0; i < 2 * n_aug_ + 1; i++)
    {
        double px = Xsig_aug(0, i);
        double py = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yawd = Xsig_aug(4, i);
        double nu_a = Xsig_aug(5, i);
        double nu_yawdd = Xsig_aug(6, i);

        // Predicted values after being passed through CTRV
        double ppx, ppy, pv, pyaw, pyawd;

        // Pass through CTRV model
        if(fabs(yaw) > 0.001)
        {
            ppx = px + v/yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
            ppy = py + v/yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
        }
        else
        {
            // Case for zero yaw -> straight line
            ppx = px + v * delta_t * cos(yaw);
            ppy = py + v * delta_t * sin(yaw);
        }

        pv = v;
        pyaw = yaw + delta_t * yawd;
        pyawd = yawd;

        // Add noise
        ppx += ((delta_t * delta_t) * cos(yaw) * nu_a) / 2;
        ppy += ((delta_t * delta_t) * sin(yaw) * nu_a) / 2;
        pv += delta_t * nu_a;
        pyaw += 0.5 * delta_t * delta_t * nu_yawdd;
        pyawd += delta_t * nu_yawdd;

        Xsig_pred_(0, i) = ppx;
        Xsig_pred_(1, i) = ppy;
        Xsig_pred_(2, i) = pv;
        Xsig_pred_(3, i) = pyaw;
        Xsig_pred_(4, i) = pyawd;
    }

    // 3) Predict new mean and covariance
    // create vector for new predicted state
    VectorXd x = VectorXd(n_x_);
    // create covariance matrix for new prediction
    MatrixXd P = MatrixXd(n_x_, n_x_);

    // predict state mean
    x.fill(0.0);
    for(int  i = 0; i < 2 * n_aug_ + 1; i++)
    {
        x = x + weights_(i) * Xsig_pred_.col(i);
    }

    // predict state covariance matrix
    P.fill(0.0);
    for(int  i = 0; i < 2 * n_aug_ + 1; i++)
    {
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x;
        // angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        P = P + weights_(i) * x_diff * x_diff.transpose() ;
    }
    std::cout << "x_: " << std::endl << x << std::endl;
    std::cout << "P_: " << std::endl << P << std::endl;

    // Update mean and covariance matrix
    x_ = x;
    P_ = P;
}

void UKF::UpdateLidar(MeasurementPackage meas_package)
{
    /**
     * TODO: Complete this function! Use lidar data to update the belief
     * about the object's position. Modify the state vector, x_, and
     * covariance, P_.
     * You can also calculate the lidar NIS, if desired.
     */
}

void UKF::UpdateRadar(MeasurementPackage meas_package)
{
    /**
     * TODO: Complete this function! Use radar data to update the belief
     * about the object's position. Modify the state vector, x_, and
     * covariance, P_.
     * You can also calculate the radar NIS, if desired.
     */
}

void UKF::InitUKF(const MeasurementPackage &meas_package)
{
    // Initialize filter based on type of measurement
    if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER)
    {
        // Laser measurement
        double px = meas_package.raw_measurements_[0];
        double py = meas_package.raw_measurements_[1];
        x_(0) = px;
        x_(1) = py;
    }
    else
    {
        // Radar measurement
        double rho = meas_package.raw_measurements_[0];
        double phi = meas_package.raw_measurements_[1];
        double rho_dot = meas_package.raw_measurements_[2];

        double px = rho * cos(phi);
        double py = rho * sin(phi);

        double vx = rho_dot * cos(phi);
        double vy = rho_dot * sin(phi);
        double v = sqrt((vx * vx) + (vy * vy));

        x_(0) = px;
        x_(1) = py;
        x_(2) = v;
    }

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
}

void UKF::Update(const MeasurementPackage &meas_package)
{
    if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER)
    {
        UpdateLidar(meas_package);
    }
    else
    {
        UpdateRadar(meas_package);
    }
}
