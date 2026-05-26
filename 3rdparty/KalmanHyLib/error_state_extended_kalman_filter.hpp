#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <ceres/jet.h>
#include <functional>
#include <limits>

namespace kalman_hybird_lib {

template<int N_X, class PredicFunc>
class ErrorStateEKF {
public:
    using MatrixXX = Eigen::Matrix<double, N_X, N_X>;
    using MatrixX1 = Eigen::Matrix<double, N_X, 1>;

    using UpdateQFunc = std::function<MatrixXX()>;
    using InjectFunc = std::function<void(const MatrixX1&, MatrixX1&)>;

    ErrorStateEKF() = default;

    explicit ErrorStateEKF(
        const PredicFunc& f,
        const UpdateQFunc& u_q,
        const InjectFunc& inject,
        const MatrixXX& P0
    ) noexcept:
        f(f),
        update_Q(u_q),
        inject_state(inject),
        P_delta(P0) {}

    void set_state(const MatrixX1& x0) noexcept {
        x_nominal = x0;
        delta_x.setZero();
    }

    void set_update_Q(const UpdateQFunc& u_q) {
        update_Q = u_q;
    }
    void set_predict_func(const PredicFunc& f) {
        this->f = f;
    }
    void set_inject_state(const InjectFunc& inject) {
        inject_state = inject;
    }

    MatrixX1 predict() noexcept {
        std::array<ceres::Jet<double, N_X>, N_X> x_jet;

        for (int i = 0; i < N_X; ++i) {
            x_jet[i].a = x_nominal[i];
            x_jet[i].v.setZero();
            x_jet[i].v[i] = 1.0;
        }

        std::array<ceres::Jet<double, N_X>, N_X> x_pred_jet;
        f(x_jet.data(), x_pred_jet.data());

        for (int i = 0; i < N_X; ++i) {
            x_nominal[i] = x_pred_jet[i].a;
            F.row(i) = x_pred_jet[i].v.transpose();
        }

        // error propagation
        delta_x = F * delta_x;

        // covariance
        Q = update_Q();
        P_delta = F * P_delta * F.transpose() + Q;
        P_delta = 0.5 * (P_delta + P_delta.transpose());

        return x_nominal;
    }

    template<int N_Z, class MeasureFunc, class UpdateRFunc, class ResidualFunc>
    MatrixX1 update(
        const Eigen::Matrix<double, N_Z, 1>& z,
        const MeasureFunc& h,
        const UpdateRFunc& update_R,
        const ResidualFunc& cal_residual
    ) noexcept {
        using MatrixZ1 = Eigen::Matrix<double, N_Z, 1>;
        using MatrixZX = Eigen::Matrix<double, N_Z, N_X>;
        using MatrixXZ = Eigen::Matrix<double, N_X, N_Z>;
        using MatrixZZ = Eigen::Matrix<double, N_Z, N_Z>;

        MatrixX1 delta_iter = delta_x;
        MatrixXX P_iter = P_delta;

        MatrixZX H = MatrixZX::Zero();
        MatrixXZ K = MatrixXZ::Zero();
        MatrixZZ R = MatrixZZ::Zero();

        for (int iter = 0; iter < iteration_num; ++iter) {
            MatrixX1 x_eval = x_nominal;
            if (inject_state)
                inject_state(delta_iter, x_eval);

            std::array<ceres::Jet<double, N_X>, N_X> x_jet;
            for (int i = 0; i < N_X; ++i) {
                x_jet[i].a = x_eval[i];
                x_jet[i].v.setZero();
                x_jet[i].v[i] = 1.0;
            }

            std::array<ceres::Jet<double, N_X>, N_Z> z_jet;
            h(x_jet.data(), z_jet.data());

            MatrixZ1 z_pred;
            for (int i = 0; i < N_Z; ++i) {
                z_pred[i] = z_jet[i].a;
                H.row(i) = z_jet[i].v.transpose();
            }

            // residual
            MatrixZ1 residual = cal_residual(z_pred, z);

            // covariance
            R = update_R(z);
            MatrixZZ S = H * P_iter * H.transpose() + R;
            K = P_iter * H.transpose() * S.ldlt().solve(MatrixZZ::Identity());
            delta_iter += K * residual;
        }

        // --- inject ---
        if (inject_state)
            inject_state(delta_iter, x_nominal);

        // --- covariance update ---
        MatrixXX I = MatrixXX::Identity();
        P_delta = (I - K * H) * P_iter * (I - K * H).transpose() + K * R * K.transpose();

        P_delta = 0.5 * (P_delta + P_delta.transpose());

        return x_nominal;
    }

    void set_iteration_num(int n) {
        iteration_num = std::max(1, n);
    }

private:
    PredicFunc f;
    UpdateQFunc update_Q;
    InjectFunc inject_state;

    MatrixXX F = MatrixXX::Zero();
    MatrixXX Q = MatrixXX::Zero();

    MatrixX1 x_nominal = MatrixX1::Zero();
    MatrixX1 delta_x = MatrixX1::Zero();
    MatrixXX P_delta = MatrixXX::Identity();

    int iteration_num = 1;
};

} // namespace kalman_hybird_lib