#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <rclcpp/rclcpp.hpp>
#include "serialPort/SerialPort.h"
#include "unitreeMotor/unitreeMotor.h"
#include "actuator/srv/adjust_motor_position.hpp"
#include "actuator/srv/set_motor_velocity.hpp"
#include "actuator/srv/read_motor_positions.hpp"
#include "actuator/srv/go_to_pose.hpp"
#include "actuator/srv/set_home.hpp"
#include "actuator/srv/set_motor_targets.hpp"

namespace {
// const std::string kDataDir         = std::string(ACTUATOR_PACKAGE_DIR) + "/data";
// const std::string kVelocityLogDir  = kDataDir + "/velocity";
// const std::string kPositionLogDir  = kDataDir + "/position";
const std::string kPresetPoseFile  = std::string(ACTUATOR_PACKAGE_DIR) + "/config/preset_pose.yaml";

// Loads a named pose (motor_id -> target angle in degrees) from
// preset_pose.yaml. Re-reads the file every call, so poses added/edited
// there show up immediately without restarting the node. Returns nullopt if
// the file or pose is missing/malformed.
std::optional<std::map<int32_t, float>> load_pose(const std::string& pose_name,
                                                    const rclcpp::Logger& logger) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(kPresetPoseFile);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger, "Failed to read %s: %s", kPresetPoseFile.c_str(), e.what());
        return std::nullopt;
    }

    YAML::Node poses = root["poses"];
    if (!poses || !poses[pose_name]) {
        RCLCPP_ERROR(logger, "Pose '%s' not found in %s", pose_name.c_str(), kPresetPoseFile.c_str());
        return std::nullopt;
    }

    std::map<int32_t, float> result;
    for (const auto& entry : poses[pose_name]) {
        int32_t motor_id = entry.first.as<int32_t>();
        float   angle_deg = entry.second.as<float>();
        result[motor_id] = angle_deg;
    }
    return result;
}

// Opens a new timestamped CSV log file for a motor under the given directory.
// Motor data no longer needs to be saved to disk -- commented out rather
// than deleted in case per-motor CSV logging is wanted again later.
// std::ofstream open_motor_log(const std::string& dir, int32_t motor_id) {
//     std::filesystem::create_directories(dir);
//
//     auto now        = std::chrono::system_clock::now();
//     std::time_t now_c = std::chrono::system_clock::to_time_t(now);
//     std::tm tm_buf{};
//     localtime_r(&now_c, &tm_buf);
//
//     std::ostringstream path;
//     path << dir << "/motor_" << motor_id << "_"
//          << std::put_time(&tm_buf, "%Y%m%d_%H%M%S") << ".csv";
//
//     std::ofstream file(path.str());
//     file << "timestamp,motor_id,position,velocity,torque\n";
//     return file;
// }
}  // namespace

enum class ControlMode { NONE, VELOCITY, POSITION };

// Per-motor command/feedback state. A motor only gets an entry here once a
// service call targets its ID — until then, nothing is sent to it.
struct MotorState {
    MotorCmd     cmd{};
    MotorData    data{};
    ControlMode  mode = ControlMode::NONE;
    float        target_position_rad = 0.0f; // final destination of the current move
    std::ofstream log_file;

    // Linear position ramp: control_loop interpolates cmd.q from
    // ramp_start_rad toward target_position_rad over ramp_duration_s,
    // instead of stepping straight to the target. ramp_duration_s <= 0
    // means "jump immediately" (no ramp).
    float ramp_start_rad = 0.0f;
    float ramp_duration_s = 0.0f;
    std::chrono::steady_clock::time_point ramp_start_time{};
};

class MotorTestNode : public rclcpp::Node {
public:
    MotorTestNode() : Node("motor_test") {
        // Declare hardware parameters
        this->declare_parameter("port", "/dev/ttyUSB0");

        // Declare velocity control parameters
        this->declare_parameter("kd_gain", 0.05); // Velocity stiffness

        // Declare position control parameters (used by the adjust_motor_position service)
        this->declare_parameter("position_kp", 16.0);
        this->declare_parameter("position_kd", 0.2);

        // Default speed (output-shaft degrees/sec) for position-mode moves.
        // Read live (not cached) so `ros2 param set` takes effect on the next
        // move without restarting the node. go_to_pose can also override it
        // per-call via the request's speed_deg_s field.
        this->declare_parameter("pose_speed_deg_s", 30.0);

        port_        = this->get_parameter("port").as_string();
        kd_gain_     = this->get_parameter("kd_gain").as_double();
        position_kp_ = this->get_parameter("position_kp").as_double();
        position_kd_ = this->get_parameter("position_kd").as_double();

        // cmd.q/cmd.dq (and the data.q/data.dq feedback) are all on the ROTOR
        // side of the gearbox, not the output side. Everything below converts
        // to/from output-shaft units using this ratio, per the SDK README.
        gear_ratio_ = queryGearRatio(MotorType::GO_M8010_6);

        serial_ = std::make_unique<SerialPort>(port_.c_str());

        // 100 Hz control loop. With no motors registered yet, this is a no-op —
        // motors only start being commanded once a service call targets them.
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&MotorTestNode::control_loop, this));

        velocity_service_ = this->create_service<actuator::srv::SetMotorVelocity>(
            "set_motor_velocity",
            std::bind(&MotorTestNode::handle_set_velocity, this,
                       std::placeholders::_1, std::placeholders::_2));

        position_service_ = this->create_service<actuator::srv::AdjustMotorPosition>(
            "adjust_motor_position",
            std::bind(&MotorTestNode::handle_adjust_position, this,
                       std::placeholders::_1, std::placeholders::_2));

        read_positions_service_ = this->create_service<actuator::srv::ReadMotorPositions>(
            "read_motor_positions",
            std::bind(&MotorTestNode::handle_read_positions, this,
                       std::placeholders::_1, std::placeholders::_2));

        pose_service_ = this->create_service<actuator::srv::GoToPose>(
            "go_to_pose",
            std::bind(&MotorTestNode::handle_go_to_pose, this,
                       std::placeholders::_1, std::placeholders::_2));

        home_service_ = this->create_service<actuator::srv::SetHome>(
            "set_home",
            std::bind(&MotorTestNode::handle_set_home, this,
                       std::placeholders::_1, std::placeholders::_2));

        set_targets_service_ = this->create_service<actuator::srv::SetMotorTargets>(
            "set_motor_targets",
            std::bind(&MotorTestNode::handle_set_motor_targets, this,
                       std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(get_logger(),
            "Actuator node ready on port %s. No motors active — call "
            "set_motor_velocity or adjust_motor_position to start one.",
            port_.c_str());
    }

    ~MotorTestNode() {
        // Safely zero out torque on every motor that was ever commanded.
        for (auto &kv : motors_) {
            MotorState &motor = kv.second;
            motor.cmd.kp  = 0.0f;
            motor.cmd.kd  = 0.0f;
            motor.cmd.tau = 0.0f;
            motor.cmd.dq  = 0.0f;
            serial_->sendRecv(&motor.cmd, &motor.data);
            RCLCPP_INFO(get_logger(), "Motor %d disabled and zeroed.", kv.first);
        }
    }

private:
    MotorState& get_or_create_motor(int32_t motor_id) {
        auto it = motors_.find(motor_id);
        if (it != motors_.end()) {
            return it->second;
        }

        auto result = motors_.emplace(motor_id, MotorState{});
        MotorState &state = result.first->second;
        state.cmd.motorType  = MotorType::GO_M8010_6;
        state.data.motorType = MotorType::GO_M8010_6;
        state.cmd.id         = motor_id;
        state.cmd.mode       = queryMotorMode(MotorType::GO_M8010_6, MotorMode::FOC);
        state.cmd.q          = 0.0f;
        state.cmd.dq         = 0.0f;
        state.cmd.tau        = 0.0f;
        state.cmd.kp         = 0.0f;
        state.cmd.kd         = 0.0f;

        // Zero-effort probe: kp/kd/tau/dq are all 0, so this can't move the
        // motor, but it populates state.data with the real current reading.
        // Without this, state.data.q would still be its default-constructed
        // 0 the first time something latches onto "current position".
        serial_->sendRecv(&state.cmd, &state.data);

        RCLCPP_INFO(get_logger(), "Motor %d registered at %.2f deg.", motor_id,
                    (state.data.q / gear_ratio_) * 180.0f / static_cast<float>(M_PI));
        return state;
    }

    void handle_set_velocity(
        const std::shared_ptr<actuator::srv::SetMotorVelocity::Request> request,
        std::shared_ptr<actuator::srv::SetMotorVelocity::Response> response) {
        MotorState &motor = get_or_create_motor(request->motor_id);

        if (motor.mode != ControlMode::VELOCITY) {
            // motor.log_file = open_motor_log(kVelocityLogDir, request->motor_id);
            motor.mode = ControlMode::VELOCITY;
        }

        motor.cmd.q   = 0.0f;       // Position target is ignored when kp is 0
        motor.cmd.dq  = request->velocity * gear_ratio_; // output rad/s -> rotor rad/s
        motor.cmd.tau = 0.0f;
        motor.cmd.kp  = 0.0f;       // ZERO position gain (disables position hold)
        // kd is used directly (not divided by gear_ratio^2) — this matches the
        // vendor's own GO_M8010_6 velocity example, which tunes kd as a raw
        // rotor-side damping term rather than converting it from an
        // output-side value. The r^2 conversion is a position-mode concern.
        motor.cmd.kd  = kd_gain_;

        response->success = true;

        RCLCPP_INFO(get_logger(), "Motor %d: velocity control at %.2f rad/s",
                    request->motor_id, request->velocity);
    }

    float default_pose_speed_deg_s() {
        return this->get_parameter("pose_speed_deg_s").as_double();
    }

    // Puts a motor into position mode (opening a new log file if it wasn't
    // already in position mode) and commands it to an ABSOLUTE output-shaft
    // angle, ramped linearly at speed_deg_s (deg/s) instead of stepping
    // straight to the target. speed_deg_s <= 0 jumps immediately. Shared by
    // handle_adjust_position (relative moves) and handle_go_to_pose
    // (absolute preset poses).
    void command_absolute_position(MotorState &motor, int32_t motor_id, float target_deg, float speed_deg_s) {
        if (motor.mode != ControlMode::POSITION) {
            motor.mode = ControlMode::POSITION;
            // motor.log_file = open_motor_log(kPositionLogDir, motor_id);
        }

        // Ramp from wherever the motor actually is right now (not the old
        // target), so calling this again mid-ramp retargets smoothly instead
        // of jumping.
        float current_deg  = (motor.data.q / gear_ratio_) * 180.0f / static_cast<float>(M_PI);
        float distance_deg = std::fabs(target_deg - current_deg);

        motor.ramp_start_rad      = current_deg * static_cast<float>(M_PI) / 180.0f;
        motor.target_position_rad = target_deg * static_cast<float>(M_PI) / 180.0f;
        motor.ramp_start_time     = std::chrono::steady_clock::now();
        motor.ramp_duration_s     = (speed_deg_s > 0.0f) ? (distance_deg / speed_deg_s) : 0.0f;

        motor.cmd.tau = 0.0f;
        motor.cmd.kp  = position_kp_ / (gear_ratio_ * gear_ratio_); // output-side gain -> rotor-side
        motor.cmd.kd  = position_kd_ / (gear_ratio_ * gear_ratio_); // output-side gain -> rotor-side
    }

    void handle_adjust_position(
        const std::shared_ptr<actuator::srv::AdjustMotorPosition::Request> request,
        std::shared_ptr<actuator::srv::AdjustMotorPosition::Response> response) {
        MotorState &motor = get_or_create_motor(request->motor_id);

        // Base off the running position-mode target if there is one,
        // otherwise the motor's current measured position — so a relative
        // move from a freshly-registered motor starts from reality, not 0.
        float current_target_deg = (motor.mode == ControlMode::POSITION)
            ? motor.target_position_rad * 180.0f / static_cast<float>(M_PI)
            : (motor.data.q / gear_ratio_) * 180.0f / static_cast<float>(M_PI);

        float new_target_deg = current_target_deg + request->degrees;
        command_absolute_position(motor, request->motor_id, new_target_deg, default_pose_speed_deg_s());

        response->success = true;
        response->resulting_position_deg = new_target_deg;

        RCLCPP_INFO(get_logger(), "Motor %d: moving by %.2f deg -> target %.2f deg",
                    request->motor_id, request->degrees, response->resulting_position_deg);
    }

    void handle_go_to_pose(
        const std::shared_ptr<actuator::srv::GoToPose::Request> request,
        std::shared_ptr<actuator::srv::GoToPose::Response> response) {
        auto pose = load_pose(request->pose_name, get_logger());
        if (!pose) {
            response->success = false;
            response->message = "Pose '" + request->pose_name + "' not found or " +
                                 kPresetPoseFile + " could not be read.";
            return;
        }

        // Poses are stored as offsets from the home/reference position (the
        // GO-M8010-6 has no absolute encoder memory across power cycles — see
        // README). Refuse to guess: every motor in the pose must have been
        // homed this session via set_home before we'll move anything.
        for (const auto& [motor_id, offset_deg] : *pose) {
            (void)offset_deg;
            if (home_deg_.find(motor_id) == home_deg_.end()) {
                response->success = false;
                response->message = "Motor " + std::to_string(motor_id) +
                    " has no home/reference position set this session — "
                    "physically pose the robot and call set_home first.";
                RCLCPP_ERROR(get_logger(), "%s", response->message.c_str());
                return;
            }
        }

        // 0 (unset in the request) means "use the node's default speed".
        float speed_deg_s = (request->speed_deg_s > 0.0f)
            ? request->speed_deg_s
            : default_pose_speed_deg_s();

        for (const auto& [motor_id, offset_deg] : *pose) {
            float target_deg = home_deg_.at(motor_id) + offset_deg;
            MotorState &motor = get_or_create_motor(motor_id);
            command_absolute_position(motor, motor_id, target_deg, speed_deg_s);
            RCLCPP_INFO(get_logger(),
                "Motor %d: moving to '%s' (home %.2f + offset %.2f = %.2f deg) at %.1f deg/s",
                motor_id, request->pose_name.c_str(), home_deg_.at(motor_id), offset_deg,
                target_deg, speed_deg_s);
        }

        response->success = true;
        response->message = "Moving " + std::to_string(pose->size()) +
                             " motor(s) to pose '" + request->pose_name + "' at " +
                             std::to_string(speed_deg_s) + " deg/s.";
    }

    void handle_set_home(
        const std::shared_ptr<actuator::srv::SetHome::Request> /*request*/,
        std::shared_ptr<actuator::srv::SetHome::Response> response) {
        RCLCPP_INFO(get_logger(), "Setting home/reference position:");
        for (int32_t motor_id = 1; motor_id <= 8; ++motor_id) {
            // Same registration path as read_motor_positions: an
            // already-active motor reports its latest control-loop reading;
            // a motor seen for the first time gets a one-off zero-effort
            // probe read (can't move it) purely to capture where it is.
            MotorState &motor = get_or_create_motor(motor_id);

            float current_deg = (motor.data.q / gear_ratio_) * 180.0f / static_cast<float>(M_PI);
            home_deg_[motor_id] = current_deg;

            response->motor_id.push_back(motor_id);
            response->home_deg.push_back(current_deg);
            RCLCPP_INFO(get_logger(), "  Motor %d home: %.2f deg", motor_id, current_deg);
        }
        response->success = true;
    }

    void handle_read_positions(
        const std::shared_ptr<actuator::srv::ReadMotorPositions::Request> request,
        std::shared_ptr<actuator::srv::ReadMotorPositions::Response> response) {
        // Empty request->motor_id means "read all motors (1-8)"; otherwise
        // only read the IDs the caller asked for.
        std::vector<int32_t> ids_to_read = request->motor_id;
        if (ids_to_read.empty()) {
            for (int32_t motor_id = 1; motor_id <= 8; ++motor_id) {
                ids_to_read.push_back(motor_id);
            }
        }

        RCLCPP_INFO(get_logger(), "Motor positions:");
        for (int32_t motor_id : ids_to_read) {
            // Already-active motors get their latest feedback from the control
            // loop (refreshed every 10ms); a motor seen for the first time
            // gets registered here, which does a one-off zero-effort read.
            MotorState &motor = get_or_create_motor(motor_id);

            float position_deg    = (motor.data.q  / gear_ratio_) * 180.0f / static_cast<float>(M_PI);
            float velocity_deg_s  = (motor.data.dq / gear_ratio_) * 180.0f / static_cast<float>(M_PI);
            response->motor_id.push_back(motor_id);
            response->position_deg.push_back(position_deg);
            response->velocity_deg_s.push_back(velocity_deg_s);

            RCLCPP_INFO(get_logger(), "  Motor %d: %.2f deg", motor_id, position_deg);
        }
    }

    // Sets an absolute output-shaft target for each listed motor, jumping
    // immediately (no ramp) rather than interpolating like
    // adjust_motor_position/go_to_pose do. Meant for callers that already
    // command a full trajectory at a fixed rate (e.g. an RL policy), where
    // ramping would just fight the caller's own timing.
    void handle_set_motor_targets(
        const std::shared_ptr<actuator::srv::SetMotorTargets::Request> request,
        std::shared_ptr<actuator::srv::SetMotorTargets::Response> response) {
        if (request->motor_id.size() != request->position_deg.size()) {
            RCLCPP_ERROR(get_logger(),
                "set_motor_targets: motor_id (%zu) and position_deg (%zu) size mismatch",
                request->motor_id.size(), request->position_deg.size());
            response->success = false;
            return;
        }

        for (size_t i = 0; i < request->motor_id.size(); ++i) {
            int32_t motor_id  = request->motor_id[i];
            float   target_deg = request->position_deg[i];
            MotorState &motor = get_or_create_motor(motor_id);
            command_absolute_position(motor, motor_id, target_deg, /*speed_deg_s=*/0.0f);
        }

        response->success = true;
    }

    void control_loop() {
        for (auto &kv : motors_) {
            int32_t     motor_id = kv.first;
            MotorState &motor    = kv.second;

            if (motor.mode == ControlMode::POSITION) {
                float commanded_rad = motor.target_position_rad;
                if (motor.ramp_duration_s > 0.0f) {
                    float elapsed_s = std::chrono::duration<float>(
                        std::chrono::steady_clock::now() - motor.ramp_start_time).count();
                    float t = elapsed_s / motor.ramp_duration_s;
                    if (t < 1.0f) {
                        commanded_rad = motor.ramp_start_rad +
                            t * (motor.target_position_rad - motor.ramp_start_rad);
                    }
                    // t >= 1: ramp finished, commanded_rad stays at the final target.
                }
                motor.cmd.q  = commanded_rad * gear_ratio_; // output rad -> rotor rad
                motor.cmd.dq = 0.0f;
            }

            // Send command and read state
            serial_->sendRecv(&motor.cmd, &motor.data);

            // Feedback is rotor-side; convert to output-shaft units for logging.
            const float output_q  = motor.data.q / gear_ratio_;
            const float output_dq = motor.data.dq / gear_ratio_;

            // Motor data no longer needs to be saved to disk.
            // if (motor.mode != ControlMode::NONE && motor.log_file.is_open()) {
            //     auto now         = std::chrono::system_clock::now().time_since_epoch();
            //     double timestamp = std::chrono::duration<double>(now).count();
            //     motor.log_file << std::fixed << std::setprecision(6)
            //                     << timestamp << ',' << motor_id << ','
            //                     << output_q << ',' << output_dq << ','
            //                     << motor.data.tau << '\n';
            // }

            // Log the state (Useful for checking if it's hitting the target velocity/position).
            // DEBUG level so it doesn't spam the terminal at 100 Hz by default.
            RCLCPP_DEBUG(get_logger(),
                "ID:%d  q:%.3f rad  dq:%.3f rad/s  tau:%.3f Nm  temp:%d°C  err:%d",
                motor_id, output_q, output_dq, motor.data.tau, motor.data.temp, motor.data.merror);
        }
    }

    std::string                  port_;
    std::unique_ptr<SerialPort>  serial_;
    std::unordered_map<int32_t, MotorState> motors_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Service<actuator::srv::SetMotorVelocity>::SharedPtr   velocity_service_;
    rclcpp::Service<actuator::srv::AdjustMotorPosition>::SharedPtr position_service_;
    rclcpp::Service<actuator::srv::ReadMotorPositions>::SharedPtr  read_positions_service_;
    rclcpp::Service<actuator::srv::GoToPose>::SharedPtr            pose_service_;
    rclcpp::Service<actuator::srv::SetHome>::SharedPtr             home_service_;
    rclcpp::Service<actuator::srv::SetMotorTargets>::SharedPtr     set_targets_service_;

    // motor_id -> output-shaft degrees captured by the last set_home call.
    // Empty until set_home is called; not persisted across node restarts —
    // see README "Homing / reference position".
    std::unordered_map<int32_t, float> home_deg_;

    double position_kp_ = 0.0;
    double position_kd_ = 0.0;
    double kd_gain_     = 0.0;
    float  gear_ratio_  = 1.0f;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MotorTestNode>());
    rclcpp::shutdown();
    return 0;
}
