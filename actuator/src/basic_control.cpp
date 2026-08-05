#include <algorithm>
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
#include "actuator/srv/set_motor_torque.hpp"

namespace {
const std::string kPresetPoseFile  = std::string(ACTUATOR_PACKAGE_DIR) + "/config/preset_pose.yaml";
const std::string kLogCsvDir       = std::string(ACTUATOR_PACKAGE_DIR) + "/data/csv_logs";

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

// Auto-generated destination for go_to_pose's log_csv, used when the caller
// leaves log_csv_path empty -- see GoToPose.srv and handle_go_to_pose().
std::string timestamped_log_path(const std::string& pose_name) {
    std::filesystem::create_directories(kLogCsvDir);

    auto now        = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
    localtime_r(&now_c, &tm_buf);

    std::ostringstream path;
    path << kLogCsvDir << "/go_to_pose_" << pose_name << "_"
         << std::put_time(&tm_buf, "%Y%m%d_%H%M%S") << ".csv";
    return path.str();
}
}  // namespace

enum class ControlMode { NONE, VELOCITY, POSITION, TORQUE };

// Per-motor command/feedback state. A motor only gets an entry here once a
// service call targets its ID — until then, nothing is sent to it.
struct MotorState {
    MotorCmd     cmd{};
    MotorData    data{};
    ControlMode  mode = ControlMode::NONE;
    float        target_position_rad = 0.0f; // final destination of the current move

    // Linear position ramp: control_loop interpolates cmd.q from
    // ramp_start_rad toward target_position_rad over ramp_duration_s,
    // instead of stepping straight to the target. ramp_duration_s <= 0
    // means "jump immediately" (no ramp).
    float ramp_start_rad = 0.0f;
    float ramp_duration_s = 0.0f;
    std::chrono::steady_clock::time_point ramp_start_time{};

    // TORQUE mode watchdog: control_loop() zeros cmd.tau if this goes
    // stale beyond torque_timeout_s -- see handle_set_motor_torque()'s
    // comment for why raw torque needs this safety net that
    // position/velocity mode don't (no bounded target/PD holding it
    // safe if the client hangs or crashes).
    std::chrono::steady_clock::time_point last_torque_cmd_time{};
};

class MotorTestNode : public rclcpp::Node {
public:
    MotorTestNode() : Node("motor_test") {
        // Declare hardware parameters
        this->declare_parameter("port", "/dev/ttyUSB0");

        // Declare velocity control parameters
        this->declare_parameter("kd_gain", 2.0); // Velocity stiffness

        // Declare position control parameters (used by the adjust_motor_position service)
        this->declare_parameter("position_kp", 60.0);
        this->declare_parameter("position_kd", 8.0);

        // Default speed (output-shaft degrees/sec) for position-mode moves.
        // Read live (not cached) so `ros2 param set` takes effect on the next
        // move without restarting the node. go_to_pose can also override it
        // per-call via the request's speed_deg_s field.
        this->declare_parameter("pose_speed_deg_s", 30.0);

        // TORQUE mode (2026-08-04, added alongside set_motor_torque for a
        // sim-trained torque-control RL policy -- see set_motor_torque.srv
        // and handle_set_motor_torque()). Raw torque has NO firmware-side
        // position clamp protecting it the way position mode's <joint
        // range> equivalent does in sim -- max_torque_nm is a hard,
        // server-side safety ceiling enforced regardless of what any
        // client requests.
        //
        // PER-MOTOR array (2026-08-04, was a single shared double):
        // real-hardware data (PPO_5000000_stand_policy_torque_v8_2.csv,
        // run at a uniform 1.0 N*m) showed the 4 THIGH motors (1,4,5,8)
        // pinned at the ceiling 99.7% of the run -- genuinely not enough
        // torque to push off -- while the 4 CALF motors (2,3,6,7) swung
        // through 150-200+ degrees at the SAME shared limit (a foot
        // losing grip, nothing checking the leg's motion) -- confirms
        // thighs and calves need independently-tunable ceilings, not one
        // shared number that's simultaneously too weak for one pair and
        // too strong for the other. Indexed by motor_id-1 (motor order:
        // 1/4/5/8=thigh, 2/3/6/7=calf, see motor_mapping.yaml). Default
        // keeps every motor at the previous uniform 1.0 N*m fallback --
        // raise the thigh entries deliberately once calf slipping is
        // otherwise addressed (see dog_gym's --domain-randomization for
        // ground-friction robustness, a separate fix for the slip itself
        // that this array doesn't replace).
        this->declare_parameter("max_torque_nm", std::vector<double>(8, 1.0));
        // Watchdog: a TORQUE-mode motor gets its cmd.tau force-zeroed by
        // control_loop() if it hasn't received a fresh set_motor_torque
        // call within this many seconds -- see MotorState::
        // last_torque_cmd_time's comment. Position mode doesn't need this
        // (a stale target is still just a bounded, PD-tracked position);
        // an unbounded, un-refreshed torque command from a hung/crashed
        // policy_node would otherwise keep being blindly resent forever.
        this->declare_parameter("torque_timeout_s", 0.5);
        // Velocity damping for TORQUE mode (2026-08-04, user request --
        // motivated by a real breakaway event: PPO_5000000_stand_policy_
        // torque_v8_5.csv showed torque ramping smoothly 0.2->4.0 N*m
        // while a thigh sat essentially motionless (static friction /
        // stiction holding it) until ~2.6-2.8 N*m, at which point it
        // broke free -- kinetic friction is lower than static, so the
        // NET torque driving acceleration jumped discontinuously right
        // as commanded torque kept climbing, and with kp=kd=0 (pure
        // feedforward, nothing resisting velocity) nothing braked it
        // until it slammed into its mechanical end stop (real_velocity_
        // deg_s: 21 -> 85 -> 300 -> 1682 across 4 ticks).
        //
        // kp stays 0 -- this does NOT turn torque mode into position
        // mode, there is still no target angle being tracked, the
        // policy still chooses the force. Only kd goes nonzero: tau_
        // actual = tau_requested - kd*qvel, a physical brake proportional
        // to velocity, applied by the FIRMWARE every tick regardless of
        // what the policy does -- unlike a reward-shaped fix (see
        // dog_gym's joint_velocity_penalty), this acts even in exactly
        // the breakaway scenario above that sim never modeled (MuJoCo's
        // <motor> actuators have no static/kinetic friction distinction),
        // so it doesn't depend on the policy having specifically learned
        // to handle a breakaway it never experienced in training.
        //
        // OUTPUT-shaft units (N*m per rad/s of OUTPUT angular velocity),
        // matching position_kp/position_kd's convention (divided by
        // gear_ratio^2 below to get the rotor-side value the SDK
        // actually wants) -- NOT velocity mode's kd_gain convention
        // (applied raw/unconverted, matching a vendor example) -- chosen
        // deliberately so dog_gym's sim-side TORQUE_KD_GAIN (dog_env.py)
        // means the exact same physical quantity as this parameter; the
        // two MUST be kept equal or sim training and real deployment
        // disagree about how much free braking exists.
        //
        // 0.2 N*m/(rad/s) is an UNTUNED placeholder: at a dangerous
        // ~17.5 rad/s (1000+ deg/s, matching the breakaway above) it
        // generates ~3.5 N*m of counter-torque -- comparable to the
        // torques actually seen -- while at a modest ~0.35 rad/s
        // (~20 deg/s, ordinary controlled motion) it only generates
        // ~0.07 N*m, small enough not to meaningfully fight genuine
        // slow movement. Needs real empirical tuning, same as position_
        // kp/kd's own placeholder status.
        this->declare_parameter("torque_kd_gain", 0.2);

        port_        = this->get_parameter("port").as_string();
        kd_gain_     = this->get_parameter("kd_gain").as_double();
        position_kp_ = this->get_parameter("position_kp").as_double();
        position_kd_ = this->get_parameter("position_kd").as_double();
        {
            auto limits = this->get_parameter("max_torque_nm").as_double_array();
            std::ostringstream limits_str;
            for (size_t i = 0; i < limits.size(); ++i) {
                limits_str << (i == 0 ? "" : ", ") << limits[i];
            }
            RCLCPP_INFO(get_logger(),
                "TORQUE mode safety limits: max_torque_nm=[%s] (motor 1..8 order), "
                "torque_timeout_s=%.2f, torque_kd_gain=%.2f N*m/(rad/s) output-shaft "
                "(all live-adjustable via `ros2 param set`, no restart needed)",
                limits_str.str().c_str(), this->get_parameter("torque_timeout_s").as_double(),
                this->get_parameter("torque_kd_gain").as_double());
        }

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

        torque_service_ = this->create_service<actuator::srv::SetMotorTorque>(
            "set_motor_torque",
            std::bind(&MotorTestNode::handle_set_motor_torque, this,
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

        motor.mode = ControlMode::VELOCITY;

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

    // Read live (not cached), same reasoning as default_pose_speed_deg_s()
    // -- but more important here: a safety ceiling should be lowerable
    // instantly via `ros2 param set` without needing a node restart (which
    // would drop all motor state) if something looks wrong mid-run.
    //
    // Per-motor (2026-08-04, was a single shared double -- see this
    // param's declare_parameter comment). FALLS BACK to a conservative
    // 1.0 N*m -- not the raw request, not unlimited -- for any motor_id
    // without an explicit entry (array shorter than 8, or motor_id out
    // of [1,8]), and logs an error every time that happens: a
    // misconfigured safety parameter should fail LOUD and SAFE, never
    // silently leave a motor unclamped.
    float max_torque_nm(int32_t motor_id) {
        static constexpr float kFallbackNm = 1.0f;
        auto limits = this->get_parameter("max_torque_nm").as_double_array();
        size_t idx = static_cast<size_t>(motor_id - 1);
        if (motor_id < 1 || idx >= limits.size()) {
            RCLCPP_ERROR(get_logger(),
                "max_torque_nm has no entry for motor %d (array size %zu) -- "
                "falling back to %.2f N*m. Fix the max_torque_nm parameter "
                "(needs exactly 8 values, motor 1..8 order).",
                motor_id, limits.size(), kFallbackNm);
            return kFallbackNm;
        }
        return static_cast<float>(limits[idx]);
    }

    float torque_timeout_s() {
        return static_cast<float>(this->get_parameter("torque_timeout_s").as_double());
    }

    // Rotor-side kd the SDK actually wants -- see torque_kd_gain's
    // declare_parameter comment for the output->rotor conversion
    // reasoning (same convention as position_kp_/position_kd_, /gear_
    // ratio_^2, NOT velocity mode's unconverted kd_gain_).
    float torque_kd_gain_rotor() {
        double output_kd = this->get_parameter("torque_kd_gain").as_double();
        return static_cast<float>(output_kd / (gear_ratio_ * gear_ratio_));
    }

    // Puts a motor into position mode (opening a new log file if it wasn't
    // already in position mode) and commands it to an ABSOLUTE output-shaft
    // angle, ramped linearly at speed_deg_s (deg/s) instead of stepping
    // straight to the target. speed_deg_s <= 0 jumps immediately. Shared by
    // handle_adjust_position (relative moves) and handle_go_to_pose
    // (absolute preset poses).
    void command_absolute_position(MotorState &motor, int32_t motor_id, float target_deg, float speed_deg_s) {
        motor.mode = ControlMode::POSITION;

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

    // Starts (or restarts) the node's single active CSV log, covering the
    // given motors for the next `duration_s` seconds of control_loop ticks
    // -- see GoToPose.srv's log_csv field and control_loop()'s write-out.
    // Only one log runs at a time; a call that arrives mid-log (a second
    // go_to_pose with log_csv before the first one's move finished) closes
    // the in-progress file first rather than interleaving both moves' rows
    // into it.
    void start_csv_log(const std::vector<int32_t>& motor_ids, const std::string& path, float duration_s) {
        if (logging_active_ && log_stream_.is_open()) {
            RCLCPP_WARN(get_logger(),
                "log_csv: closing in-progress log %s early -- a new go_to_pose(log_csv=true) call started.",
                log_stream_path_.c_str());
            log_stream_.close();
        }

        log_stream_.open(path);
        if (!log_stream_.is_open()) {
            RCLCPP_ERROR(get_logger(), "log_csv: failed to open %s -- nothing will be logged.", path.c_str());
            logging_active_ = false;
            return;
        }
        log_stream_ << "tick,elapsed_s,motor_id,position_deg,velocity_deg_s,torque_nm,temp_c,error_code\n";

        log_motor_ids_   = motor_ids;
        log_stream_path_ = path;
        log_tick_        = 0;
        log_start_time_  = std::chrono::steady_clock::now();
        // A little cushion past the move's own ramp duration so the final,
        // settled reading gets captured too, not just the last in-flight one.
        log_deadline_ = log_start_time_ +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<float>(duration_s + 0.2f));
        logging_active_ = true;

        RCLCPP_INFO(get_logger(), "log_csv: logging %zu motor(s) to %s for %.2fs",
                    motor_ids.size(), path.c_str(), duration_s);
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

        std::vector<int32_t> pose_motor_ids;
        float max_ramp_duration_s = 0.0f;
        for (const auto& [motor_id, offset_deg] : *pose) {
            float target_deg = home_deg_.at(motor_id) + offset_deg;
            MotorState &motor = get_or_create_motor(motor_id);
            command_absolute_position(motor, motor_id, target_deg, speed_deg_s);
            pose_motor_ids.push_back(motor_id);
            max_ramp_duration_s = std::max(max_ramp_duration_s, motor.ramp_duration_s);
            RCLCPP_INFO(get_logger(),
                "Motor %d: moving to '%s' (home %.2f + offset %.2f = %.2f deg) at %.1f deg/s",
                motor_id, request->pose_name.c_str(), home_deg_.at(motor_id), offset_deg,
                target_deg, speed_deg_s);
        }

        response->success = true;
        response->message = "Moving " + std::to_string(pose->size()) +
                             " motor(s) to pose '" + request->pose_name + "' at " +
                             std::to_string(speed_deg_s) + " deg/s.";

        if (request->log_csv) {
            std::string path = request->log_csv_path.empty()
                ? timestamped_log_path(request->pose_name)
                : request->log_csv_path;
            start_csv_log(pose_motor_ids, path, max_ramp_duration_s);
            response->message += " Logging to " + path + ".";
        }
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
            // motor.data.tau is ROTOR-side, same convention as data.q/
            // data.dq (see their own comments) -- but torque converts
            // the OPPOSITE direction: a reduction gearbox MULTIPLIES
            // torque going rotor -> output (it DIVIDES q/dq), matching
            // the inverse of handle_set_motor_torque's output->rotor
            // conversion (cmd.tau = requested_nm / gear_ratio_) elsewhere
            // in this file. Real, MEASURED torque -- reported by the
            // motor's own feedback in every control mode, not just
            // torque mode (e.g. a position-mode motor holding a pose
            // against gravity genuinely reports nonzero torque here).
            float torque_nm = motor.data.tau * gear_ratio_;
            response->motor_id.push_back(motor_id);
            response->position_deg.push_back(position_deg);
            response->velocity_deg_s.push_back(velocity_deg_s);
            response->torque_nm.push_back(torque_nm);

            RCLCPP_INFO(get_logger(), "  Motor %d: %.2f deg, %.2f N*m", motor_id, position_deg, torque_nm);
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

    // Sets a raw output-shaft TORQUE target for each listed motor -- no PD
    // position/velocity tracking at all (kp=kd=0), matching dog_gym's sim
    // <motor> torque actuators exactly (gear="1" there too, so the sim
    // policy's output IS output-shaft N*m with no separate gearbox
    // modeled) -- see set_motor_torque.srv and dog_deploy/policy_node.py's
    // control_mode='torque' path.
    //
    // cmd.tau is ROTOR-side (same convention as cmd.q/cmd.dq -- see their
    // "output -> rotor" comments elsewhere in this file), but the
    // conversion direction is the OPPOSITE of position/velocity's: a
    // reduction gearbox multiplies torque by gear_ratio_ going rotor ->
    // output, so converting the other way (output target -> rotor cmd)
    // means DIVIDING by gear_ratio_, not multiplying.
    //
    // max_torque_nm(motor_id) is a hard, PER-MOTOR, server-side safety
    // ceiling applied regardless of what the client requests -- see its
    // declare_parameter comment. This is the ONLY thing standing between
    // a bad policy output and full motor torque; unlike position mode
    // there's no <joint range>-equivalent hard stop protecting this on
    // the real robot.
    void handle_set_motor_torque(
        const std::shared_ptr<actuator::srv::SetMotorTorque::Request> request,
        std::shared_ptr<actuator::srv::SetMotorTorque::Response> response) {
        if (request->motor_id.size() != request->torque_nm.size()) {
            RCLCPP_ERROR(get_logger(),
                "set_motor_torque: motor_id (%zu) and torque_nm (%zu) size mismatch",
                request->motor_id.size(), request->torque_nm.size());
            response->success = false;
            return;
        }

        const auto now = std::chrono::steady_clock::now();
        for (size_t i = 0; i < request->motor_id.size(); ++i) {
            int32_t motor_id = request->motor_id[i];
            float limit = max_torque_nm(motor_id);
            float requested_nm = request->torque_nm[i];
            float clamped_nm = std::max(-limit, std::min(limit, requested_nm));
            if (clamped_nm != requested_nm) {
                RCLCPP_WARN(get_logger(),
                    "Motor %d: requested torque %.2f N*m clamped to %.2f N*m (max_torque_nm[%d]=%.2f)",
                    motor_id, requested_nm, clamped_nm, motor_id, limit);
            }

            MotorState &motor = get_or_create_motor(motor_id);
            if (motor.mode != ControlMode::TORQUE) {
                motor.mode = ControlMode::TORQUE;
            }
            motor.cmd.kp  = 0.0f;               // still no position tracking -- see torque_kd_gain's comment
            motor.cmd.kd  = torque_kd_gain_rotor();  // velocity damping brake, kp=0 keeps this true torque control
            motor.cmd.dq  = 0.0f;               // damping target is zero velocity, not a moving setpoint
            motor.cmd.tau = clamped_nm / gear_ratio_;  // output N*m -> rotor N*m
            motor.last_torque_cmd_time = now;
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
            } else if (motor.mode == ControlMode::TORQUE) {
                // Watchdog (see MotorState::last_torque_cmd_time's comment
                // and handle_set_motor_torque()): a stale, un-refreshed
                // torque command would otherwise keep being blindly resent
                // by sendRecv() below forever, unlike position mode where
                // a stale target is at least a bounded, PD-tracked pose.
                // Only cmd.tau gets zeroed here -- cmd.kd (set in
                // handle_set_motor_torque, torque_kd_gain's damping) is
                // deliberately left alone, so a timed-out motor becomes a
                // pure velocity damper (resists motion, settles to rest)
                // rather than a fully passive/free-spinning joint.
                float age_s = std::chrono::duration<float>(
                    std::chrono::steady_clock::now() - motor.last_torque_cmd_time).count();
                if (age_s > torque_timeout_s()) {
                    if (motor.cmd.tau != 0.0f) {
                        RCLCPP_WARN(get_logger(),
                            "Motor %d: no set_motor_torque call in %.2fs (> torque_timeout_s=%.2f) "
                            "-- zeroing torque.", motor_id, age_s, torque_timeout_s());
                    }
                    motor.cmd.tau = 0.0f;
                }
            }

            // Send command and read state
            serial_->sendRecv(&motor.cmd, &motor.data);

            // Feedback is rotor-side; convert to output-shaft units for logging.
            const float output_q  = motor.data.q / gear_ratio_;
            const float output_dq = motor.data.dq / gear_ratio_;

            // Log the state (Useful for checking if it's hitting the target velocity/position).
            // DEBUG level so it doesn't spam the terminal at 100 Hz by default.
            RCLCPP_DEBUG(get_logger(),
                "ID:%d  q:%.3f rad  dq:%.3f rad/s  tau:%.3f Nm  temp:%d°C  err:%d",
                motor_id, output_q, output_dq, motor.data.tau, motor.data.temp, motor.data.merror);
        }

        // go_to_pose's log_csv (see start_csv_log()): write one row per
        // logged motor for this tick, using the sendRecv feedback the loop
        // above just refreshed. Runs until log_deadline_, then closes itself
        // -- no explicit "stop logging" call needed from the caller.
        if (logging_active_) {
            auto now = std::chrono::steady_clock::now();
            if (now >= log_deadline_) {
                log_stream_.close();
                logging_active_ = false;
                RCLCPP_INFO(get_logger(), "log_csv: done, wrote %s", log_stream_path_.c_str());
            } else {
                float elapsed_s = std::chrono::duration<float>(now - log_start_time_).count();
                for (int32_t motor_id : log_motor_ids_) {
                    auto it = motors_.find(motor_id);
                    if (it == motors_.end()) {
                        continue;
                    }
                    MotorState &motor = it->second;
                    float position_deg   = (motor.data.q  / gear_ratio_) * 180.0f / static_cast<float>(M_PI);
                    float velocity_deg_s = (motor.data.dq / gear_ratio_) * 180.0f / static_cast<float>(M_PI);
                    float torque_nm      = motor.data.tau * gear_ratio_;
                    log_stream_ << log_tick_ << ',' << std::fixed << std::setprecision(4) << elapsed_s << ','
                                << motor_id << ',' << position_deg << ',' << velocity_deg_s << ','
                                << torque_nm << ',' << motor.data.temp << ',' << motor.data.merror << '\n';
                }
                ++log_tick_;
            }
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
    rclcpp::Service<actuator::srv::SetMotorTorque>::SharedPtr      torque_service_;

    // motor_id -> output-shaft degrees captured by the last set_home call.
    // Empty until set_home is called; not persisted across node restarts —
    // see README "Homing / reference position".
    std::unordered_map<int32_t, float> home_deg_;

    // go_to_pose's log_csv state -- see start_csv_log() and control_loop()'s
    // write-out. One log active at a time, node-wide (not per-motor): a
    // go_to_pose call typically drives several motors together, and a
    // single file with all of them interleaved by tick is what's useful for
    // comparing them, not one file per motor.
    std::ofstream                         log_stream_;
    std::string                           log_stream_path_;
    bool                                  logging_active_ = false;
    std::vector<int32_t>                  log_motor_ids_;
    int64_t                               log_tick_ = 0;
    std::chrono::steady_clock::time_point log_start_time_{};
    std::chrono::steady_clock::time_point log_deadline_{};

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
