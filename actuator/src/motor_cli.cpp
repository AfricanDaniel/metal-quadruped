// Pure C++ motor test tool — no ROS 2 involved at runtime. Useful for quick
// hardware bring-up / bench testing of a single motor before wiring it into
// the `basic_control` ROS 2 node's services.
//
// Usage: motor_cli <motor_id> [port]
#include <iostream>
#include <unistd.h>
#include <csignal>
#include <cstdlib>
#include "serialPort/SerialPort.h"
#include "unitreeMotor/unitreeMotor.h"

static bool running = true;
void sigint_handler(int) { running = false; }

int main(int argc, char* argv[]) {
    const char* port = (argc > 2) ? argv[2] : "/dev/ttyUSB0";
    int motor_id     = (argc > 1) ? std::atoi(argv[1]) : 0;

    std::signal(SIGINT, sigint_handler);

    SerialPort serial(port);

    MotorCmd  cmd;
    MotorData data;

    cmd.motorType  = MotorType::GO_M8010_6;
    data.motorType = MotorType::GO_M8010_6;
    cmd.id         = motor_id;
    cmd.mode       = queryMotorMode(MotorType::GO_M8010_6, MotorMode::FOC);
    cmd.q          = 0.0f;
    cmd.dq         = 0.0f;
    cmd.tau        = 0.0f;
    cmd.kp         = 0.1f;
    cmd.kd         = 0.01f;

    std::cout << "Testing GO-M8010-6 motor ID " << motor_id
              << " on " << port << " — Ctrl+C to stop\n";

    while (running) {
        serial.sendRecv(&cmd, &data);

        std::cout << "ID:" << motor_id
                  << "  q:"    << data.q    << " rad"
                  << "  dq:"   << data.dq   << " rad/s"
                  << "  tau:"  << data.tau  << " Nm"
                  << "  temp:" << data.temp << "C"
                  << "  err:"  << data.merror << "\n";

        usleep(10000); // 100 Hz
    }

    // Zero torque on exit
    cmd.kp  = 0.0f;
    cmd.kd  = 0.0f;
    cmd.tau = 0.0f;
    serial.sendRecv(&cmd, &data);

    std::cout << "\nMotor " << motor_id << " disabled.\n";
    return 0;
}
