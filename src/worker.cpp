#include "worker.hpp"

#include <sys/types.h>
#include <unistd.h>

#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

void setup_parent_exit_guard(const pid_t ppid)
{
    if (kill(ppid, 0) != 0) {
        std::_Exit(0);
    }
    std::thread([ppid]() {
        while (kill(ppid, 0) == 0) {
            sleep(1);
        }
        std::_Exit(0);
    }).detach();
}

int main(const int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: worker <device_id>" << std::endl;
        return 1;
    }

    pid_t ppid = getppid();
    int32_t device_id = std::stoi(argv[1]);
    setup_parent_exit_guard(ppid);

    // Ignore SIGINT in the worker process to prevent it from being terminated
    // by Ctrl-C from the terminal.
    signal(SIGINT, SIG_IGN);

    std::unique_ptr<xspex::worker::Worker> worker;

    try {
        worker = std::make_unique<xspex::worker::Worker>(ppid, device_id);
    } catch (const std::runtime_error& e) {
        std::cerr << "Failed to initialize worker for device " << device_id
                  << " -> " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Failed to initialize worker for device " << device_id
                  << " -> unknown error" << std::endl;
        return 1;
    }

    try {
        worker->run_worker_loop();
    } catch (const std::runtime_error& e) {
        std::cerr << "Failed to enter worker loop for device " << device_id
                  << " -> " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
