#include "worker.hpp"

#include <signal.h>
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

// For Linux, use prctl to set the parent process death signal
#if defined(__linux__)
#include <sys/prctl.h>

#include <csignal>
#include <cstdio>
#endif

void setup_parent_exit_guard(const pid_t ppid)
{
    if (kill(ppid, 0) != 0) {
        std::_Exit(0);
    }
#if defined(__linux__)
    // Exit when parent process exits
    if (prctl(PR_SET_PDEATHSIG, SIGTERM) == -1) {
        perror("prctl(PR_SET_PDEATHSIG) failed");
        std::_Exit(1);
    }
#else
    // Start a watchdog thread to monitor the parent process
    std::thread([ppid]() {
        while (kill(ppid, 0) == 0) {
            sleep(1);
        }
        std::_Exit(0);
    }).detach();
#endif
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
