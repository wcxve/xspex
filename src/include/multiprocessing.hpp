#ifndef XSPEX_MULTIPROCESSING_HPP_
#define XSPEX_MULTIPROCESSING_HPP_

#include <sys/signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include "config.hpp"
#include "utils.hpp"

namespace xspex::multiprocessing
{
using Task = config::worker::Task;
using TaskStatus = std::pair<bool, std::string>;

class WorkerProcessManager
{
   public:
    // Constructor
    explicit WorkerProcessManager(const int32_t device_id)
        : device_id_{device_id},
          worker_shmem_manager_{getpid(), device_id, true}
    {
        start_worker_process();
        start_worker_monitor();
    }

    // Destructor
    ~WorkerProcessManager()
    {
        // Set running flag to false to exit worker loop and monitor loop
        worker_shmem_manager_.running(false);

        terminate_worker_process();
        terminate_worker_monitor();
    }

    // Delete copy constructor and copy assignment operator
    WorkerProcessManager(const WorkerProcessManager&) = delete;
    WorkerProcessManager& operator=(const WorkerProcessManager&) = delete;

    // Move constructor
    WorkerProcessManager(WorkerProcessManager&& other) noexcept
        : device_id_{other.device_id_},
          worker_pid_{other.worker_pid_},
          monitor_{std::move(other.monitor_)},
          worker_shmem_manager_{std::move(other.worker_shmem_manager_)}
    {
        other.worker_pid_ = -1;
    }

    // Move assignment operator
    WorkerProcessManager& operator=(WorkerProcessManager&& other) noexcept
    {
        if (this != &other) {
            // Cleanup current resources
            worker_shmem_manager_.running(false);
            terminate_worker_process();
            terminate_worker_monitor();

            // Move resources from other
            device_id_ = other.device_id_;
            worker_pid_ = other.worker_pid_;
            monitor_ = std::move(other.monitor_);
            worker_shmem_manager_ = std::move(other.worker_shmem_manager_);
            other.worker_pid_ = -1;
        }
        return *this;
    }

    [[nodiscard]] TaskStatus worker_execute_task(
        const Task task) const noexcept
    {
        if (!worker_shmem_manager_.running()) {
            std::cerr << "worker process exited unexpectedly" << std::endl;
            return {false, "worker process exited unexpectedly"};
        }
        worker_shmem_manager_.task(task);
        worker_shmem_manager_.success(false);
        worker_shmem_manager_.notify_task_start();
        worker_shmem_manager_.wait_for_task_end();
        return {worker_shmem_manager_.success(),
                worker_shmem_manager_.message()};
    }

    TaskStatus worker_execute_model_task(const uint32_t func_id,
                                         const double* params,
                                         const uint32_t n_params,
                                         const double* egrid,
                                         const uint32_t n_out,
                                         double* output,
                                         const int spec_num,
                                         const std::string& init_string,
                                         const double* input_model = nullptr)
    {
        // Buffer size should be >= params + egrid + model + model_error
        bool resized = false;
        try {
            resized =
                worker_shmem_manager_.resize_buffer(n_params + n_out * 3 + 1);
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "failed to resize buffer for device " << device_id_
                << " -> " << e.what();
            return {false, oss.str()};
        } catch (...) {
            std::ostringstream oss;
            oss << "failed to resize buffer for device " << device_id_
                << " -> unknown error";
            return {false, oss.str()};
        }

        if (resized) {
            const TaskStatus& ret = worker_execute_task(Task::ResizeBuffer);
            if (!ret.first) {
                std::ostringstream oss;
                oss << "failed to resize buffer for device " << device_id_
                    << " in worker process -> " << ret.second;
                return {false, oss.str()};
            }
        }

        const double* model = worker_shmem_manager_.model_task(func_id,
                                                               params,
                                                               n_params,
                                                               egrid,
                                                               n_out,
                                                               spec_num,
                                                               init_string,
                                                               input_model);

        const TaskStatus& ret = worker_execute_task(Task::EvaluateModel);

        if (ret.first) {
            memcpy(output, model, n_out * sizeof(double));
        }

        return ret;
    }

   private:
    int32_t device_id_;
    pid_t worker_pid_{-1};
    std::thread monitor_;
    config::worker::WorkerShmemManager worker_shmem_manager_;

    void start_worker_process()
    {
        static const std::string worker_path = utils::worker_executable_path();

        worker_pid_ = fork();
        if (worker_pid_ < 0) {
            std::ostringstream oss;
            oss << "failed to fork worker process for device" << device_id_
                << ": " << strerror(errno);
            throw std::runtime_error(oss.str());
        }

        if (worker_pid_ == 0) {
            std::string device_str = std::to_string(device_id_);
            execl(worker_path.c_str(),
                  worker_path.c_str(),
                  device_str.c_str(),
                  nullptr);

            std::ostringstream oss;
            oss << "exec failed for device " << device_str;
            perror(oss.str().c_str());
            std::_Exit(127);
        }

        // Wait until worker process enters worker loop
        worker_shmem_manager_.wait_for_task_end();
    }

    void start_worker_monitor()
    {
        monitor_ = std::thread([this]() {
            int status;
            while (worker_shmem_manager_.running()) {
                pid_t result = waitpid(worker_pid_, &status, WNOHANG);
                if (result != 0) {
                    std::ostringstream oss;
                    oss << "worker process (device " << device_id_ << ", pid "
                        << worker_pid_ << ")";

                    if (result == worker_pid_) {  // Child process has exited
                        if (WIFEXITED(status)) {
                            oss << " exited with code " << WEXITSTATUS(status);
                        } else if (WIFSIGNALED(status)) {
                            oss << " killed by signal " << WTERMSIG(status);
                        } else {
                            oss << " exited abnormally (status=" << status
                                << ")";
                        }
                    } else if (result == -1) {
                        oss << "waitpid error: " << strerror(errno);
                    } else {
                        oss << "unexpected waitpid result: " << result;
                    }

                    worker_shmem_manager_.running(false);
                    worker_shmem_manager_.success(false);
                    worker_shmem_manager_.message(oss.str());
                    worker_shmem_manager_.notify_task_end();
                    worker_pid_ = -1;
                }
                sleep(1);
            }
        });
    }

    void terminate_worker_process() noexcept
    {
        if (worker_pid_ <= 0) {
            return;
        }

        const auto process_str = "worker process (device " +
                                 std::to_string(device_id_) + ", pid " +
                                 std::to_string(worker_pid_) + ")";

        try {
            // Send SIGTERM to the worker process
            if (kill(worker_pid_, SIGTERM) != 0) {
                std::cerr << "failed to send SIGTERM to " << process_str
                          << ": " << strerror(errno) << std::endl;
            }

            // Wait for the worker process to exit with timeout
            int status;
            constexpr auto timeout = std::chrono::seconds(5);
            const auto start = std::chrono::steady_clock::now();

            bool worker_exited = false;
            while (std::chrono::steady_clock::now() - start < timeout) {
                const int ret = waitpid(worker_pid_, &status, WNOHANG);
                if (ret == worker_pid_) {
                    worker_exited = true;
                    break;
                } else if (ret == -1) {
                    if (errno == ECHILD) {
                        // The worker process has already exited
                        worker_exited = true;
                    } else {
                        std::cerr << process_str
                                  << " waitpid error: " << strerror(errno)
                                  << std::endl;
                    }
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }

            if (!worker_exited) {
                // Force kill the worker process after timeout
                std::cerr << "force killing " << process_str << std::endl;
                if (kill(worker_pid_, SIGKILL) == 0) {
                    // Wait for the zombie process to be cleaned up
                    waitpid(worker_pid_, nullptr, 0);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "exception while terminating " << process_str << ": "
                      << e.what() << std::endl;
        } catch (...) {
            std::cerr << "unknown error while terminating " << process_str
                      << std::endl;
        }

        worker_pid_ = -1;
    }

    void terminate_worker_monitor() noexcept
    {
        if (!monitor_.joinable()) {
            return;
        }

        const auto thread_str = "worker monitor thread (device " +
                                std::to_string(device_id_) + ")";

        try {
            // Use future and timeout to wait for the thread to end
            auto future =
                std::async(std::launch::async, [&]() { monitor_.join(); });

            if (future.wait_for(std::chrono::seconds(5)) ==
                std::future_status::timeout) {
                monitor_.detach();
                std::cerr << thread_str << " join timeout and detached"
                          << std::endl;
            } else {
                future.get();
            }
        } catch (const std::exception& e) {
            std::cerr << "exception while joining " << thread_str << ": "
                      << e.what() << std::endl;
            if (monitor_.joinable()) {
                monitor_.detach();
                std::cerr << thread_str << " detached" << std::endl;
            }
        } catch (...) {
            std::cerr << "unknown exception while joining " << thread_str
                      << std::endl;
        }
    }
};

class WorkerProcessPool
{
   public:
    // Constructor
    WorkerProcessPool() : xspec_config_manager_{getpid(), true}
    {
        // Initialize mutex for each XLA device
        const auto n_devices = utils::xla_device_number();
        mtx_.reserve(n_devices);
        pool_.reserve(n_devices);
        for (int32_t device_id = 0; device_id < n_devices; ++device_id) {
            mtx_[device_id] = std::make_unique<std::mutex>();
            pool_[device_id] = nullptr;
        }

        // Initialize worker process for device 0 and sync config to shmem
        const TaskStatus& ret = start_worker_process(0, false);
        if (!ret.first) {
            std::ostringstream oss;
            oss << "failed to start worker process pool -> " << ret.second;
            throw std::runtime_error(oss.str());
        }
        const TaskStatus& ret2 = xspec_config(Task::SyncConfigToShmem);
        if (!ret2.first) {
            std::ostringstream oss;
            oss << "failed to start worker process pool -> failed to sync "
                   "XSPEC config to shmem -> "
                << ret2.second;
            throw std::runtime_error(oss.str());
        }
    }

    // Destructor
    ~WorkerProcessPool() { pool_.clear(); }

    [[nodiscard]] int chatter() const noexcept
    {
        return xspec_config_manager_.chatter.level();
    }

    TaskStatus chatter(const int level) noexcept
    {
        backup_xspec_config();
        xspec_config_manager_.chatter.level(level);
        return xspec_config(Task::SyncChatterFromShmem);
    }

    [[nodiscard]] std::string abund() const noexcept
    {
        return xspec_config_manager_.abund.table();
    }

    TaskStatus abund(const std::string& table) noexcept
    {
        backup_xspec_config();
        xspec_config_manager_.abund.table(table);
        return xspec_config(Task::SyncAbundFromShmem);
    }

    TaskStatus abund_file(const std::string& file) noexcept
    {
        backup_xspec_config();
        xspec_config_manager_.abund.file(file);
        return xspec_config(Task::SyncAbundFromShmem);
    }

    [[nodiscard]] std::string xsect() const noexcept
    {
        return xspec_config_manager_.xsect.table();
    }

    TaskStatus xsect(const std::string& table) noexcept
    {
        backup_xspec_config();
        xspec_config_manager_.xsect.table(table);
        return xspec_config(Task::SyncXsectFromShmem);
    }

    [[nodiscard]] std::map<std::string, float> cosmo() const noexcept
    {
        return {
            {"H0", xspec_config_manager_.cosmo.H0()},
            {"q0", xspec_config_manager_.cosmo.q0()},
            {"lambda0", xspec_config_manager_.cosmo.lambda0()},
        };
    }

    TaskStatus cosmo(const float h0,
                     const float q0,
                     const float lambda0) noexcept
    {
        backup_xspec_config();
        xspec_config_manager_.cosmo.H0(h0);
        xspec_config_manager_.cosmo.q0(q0);
        xspec_config_manager_.cosmo.lambda0(lambda0);
        return xspec_config(Task::SyncCosmoFromShmem);
    }

    [[nodiscard]] std::string xspec_version() const noexcept
    {
        return xspec_config_manager_.xspec_version.version();
    }

    [[nodiscard]] std::string atomdb_version() const noexcept
    {
        return xspec_config_manager_.atomdb_version.version();
    }

    TaskStatus atomdb_version(const std::string& version) noexcept
    {
        backup_xspec_config();
        xspec_config_manager_.atomdb_version.version(version);
        return xspec_config(Task::SyncAtomDBVersionFromShmem);
    }

    [[nodiscard]] std::string spex_version() const noexcept
    {
        return xspec_config_manager_.spex_version.version();
    }

    TaskStatus spex_version(const std::string& version) noexcept
    {
        backup_xspec_config();
        xspec_config_manager_.spex_version.version(version);
        return xspec_config(Task::SyncSPEXVersionFromShmem);
    }

    [[nodiscard]] std::string nei_version() const noexcept
    {
        return xspec_config_manager_.nei_version.version();
    }

    TaskStatus nei_version(const std::string& version) noexcept
    {
        backup_xspec_config();
        xspec_config_manager_.nei_version.version(version);
        return xspec_config(Task::SyncNEIVersionFromShmem);
    }

    // Get single model string
    [[nodiscard]] std::string mstr(const std::string& key) const noexcept
    {
        return xspec_config_manager_.mstr.mstr(key);
    }

    // Get all model strings
    [[nodiscard]] config::xspec::MStrMap mstr() const noexcept
    {
        return xspec_config_manager_.mstr.mstr();
    }

    // Set single model string or multiple model strings
    TaskStatus mstr(const std::string& key, const std::string& value) noexcept
    {
        backup_xspec_config();
        xspec_config_manager_.mstr.mstr(key, value);
        return xspec_config(Task::SyncModelStringFromShmem);
    }

    TaskStatus mstr(const config::xspec::MStrMap& map) noexcept
    {
        backup_xspec_config();
        xspec_config_manager_.mstr.mstr(map);
        return xspec_config(Task::SyncModelStringFromShmem);
    }

    // Clear model string
    TaskStatus clear_mstr() noexcept
    {
        backup_xspec_config();
        xspec_config_manager_.mstr.clear();
        return xspec_config(Task::SyncModelStringFromShmem);
    }

    // Get XFLT entries for a given spectrum number
    [[nodiscard]] config::xspec::XFLTMap xflt(
        const int spec_num) const noexcept
    {
        return xspec_config_manager_.xflt.xflt(spec_num);
    }

    // Get all XFLT entries
    [[nodiscard]] config::xspec::XFLTMaps xflt() const noexcept
    {
        return xspec_config_manager_.xflt.xflt();
    }

    // Set XFLT entries for a given spectrum number
    TaskStatus xflt(const int spec_num,
                    const config::xspec::XFLTMap& map) noexcept
    {
        backup_xspec_config();
        xspec_config_manager_.xflt.xflt(spec_num, map);
        return xspec_config(Task::SyncXFLTFromShmem);
    }

    // Set multiple XFLT entries for multiple spectra
    TaskStatus xflt(const config::xspec::XFLTMaps& maps) noexcept
    {
        backup_xspec_config();
        xspec_config_manager_.xflt.xflt(maps);
        return xspec_config(Task::SyncXFLTFromShmem);
    }

    // Clear XFLT entries for a given spectrum number or all XFLT entries
    TaskStatus clear_xflt(const int spec_num) noexcept
    {
        backup_xspec_config();
        xspec_config_manager_.xflt.clear(spec_num);
        return xspec_config(Task::SyncXFLTFromShmem);
    }

    // Clear all XFLT entries
    TaskStatus clear_xflt() noexcept
    {
        backup_xspec_config();
        xspec_config_manager_.xflt.clear();
        return xspec_config(Task::SyncXFLTFromShmem);
    }

    // Sync XFLT entries to XSPEC in current process
    void sync_xflt_to_xspec() { xspec_config_manager_.xflt.sync_from_shmem(); }

    TaskStatus evaluate_model(const int32_t device_id,
                              const uint32_t func_id,
                              const double* params,
                              const uint32_t n_params,
                              const double* egrid,
                              const uint32_t n_out,
                              double* output,
                              const int spec_num,
                              const std::string& init_string,
                              const double* input_model = nullptr)
    {
        // Lock the mutex for the given device
        const auto& mtx = mtx_.find(device_id);
        if (mtx == mtx_.end()) {
            std::ostringstream oss;
            oss << "model evaluation failed -> invalid device id ("
                << device_id << ")";
            return {false, oss.str()};
        }
        std::lock_guard<std::mutex> lock(*mtx->second);

        // Initialize the worker process if it is not initialized
        if (pool_.at(device_id) == nullptr) {
            const TaskStatus& ret = start_worker_process(device_id);
            if (!ret.first) {
                std::ostringstream oss;
                oss << "model evaluation failed -> " << ret.second;
                return {false, oss.str()};
            }
        }

        // Execute model task
        return pool_.at(device_id)->worker_execute_model_task(func_id,
                                                              params,
                                                              n_params,
                                                              egrid,
                                                              n_out,
                                                              output,
                                                              spec_num,
                                                              init_string,
                                                              input_model);
    }

   private:
    std::unordered_map<int32_t, std::unique_ptr<std::mutex>> mtx_;
    std::unordered_map<int32_t, std::unique_ptr<WorkerProcessManager>> pool_;
    config::xspec::XspecConfigManager xspec_config_manager_;
    std::unique_ptr<config::xspec::XspecConfig> xspec_config_backup_;

    std::pair<bool, std::string> start_worker_process(
        const int32_t device_id,
        const bool sync_config_from_shmem = true) noexcept
    {
        if (pool_.at(device_id) != nullptr) {
            return {true, ""};
        }

        std::unique_ptr<WorkerProcessManager> wpm;
        try {
            wpm = std::make_unique<WorkerProcessManager>(device_id);
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "failed to start worker process for device " << device_id
                << " -> " << e.what();
            return {false, oss.str()};
        } catch (...) {
            std::ostringstream oss;
            oss << "failed to start worker process for device " << device_id
                << " -> unknown error";
            return {false, oss.str()};
        }

        auto ret = wpm->worker_execute_task(Task::InitializeXSPEC);
        if (!ret.first) {
            std::ostringstream oss;
            oss << "failed to start worker process for device " << device_id
                << " -> failed to initialize XSPEC model library in worker "
                   "process -> "
                << ret.second;
            return {false, oss.str()};
        }
        if (sync_config_from_shmem) {
            ret = wpm->worker_execute_task(Task::SyncConfigFromShmem);
            if (!ret.first) {
                std::ostringstream oss;
                oss << "failed to start worker process for device "
                    << device_id
                    << " -> failed to sync XSPEC config from shared memory in "
                       "worker process -> "
                    << ret.second;
                return {false, oss.str()};
            }
        }
        pool_[device_id] = std::move(wpm);
        return {true, ""};
    }

    [[nodiscard]] std::pair<bool, std::string> xspec_config(
        const Task task) const noexcept
    {
        bool success = true;
        std::string message;

        for (const auto& [device_id, worker_manager] : pool_) {
            if (worker_manager == nullptr) {
                continue;
            }
            const auto& ret = worker_manager->worker_execute_task(task);
            if (!ret.first) {
                success = false;
                message = ret.second;
                break;
            }
        }

        // Restore xspec config if failed
        if (!success && task != Task::SyncConfigFromShmem) {
            restore_xspec_config();
        }

        return {success, message};
    }

    void backup_xspec_config() noexcept
    {
        xspec_config_backup_ = std::make_unique<config::xspec::XspecConfig>(
            xspec_config_manager_.config());
    }

    void restore_xspec_config() const noexcept
    {
        xspec_config_manager_.restore(*xspec_config_backup_);
        const auto& ret = xspec_config(Task::SyncConfigFromShmem);
        if (!ret.first) {
            std::cerr << "failed to restore xspec config -> " << ret.second
                      << std::endl;
        }
    }
};

inline WorkerProcessPool& worker_process_pool()
{
    static WorkerProcessPool pool;
    return pool;
}
}  // namespace xspex::multiprocessing

#endif  // XSPEX_MULTIPROCESSING_HPP_
