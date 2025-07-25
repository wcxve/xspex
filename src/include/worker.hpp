#ifndef XSPEX_WORKER_HPP_
#define XSPEX_WORKER_HPP_

#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <exception>
#include <stdexcept>

#include "config.hpp"
#include "xspec.hpp"

namespace xspex::worker
{
using Task = xspex::config::worker::Task;

class Worker
{
   public:
    Worker(pid_t parent_pid, int32_t device_id)
        : xspec_config_manager_{parent_pid, false},
          worker_shmem_manager_{parent_pid, device_id, false}
    {
    }

    void run_worker_loop()
    {
        worker_shmem_manager_.running(true);
        while (worker_shmem_manager_.running()) {
            // Signal the end of initialization or task
            worker_shmem_manager_.notify_task_end();
            // Wait for the main process to send task
            worker_shmem_manager_.wait_for_task_start();
            // Execute task
            try {
                execute_task();
                worker_shmem_manager_.success(true);
            } catch (const std::exception& e) {
                worker_shmem_manager_.success(false);
                worker_shmem_manager_.message(e.what());
            } catch (...) {
                worker_shmem_manager_.success(false);
                worker_shmem_manager_.message("unknown error");
            }
        }
        worker_shmem_manager_.running(false);
    }

   private:
    config::xspec::XspecConfigManager xspec_config_manager_;
    config::worker::WorkerShmemManager worker_shmem_manager_;

    void execute_task()
    {
        switch (worker_shmem_manager_.task()) {
            case Task::EvaluateModel: {
                auto [func_id,
                      egrid,
                      n_flux,
                      params,
                      spec_num,
                      model,
                      model_error,
                      init_string] = worker_shmem_manager_.model_task();
                xspec::functions[func_id](egrid,
                                          n_flux,
                                          params,
                                          spec_num,
                                          model,
                                          model_error,
                                          init_string);
                return;
            }
            case Task::InitializeXSPEC:
                xspec::initialize_xspec_model_library();
                return;
            case Task::SyncConfigToShmem:
                xspec_config_manager_.sync_to_shmem();
                return;
            case Task::SyncConfigFromShmem:
                xspec_config_manager_.sync_from_shmem();
                return;
            case Task::ResizeBuffer:
                worker_shmem_manager_.resize_buffer();
                return;
            case Task::SyncChatterFromShmem:
                xspec_config_manager_.chatter.sync_from_shmem();
                return;
            case Task::SyncAbundFromShmem:
                xspec_config_manager_.abund.sync_from_shmem();
                return;
            case Task::SyncXsectFromShmem:
                xspec_config_manager_.xsect.sync_from_shmem();
                return;
            case Task::SyncCosmoFromShmem:
                xspec_config_manager_.cosmo.sync_from_shmem();
                return;
            case Task::SyncAtomDBVersionFromShmem:
                xspec_config_manager_.atomdb_version.sync_from_shmem();
                return;
            case Task::SyncSPEXVersionFromShmem:
                xspec_config_manager_.spex_version.sync_from_shmem();
                return;
            case Task::SyncNEIVersionFromShmem:
                xspec_config_manager_.nei_version.sync_from_shmem();
                return;
            case Task::SyncModelStringFromShmem:
                xspec_config_manager_.mstr.sync_from_shmem();
                return;
            case Task::SyncXFLTFromShmem:
                xspec_config_manager_.xflt.sync_from_shmem();
                return;
            default:
                throw std::runtime_error("unknown task");
        }
    }
};
}  // namespace xspex::worker

#endif  // XSPEX_WORKER_HPP_
