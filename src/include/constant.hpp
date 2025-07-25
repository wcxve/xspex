#ifndef XSPEX_CONSTANT_HPP_
#define XSPEX_CONSTANT_HPP_

#include <sys/types.h>

#include <cstdint>
#include <sstream>
#include <string>

// ABUND and XSECT table name length: 4 characters + 1 null terminator
#define TABLE_LENGTH (5)

// File path length: 1023 characters + 1 null terminator
#ifndef PATH_LENGTH
#define PATH_LENGTH (1024)
#endif

// Version string length: 127 characters + 1 null terminator
#ifndef VERSION_LENGTH
#define VERSION_LENGTH (128)
#endif

// See XSPEC xset manual for details
// Model string length: 127 characters + 1 null terminator
#ifndef MSTR_LENGTH
#define MSTR_LENGTH (128)
#endif

// Number of model string entries: 128
#ifndef MSTR_DB_SIZE
#define MSTR_DB_SIZE (128)
#endif

// XFLT key length: 127 characters + 1 null terminator
#ifndef XFLT_KEY_LENGTH
#define XFLT_KEY_LENGTH (128)
#endif

// Number of XFLT entries: 1024
#ifndef XFLT_DB_SIZE
#define XFLT_DB_SIZE (1024)
#endif

// Config content length: 4095 characters + 1 null terminator
#ifndef CONFIG_CONTENT_LENGTH
#define CONFIG_CONTENT_LENGTH (4096)
#endif

// Initial shared memory size is the next power of 2 of
// params(128) + egrid(4097) + flux(4096) + flux_error(4096)
#ifndef SHARED_MEMORY_SIZE_INIT
#define SHARED_MEMORY_SIZE_INIT (16384)
#endif

namespace xspex::constant
{
constexpr uint32_t table_length{TABLE_LENGTH};
constexpr uint32_t mstr_length{MSTR_LENGTH};
constexpr uint32_t xflt_key_length{XFLT_KEY_LENGTH};
constexpr uint32_t mstr_db_size{MSTR_DB_SIZE};
constexpr uint32_t xflt_db_size{XFLT_DB_SIZE};
constexpr uint32_t path_length{PATH_LENGTH};
constexpr uint32_t version_length{VERSION_LENGTH};
constexpr uint32_t config_content_length{CONFIG_CONTENT_LENGTH};
constexpr uint32_t shared_memory_size_init{SHARED_MEMORY_SIZE_INIT};

inline std::string shm_name_of_xspec_config(pid_t pid)
{
    std::ostringstream oss;
    oss << "/xspex" << "_p" << pid << "_config_xspec";
    return oss.str();
}

inline std::string shm_name_of_worker_config(pid_t pid, int32_t device_id)
{
    std::ostringstream oss;
    oss << "/xspex" << "_p" << pid << "_config_d" << device_id;
    return oss.str();
}

inline std::string shm_name_of_worker_buffer(pid_t pid, int32_t device_id)
{
    std::ostringstream oss;
    oss << "/xspex" << "_p" << pid << "_buffer_d" << device_id;
    return oss.str();
}
}  // namespace xspex::constant

#endif  // XSPEX_CONSTANT_HPP_
