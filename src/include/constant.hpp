#ifndef XSPEX_CONSTANT_HPP_
#define XSPEX_CONSTANT_HPP_

#include <sys/types.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>

// ABUND and XSECT table name length: 4 characters + 1 null terminator
#define XSPEX_TABLE_LENGTH (5)

// File path length: 1023 characters + 1 null terminator
#ifndef XSPEX_PATH_LENGTH
#define XSPEX_PATH_LENGTH (4096)
#endif

// Version string length: 127 characters + 1 null terminator
#ifndef XSPEX_VERSION_LENGTH
#define XSPEX_VERSION_LENGTH (128)
#endif

// See XSPEC xset manual for details
// Model string length: 127 characters + 1 null terminator
#ifndef XSPEX_MSTR_LENGTH
#define XSPEX_MSTR_LENGTH (128)
#endif

// Number of model string entries: 128
#ifndef XSPEX_MSTR_DB_SIZE
#define XSPEX_MSTR_DB_SIZE (128)
#endif

// XFLT key length: 127 characters + 1 null terminator
#ifndef XSPEX_XFLT_KEY_LENGTH
#define XSPEX_XFLT_KEY_LENGTH (128)
#endif

// Number of XFLT entries: 1024
#ifndef XSPEX_XFLT_DB_SIZE
#define XSPEX_XFLT_DB_SIZE (1024)
#endif

// Config content length: 4095 characters + 1 null terminator
#ifndef XSPEX_CONFIG_CONTENT_LENGTH
#define XSPEX_CONFIG_CONTENT_LENGTH (4096)
#endif

// Initial shared memory size is the next power of 2 of
// params(128) + egrid(4097) + flux(4096) + flux_error(4096)
#ifndef XSPEX_SHARED_MEMORY_SIZE_INIT
#define XSPEX_SHARED_MEMORY_SIZE_INIT (16384)
#endif

namespace xspex::constant
{
constexpr uint32_t table_length{XSPEX_TABLE_LENGTH};
constexpr uint32_t mstr_length{XSPEX_MSTR_LENGTH};
constexpr uint32_t xflt_key_length{XSPEX_XFLT_KEY_LENGTH};
constexpr uint32_t mstr_db_size{XSPEX_MSTR_DB_SIZE};
constexpr uint32_t xflt_db_size{XSPEX_XFLT_DB_SIZE};
constexpr uint32_t path_length{XSPEX_PATH_LENGTH};
constexpr uint32_t version_length{XSPEX_VERSION_LENGTH};
constexpr uint32_t config_content_length{XSPEX_CONFIG_CONTENT_LENGTH};
constexpr uint32_t shared_memory_size_init{XSPEX_SHARED_MEMORY_SIZE_INIT};
// Must remain a power of two for alignment/fragmentation assumptions.
static_assert((shared_memory_size_init & (shared_memory_size_init - 1U)) == 0U,
              "shared_memory_size_init must be a power of two");
constexpr std::string_view shm_prefix{"/xspex"};

inline std::string shm_name_of_xspec_config(pid_t pid)
{
    std::ostringstream oss;
    oss << shm_prefix << "_p" << pid << "_config_xspec";
    return oss.str();
}

inline std::string shm_name_of_worker_config(pid_t pid, int32_t device_id)
{
    std::ostringstream oss;
    oss << shm_prefix << "_p" << pid << "_config_d" << device_id;
    return oss.str();
}

inline std::string shm_name_of_worker_buffer(pid_t pid, int32_t device_id)
{
    std::ostringstream oss;
    oss << shm_prefix << "_p" << pid << "_buffer_d" << device_id;
    return oss.str();
}
}  // namespace xspex::constant

#endif  // XSPEX_CONSTANT_HPP_
