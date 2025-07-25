#ifndef XSPEX_INTERFACE_HPP_
#define XSPEX_INTERFACE_HPP_

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "config.hpp"
#include "multiprocessing.hpp"

namespace xspex::interface
{
using TaskStatus = std::pair<bool, std::string>;

static auto& pool() { return multiprocessing::worker_process_pool(); }

static void throw_if_failed(const std::string& task_name,
                            const TaskStatus& status)
{
    if (!status.first) {
        std::ostringstream oss;
        oss << "failed to " << task_name << " -> " << status.second;
        throw std::invalid_argument(oss.str());
    }
}

inline auto get_chatter() noexcept { return pool().chatter(); }

inline void set_chatter(const int level)
{
    throw_if_failed("set chatter level", pool().chatter(level));
}

inline auto get_abund() noexcept { return pool().abund(); }

inline void set_abund(const std::string& table)
{
    throw_if_failed("set abundance table", pool().abund(table));
}

inline void set_abund_file(const std::string& file)
{
    throw_if_failed("set abundance table", pool().abund_file(file));
}

inline auto get_xsect() noexcept { return pool().xsect(); }

inline void set_xsect(const std::string& table)
{
    throw_if_failed("set photo-electric cross-section table",
                    pool().xsect(table));
}

inline auto get_cosmo() noexcept { return pool().cosmo(); }

inline void set_cosmo(const float h0, const float q0, const float lambda0)
{
    throw_if_failed("set cosmological parameters",
                    pool().cosmo(h0, q0, lambda0));
}

inline auto xspec_version() noexcept { return pool().xspec_version(); }

inline auto atomdb_version() noexcept { return pool().atomdb_version(); }

inline auto get_mstr(const std::string& key) noexcept
{
    return pool().mstr(key);
}

inline auto get_mstrs() noexcept { return pool().mstr(); }

inline void set_mstr(const std::string& key, const std::string& value)
{
    throw_if_failed("set model string", pool().mstr(key, value));
}

inline void set_mstrs(const config::xspec::MStrMap& map)
{
    throw_if_failed("set model string", pool().mstr(map));
}

inline void clear_mstr()
{
    throw_if_failed("clear model string", pool().clear_mstr());
}

inline auto get_xflt(const int spec_num) noexcept
{
    return pool().xflt(spec_num);
}

inline auto get_xflts() noexcept { return pool().xflt(); }

inline void set_xflt(const int spec_num, const config::xspec::XFLTMap& map)
{
    throw_if_failed("set XFLT", pool().xflt(spec_num, map));
}

inline void set_xflts(const config::xspec::XFLTMaps& maps)
{
    throw_if_failed("set XFLT", pool().xflt(maps));
}

inline void clear_xflt(const int spec_num)
{
    throw_if_failed("clear XFLT", pool().clear_xflt(spec_num));
}

inline void clear_all_xflt()
{
    throw_if_failed("clear all XFLT", pool().clear_xflt());
}

inline void sync_xflt_to_xspec() { pool().sync_xflt_to_xspec(); }

inline TaskStatus evaluate_model(const int32_t device_id,
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
    return pool().evaluate_model(device_id,
                                 func_id,
                                 params,
                                 n_params,
                                 egrid,
                                 n_out,
                                 output,
                                 spec_num,
                                 init_string,
                                 input_model);
}
}  // namespace xspex::interface

#endif  // XSPEX_INTERFACE_HPP_
