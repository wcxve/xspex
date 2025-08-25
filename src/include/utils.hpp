#ifndef XSPEX_UTILS_HPP_
#define XSPEX_UTILS_HPP_

#include <dlfcn.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>

// clang-format off
#include <xla/ffi/api/ffi.h>
// clang-format on

#define STATIC_ASSERT_SHM_ELIGIBLE(T)                                   \
    static_assert(std::is_trivially_copyable_v<T>,                      \
                  #T " must be trivially copyable to be SHM eligible"); \
    static_assert(std::is_standard_layout_v<T>,                         \
                  #T " must be standard layout to be SHM eligible")

namespace xspex::utils
{
namespace ffi = xla::ffi;

inline void copy_string(const std::string& src,
                        char* dst,
                        uint32_t size,
                        bool allow_empty = false)
{
    if (src.empty()) {
        if (!allow_empty) {
            throw std::invalid_argument("empty string");
        }
        std::memset(dst, 0, size);
        return;
    }

    if (src.size() > size - 1) {
        std::ostringstream oss;
        oss << "string exceeds the length limit (" << size - 1 << "): " << src;
        throw std::length_error(oss.str());
    }

    std::strncpy(dst, src.c_str(), size);
    dst[size - 1] = '\0';
}

// Convert double to string without precision loss
inline std::string double_to_string(double value)
{
    char buf[32];
    auto [ptr, ec] = std::to_chars(buf,
                                   buf + sizeof(buf),
                                   value,
                                   std::chars_format::general,
                                   std::numeric_limits<double>::max_digits10);
    if (ec != std::errc{}) {
        throw std::runtime_error("XFLT: double to str conversion failed");
    }
    return {buf, ptr};
}

inline std::string to_lower_case(const std::string& input)
{
    std::string output(input.size(), '\0');
    std::transform(
        input.begin(), input.end(), output.begin(), [](unsigned char ch) {
            return std::tolower(ch);
        });
    return output;
}

inline std::string map_to_json(const std::map<std::string, std::string>& map)
{
    std::ostringstream oss;
    oss << "{";

    bool first = true;
    for (const auto& pair : map) {
        if (!first) {
            oss << ", ";
        }
        first = false;

        // Add key-value pairs, surround keys and values with double quotes
        oss << "\"" << pair.first << "\": \"" << pair.second << "\"";
    }

    oss << "}";
    return oss.str();
}

inline std::string map_to_json(
    const std::map<int, std::map<std::string, std::string>>& map)
{
    std::ostringstream oss;
    oss << "{";

    bool first = true;
    for (const auto& outerPair : map) {
        if (!first) {
            oss << ", ";
        }
        first = false;

        // Outer key (int type, no quotes)
        oss << "\"" << outerPair.first << "\": ";

        // Convert inner map to JSON object
        oss << map_to_json(outerPair.second);
    }

    oss << "}";
    return oss.str();
}

inline uint32_t next_power_of_two(uint32_t n) noexcept
{
    if (n <= 1) return 2;

    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

inline std::string shape_string_of(const ffi::AnyBuffer& buffer) noexcept
{
    std::ostringstream ss;
    ss << "(";
    const auto& dims = buffer.dimensions();
    for (size_t i = 0; i < dims.size(); ++i) {
        ss << dims[i];
        if (i < dims.size() - 1) {
            ss << ", ";
        }
    }
    if (dims.size() == 1) {
        ss << ",";
    }
    ss << ")";
    return ss.str();
}

inline void check_input(const std::string& model_name,
                        const uint32_t n_params,
                        const ffi::AnyBuffer& params,
                        const ffi::AnyBuffer& egrid,
                        const ffi::AnyBuffer& spec_num,
                        ffi::Result<ffi::AnyBuffer>& model)
{
    const auto& dim_p = params.dimensions();
    const auto& dim_e = egrid.dimensions();
    const auto& dim_sn = spec_num.dimensions();
    const auto& dim_m = model->dimensions();
    const auto n_p = dim_p.back();
    const auto n_e = dim_e.back();
    const auto n_m = dim_m.back();

    std::ostringstream oss;

    if (dim_p.size() > 1) {
        oss << "batching over parameters " << utils::shape_string_of(params)
            << " is not supported yet";
        throw std::runtime_error(oss.str());
    }

    if (dim_e.size() > 1) {
        oss << "batching over egrid " << utils::shape_string_of(egrid)
            << " is not supported yet";
        throw std::runtime_error(oss.str());
    }

    if (dim_sn.size() >= 1) {
        oss << "batching over spectrum number "
            << utils::shape_string_of(spec_num) << " is not supported yet";
        throw std::runtime_error(oss.str());
    }

    if (dim_m.size() > 1) {
        oss << "batching over model output " << utils::shape_string_of(*model)
            << " is not supported yet";
        throw std::runtime_error(oss.str());
    }

    if (n_p != n_params) {
        oss << model_name << " model expected " << n_params
            << (n_params != 1 ? " parameters" : " parameter") << ", but got "
            << n_p;
        throw std::runtime_error(oss.str());
    }

    if (n_e < 2) {
        oss << model_name << " model expected egrid of length >= 2, but got "
            << n_e;
        throw std::runtime_error(oss.str());
    }

    if (n_m != n_e - 1) {
        oss << model_name << " model got inconsistent size of egrid (" << n_e
            << ") and model output (" << n_m << ")";
        throw std::runtime_error(oss.str());
    }
}

inline void check_input(const std::string& model_name,
                        const uint32_t n_params,
                        const ffi::AnyBuffer& params,
                        const ffi::AnyBuffer& egrid,
                        const ffi::AnyBuffer& spec_num,
                        ffi::Result<ffi::AnyBuffer>& model,
                        const ffi::AnyBuffer& input_model)
{
    check_input(model_name, n_params, params, egrid, spec_num, model);
    const auto& dim_e = egrid.dimensions();
    const auto& dim_im = input_model.dimensions();
    const auto n_e = dim_e.back();
    const auto n_im = dim_im.back();
    if (n_im != n_e - 1) {
        std::ostringstream oss;
        oss << model_name << " model got inconsistent size of egrid (" << n_e
            << ") and input model (" << n_im << ")";
        throw std::runtime_error(oss.str());
    }
}

inline std::string worker_executable_path()
{
    Dl_info info;
    if (dladdr((void*)&worker_executable_path, &info) && info.dli_fname) {
        std::filesystem::path lib_path = info.dli_fname;
        std::filesystem::path worker_path =
            lib_path.parent_path().parent_path() / "bin" / "worker";
        if (!std::filesystem::exists(worker_path)) {
            throw std::runtime_error("xspex worker executable not found");
        }
        return worker_path.string();
    }
    throw std::runtime_error("xspex shared library not found");
}

inline uint32_t xla_device_number() noexcept
{
    static const uint32_t result = []() -> uint32_t {
        const char* xla_flags_p = std::getenv("XLA_FLAGS");
        if (xla_flags_p != nullptr) {
            std::string xla_flags(xla_flags_p);
            std::regex re("--xla_force_host_platform_device_count=(\\d+)");
            std::smatch match;
            if (std::regex_search(xla_flags, match, re)) {
                try {
                    uint32_t n = std::stoi(match[1].str());
                    if (n > 0) return n;
                } catch (...) {
                    std::cerr << "failed to parse device number from XLA_FLAGS"
                              << std::endl;
                }
            }
        }
        return 1;
    }();
    return result;
}
}  // namespace xspex::utils

#endif  // XSPEX_UTILS_HPP_
