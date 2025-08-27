#ifndef XSPEX_XSPEC_WRAPPER_HPP_
#define XSPEX_XSPEC_WRAPPER_HPP_

#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <string_view>

// clang-format off
#include <nanobind/nanobind.h>
#include <xla/ffi/api/c_api.h>
#include <xla/ffi/api/ffi.h>
// clang-format on

#include "interface.hpp"
#include "mdef.hpp"
#include "utils.hpp"

#define DEFINE_XSPEC_MODEL_HANDLER(func_name, func_id, n_param)              \
    ffi::Error func_name##_wrapper(const ffi::AnyBuffer& params,             \
                                   const ffi::AnyBuffer& egrid,              \
                                   const ffi::AnyBuffer& spec_num,           \
                                   ffi::Result<ffi::AnyBuffer> model,        \
                                   const std::string_view& init_string,      \
                                   const bool skip_check,                    \
                                   const int32_t device_ordinal)             \
    {                                                                        \
        if (!skip_check)                                                     \
            xspex::utils::check_input(                                       \
                #func_name, n_param, params, egrid, spec_num, model);        \
                                                                             \
        return xspex::wrapper::model_wrapper(device_ordinal,                 \
                                             func_id,                        \
                                             params,                         \
                                             egrid,                          \
                                             spec_num,                       \
                                             model,                          \
                                             init_string);                   \
    }                                                                        \
    XLA_FFI_DEFINE_HANDLER_SYMBOL(func_name##_handler,                       \
                                  func_name##_wrapper,                       \
                                  ffi::Ffi::Bind()                           \
                                      .Arg<ffi::AnyBuffer>()                 \
                                      .Arg<ffi::AnyBuffer>()                 \
                                      .Arg<ffi::AnyBuffer>()                 \
                                      .Ret<ffi::AnyBuffer>()                 \
                                      .Attr<std::string_view>("init_string") \
                                      .Attr<bool>("skip_check")              \
                                      .Ctx<ffi::DeviceOrdinal>());

#define DEFINE_XSPEC_CON_MODEL_HANDLER(func_name, func_id, n_param)          \
    ffi::Error func_name##_wrapper(const ffi::AnyBuffer& params,             \
                                   const ffi::AnyBuffer& egrid,              \
                                   const ffi::AnyBuffer& input_model,        \
                                   const ffi::AnyBuffer& spec_num,           \
                                   ffi::Result<ffi::AnyBuffer> model,        \
                                   const std::string_view& init_string,      \
                                   const bool skip_check,                    \
                                   const int32_t device_ordinal)             \
    {                                                                        \
        if (!skip_check)                                                     \
            xspex::utils::check_input(#func_name,                            \
                                      n_param,                               \
                                      params,                                \
                                      egrid,                                 \
                                      spec_num,                              \
                                      model,                                 \
                                      input_model);                          \
                                                                             \
        return xspex::wrapper::con_model_wrapper(device_ordinal,             \
                                                 func_id,                    \
                                                 params,                     \
                                                 egrid,                      \
                                                 input_model,                \
                                                 spec_num,                   \
                                                 model,                      \
                                                 init_string);               \
    }                                                                        \
    XLA_FFI_DEFINE_HANDLER_SYMBOL(func_name##_handler,                       \
                                  func_name##_wrapper,                       \
                                  ffi::Ffi::Bind()                           \
                                      .Arg<ffi::AnyBuffer>()                 \
                                      .Arg<ffi::AnyBuffer>()                 \
                                      .Arg<ffi::AnyBuffer>()                 \
                                      .Arg<ffi::AnyBuffer>()                 \
                                      .Ret<ffi::AnyBuffer>()                 \
                                      .Attr<std::string_view>("init_string") \
                                      .Attr<bool>("skip_check")              \
                                      .Ctx<ffi::DeviceOrdinal>());

namespace nb = nanobind;
namespace ffi = xla::ffi;

namespace xspex::wrapper
{
inline ffi::Error model_wrapper(const int32_t device_id,
                                const uint32_t func_id,
                                const ffi::AnyBuffer& params,
                                const ffi::AnyBuffer& egrid,
                                const ffi::AnyBuffer& spec_num,
                                ffi::Result<ffi::AnyBuffer>& model,
                                const std::string_view& init_string)
{
    const auto& status =
        interface::evaluate_model(device_id,
                                  func_id,
                                  params.typed_data<double>(),
                                  params.dimensions().back(),
                                  egrid.typed_data<double>(),
                                  model->dimensions().back(),
                                  model->typed_data<double>(),
                                  spec_num.typed_data<int>()[0],
                                  init_string.data());
    if (!status.first) {
        return {ffi::ErrorCode::kInternal, status.second};
    }
    return ffi::Error::Success();
}

inline ffi::Error con_model_wrapper(const int32_t device_id,
                                    const uint32_t func_id,
                                    const ffi::AnyBuffer& params,
                                    const ffi::AnyBuffer& egrid,
                                    const ffi::AnyBuffer& input_model,
                                    const ffi::AnyBuffer& spec_num,
                                    ffi::Result<ffi::AnyBuffer>& model,
                                    const std::string_view& init_string)
{
    const auto& status =
        interface::evaluate_model(device_id,
                                  func_id,
                                  params.typed_data<double>(),
                                  params.dimensions().back(),
                                  egrid.typed_data<double>(),
                                  model->dimensions().back(),
                                  model->typed_data<double>(),
                                  spec_num.typed_data<int>()[0],
                                  init_string.data(),
                                  input_model.typed_data<double>());
    if (!status.first) {
        return {ffi::ErrorCode::kInternal, status.second};
    }
    return ffi::Error::Success();
}

template <typename T>
inline nb::capsule encapsulate_ffi_handler(T* fn)
{
    static_assert(std::is_invocable_r_v<XLA_FFI_Error*, T, XLA_FFI_CallFrame*>,
                  "Encapsulated function must be an XLA FFI handler");
    return {reinterpret_cast<void*>(fn)};
}

namespace
{
// Auto-generate model ID enum, matching the order of functions in xspec.hpp
enum class XspecModelId : uint32_t {
#define ENTRY(name, n_params) name,
    XSPEC_MODELS XSPEC_CON_MODELS
#undef ENTRY
};

// Auto-generate handler definitions using macros
#define ENTRY(name, n_params)   \
    DEFINE_XSPEC_MODEL_HANDLER( \
        name, static_cast<uint32_t>(XspecModelId::name), n_params);
XSPEC_MODELS
#undef ENTRY

#define ENTRY(name, n_params)       \
    DEFINE_XSPEC_CON_MODEL_HANDLER( \
        name, static_cast<uint32_t>(XspecModelId::name), n_params);
XSPEC_CON_MODELS
#undef ENTRY
}  // namespace

inline nb::capsule get_xla_ffi_handler(const std::string& name)
{
    static std::map<std::string, XLA_FFI_Handler*> handlers = {
#define ENTRY(name, n_params) {#name, name##_handler},
        XSPEC_MODELS XSPEC_CON_MODELS
#undef ENTRY
    };
    if (handlers.find(name) == handlers.end()) {
        throw std::runtime_error("XLA FFI handler not found for " + name);
    }
    return encapsulate_ffi_handler(handlers.at(name));
}
}  // namespace xspex::wrapper

#endif  // XSPEX_XSPEC_WRAPPER_HPP_
