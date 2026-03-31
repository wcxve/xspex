#ifndef XSPEX_XSPEC_DBVERSION_HPP_
#define XSPEX_XSPEC_DBVERSION_HPP_

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

// clang-format off
#include <XSFunctions/Utilities/FunctionUtility.h>
#include <XSUtil/Utils/XSutility.h>
// clang-format on

#include "utils.hpp"

namespace xspex::xspec
{
struct DatabaseVersionSpec {
    const char* setting_key;
};

inline constexpr DatabaseVersionSpec atomdb_version_spec{
    "ATOMDB_VERSION",
};

inline constexpr DatabaseVersionSpec spex_version_spec{
    "SPEX_VERSION",
};

inline constexpr DatabaseVersionSpec nei_version_spec{
    "NEI_VERSION",
};

inline std::string join_path(std::string base, const std::string& name)
{
    if (!base.empty() && base.back() != '/') {
        base.push_back('/');
    }
    return base + name;
}

inline std::string default_database_version(const DatabaseVersionSpec& spec)
{
    const auto settings = XSutility::readSettingsFile(
        join_path(FunctionUtility::managerPath(), "Xspec.init"));
    const auto it = settings.find(spec.setting_key);
    if (it != settings.end() && !it->second.empty() &&
        utils::to_lower_case(it->second) != "latest") {
        return it->second;
    }
    return {};
}

inline std::string resolve_database_version(const std::string& version,
                                            const DatabaseVersionSpec& spec)
{
    if (utils::to_lower_case(version) != "latest") {
        return version;
    }

    const auto latest_path =
        join_path(FunctionUtility::modelDataPath(), "latest.txt");
    std::ifstream latest_file(latest_path);
    if (!latest_file.is_open()) {
        const auto default_version = default_database_version(spec);
        if (!default_version.empty()) {
            return default_version;
        }
        std::ostringstream oss;
        oss << "failed to resolve latest value for " << spec.setting_key
            << ": \"" << latest_path
            << "\" is missing and Xspec.init does not provide a concrete "
               "version";
        throw std::runtime_error(oss.str());
    }

    const auto expected_key = std::string(spec.setting_key) + ":";
    std::string key;
    std::string resolved;
    while (latest_file >> key >> resolved) {
        if (key == expected_key) {
            return resolved;
        }
    }

    std::ostringstream oss;
    oss << "failed to resolve latest value for " << spec.setting_key
        << " from \"" << latest_path << "\"";
    throw std::runtime_error(oss.str());
}

inline void normalize_database_versions()
{
    const auto atomdb_version =
        resolve_database_version(FunctionUtility::atomdbVersion(),
                                 atomdb_version_spec);
    if (atomdb_version != FunctionUtility::atomdbVersion()) {
        FunctionUtility::atomdbVersion(atomdb_version);
    }

    const auto spex_version =
        resolve_database_version(FunctionUtility::spexVersion(),
                                 spex_version_spec);
    if (spex_version != FunctionUtility::spexVersion()) {
        FunctionUtility::spexVersion(spex_version);
    }

    const auto nei_version =
        resolve_database_version(FunctionUtility::neiVersion(),
                                 nei_version_spec);
    if (nei_version != FunctionUtility::neiVersion()) {
        FunctionUtility::neiVersion(nei_version);
    }
}
}  // namespace xspex::xspec

#endif  // XSPEX_XSPEC_DBVERSION_HPP_
