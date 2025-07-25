#ifndef XSPEX_CONFIG_HPP_
#define XSPEX_CONFIG_HPP_

#include <pthread.h>
#include <sys/mman.h>
#include <sys/types.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

// clang-format off
#include <XSFunctions/Utilities/FunctionUtility.h>
#include <XSFunctions/Utilities/xsFortran.h>
#include <XSUtil/Utils/XSutility.h>
// clang-format on

#include "constant.hpp"
#include "shmem.hpp"
#include "utils.hpp"

namespace xspex::config::xspec
{
struct XspecConfig {
    template <size_t N>
    using String = char[N];

    struct Chatter {
        int level;  // terminal chatter level
    } chatter;

    struct Abund {
        String<constant::table_length> table;
        String<constant::path_length> file;
    } abund;  // relative abundance table

    struct Xsect {
        String<constant::table_length> table;
    } xsect;  // photoelectric cross-section table

    struct Cosmo {
        float H0;       // Hubble constant
        float q0;       // deceleration parameter
        float lambda0;  // cosmological constant
    } cosmo;

    struct Version {
        String<constant::version_length> version;
    } xspec_version, atomdb_version, spex_version, nei_version;

    struct MStrDB {
        uint32_t size;
        struct MStrEntry {
            String<constant::mstr_length> key;
            String<constant::mstr_length> value;
        } entries[constant::mstr_db_size];
    } mstr;  // model string database

    struct XFLTDB {
        uint64_t version;  // database version, incremented when modified
        uint32_t spec_num_count;  // count of spectra numbers having XFLT
        int spec_num[constant::xflt_db_size];  // spectra numbers having XFLT
        uint32_t size;
        struct XFLTEntry {
            int spec_num;
            String<constant::xflt_key_length> key;
            double value;
        } entries[constant::xflt_db_size];
    } xflt;  // XFLT database
};
STATIC_ASSERT_SHM_ELIGIBLE(XspecConfig);

template <typename T>
class XspecConfigWrapperBase
{
   public:
    // Use default constructor
    XspecConfigWrapperBase() : shmem_{} {};

    // Constructor
    explicit XspecConfigWrapperBase(T* shm) : shmem_{shm} {}

    // Use default destructor
    virtual ~XspecConfigWrapperBase() = default;

    // Delete copy constructor and copy assignment operator
    XspecConfigWrapperBase(const XspecConfigWrapperBase&) = delete;
    XspecConfigWrapperBase& operator=(const XspecConfigWrapperBase&) = delete;

    // Use default move constructor and move assignment operator
    XspecConfigWrapperBase(XspecConfigWrapperBase&&) = default;
    XspecConfigWrapperBase& operator=(XspecConfigWrapperBase&&) = default;

    // Synchronize XSPEC config to shared memory
    void sync_to_shmem()
    {
        do_sync_to_shmem();
        throw_if_mismatch("to shared memory");
    }

    // Synchronize config in shared memory to XSPEC
    void sync_from_shmem()
    {
        if (!match_to_shmem()) {
            do_sync_from_shmem();
            throw_if_mismatch("from shared memory");
        }
    }

    // Whether XSPEC config and config in shared memory are consistent
    [[nodiscard]] virtual bool match_to_shmem() const noexcept = 0;

    void shmem(T* shm) noexcept { shmem_ = shm; }

   protected:
    T* shmem_;

    virtual void do_sync_to_shmem() = 0;
    virtual void do_sync_from_shmem() = 0;

    // Get string representation of XSPEC config
    [[nodiscard]] virtual std::string xspec_config_string() const noexcept = 0;

    // Get string representation of config in shared memory
    [[nodiscard]] virtual std::string shmem_config_string() const noexcept = 0;

    // Get name of XSPEC config
    [[nodiscard]] virtual std::string name() const noexcept = 0;

   private:
    // Check consistency between XSPEC config and config in shared memory
    void throw_if_mismatch(const char* action) const
    {
        if (!match_to_shmem()) {
            std::ostringstream oss;
            oss << name() << " mismatch after sync XSPEC config " << action
                << std::endl
                << "    XSPEC: " << xspec_config_string() << std::endl
                << "    SHMEM: " << shmem_config_string() << std::endl;
            throw std::runtime_error(oss.str());
        }
    }
};

using XspecChatterWrapper = XspecConfigWrapperBase<XspecConfig::Chatter>;
class ChatterWrapper : public XspecChatterWrapper
{
   public:
    // Use base class constructor
    using XspecChatterWrapper::XspecChatterWrapper;

    [[nodiscard]] int level() const noexcept { return shmem_->level; }
    void level(const int level) const noexcept { shmem_->level = level; }

    [[nodiscard]] bool match_to_shmem() const noexcept override
    {
        return FunctionUtility::xwriteChatter() == shmem_->level;
    }

   protected:
    void do_sync_to_shmem() override
    {
        shmem_->level = FunctionUtility::xwriteChatter();
    }

    void do_sync_from_shmem() override
    {
        FunctionUtility::xwriteChatter(shmem_->level);
    }

    [[nodiscard]] std::string xspec_config_string() const noexcept override
    {
        return std::to_string(FunctionUtility::xwriteChatter());
    }

    [[nodiscard]] std::string shmem_config_string() const noexcept override
    {
        return std::to_string(shmem_->level);
    }

    [[nodiscard]] std::string name() const noexcept override
    {
        return "chatter";
    }
};

using XspecAbundWrapper = XspecConfigWrapperBase<XspecConfig::Abund>;
class AbundWrapper : public XspecAbundWrapper
{
   public:
    // Use base class constructor
    using XspecAbundWrapper::XspecAbundWrapper;

    [[nodiscard]] std::string table() const noexcept
    {
        return shmem_config_string();
    }

    void table(const std::string& table) const
    {
        utils::copy_string(utils::to_lower_case(table),
                           shmem_->table,
                           constant::table_length);
    }

    void file(const std::string& file) const
    {
        utils::copy_string("file", shmem_->table, constant::table_length);
        utils::copy_string(file, shmem_->file, constant::path_length);
    }

    [[nodiscard]] bool match_to_shmem() const noexcept override
    {
        if (FunctionUtility::ABUND() != "file") {
            return FunctionUtility::ABUND() == shmem_->table;
        } else {
            return FunctionUtility::ABUND() == shmem_->table &&
                   FunctionUtility::abundanceFile() == shmem_->file;
        }
    }

   protected:
    void do_sync_to_shmem() override
    {
        utils::copy_string(
            FunctionUtility::ABUND(), shmem_->table, constant::table_length);
        utils::copy_string(FunctionUtility::abundanceFile(),
                           shmem_->file,
                           constant::path_length);
    }

    void do_sync_from_shmem() override
    {
        int ierr{0};
        if (std::string(shmem_->table) != "file") {
            FPSOLR(shmem_->table, &ierr);
        } else {
            RFLABD(shmem_->file, &ierr);
        }
        if (ierr != 0) {
            std::ostringstream oss;
            if (std::string(shmem_->table) == "file") {
                oss << "invalid abundance file: \"" << shmem_->file << "\"";
            } else {
                oss << "invalid table: \"" << shmem_->table << "\"";
            }
            throw std::runtime_error(oss.str());
        }
    }

    [[nodiscard]] std::string xspec_config_string() const noexcept override
    {
        if (FunctionUtility::ABUND() != "file") {
            return FunctionUtility::ABUND();
        } else {
            return "file:" + FunctionUtility::abundanceFile();
        }
    }

    [[nodiscard]] std::string shmem_config_string() const noexcept override
    {
        if (std::string(shmem_->table) != "file") {
            return shmem_->table;
        } else {
            return "file:" + std::string(shmem_->file);
        }
    }

    [[nodiscard]] std::string name() const noexcept override
    {
        return "abund";
    }
};

using XspecXsectWrapper = XspecConfigWrapperBase<XspecConfig::Xsect>;
class XsectWrapper : public XspecXsectWrapper
{
   public:
    // Use base class constructor
    using XspecXsectWrapper::XspecXsectWrapper;

    [[nodiscard]] std::string table() const noexcept { return shmem_->table; }

    void table(const std::string& table) const
    {
        utils::copy_string(utils::to_lower_case(table),
                           shmem_->table,
                           constant::table_length);
    }

    [[nodiscard]] bool match_to_shmem() const noexcept override
    {
        return FunctionUtility::XSECT() == std::string(shmem_->table);
    }

   protected:
    void do_sync_to_shmem() override
    {
        utils::copy_string(
            FunctionUtility::XSECT(), shmem_->table, constant::table_length);
    }

    void do_sync_from_shmem() override
    {
        int ierr{0};
        FPXSCT(shmem_->table, &ierr);
        if (ierr != 0) {
            std::ostringstream oss;
            oss << "invalid table: \"" << shmem_->table << "\"";
            throw std::runtime_error(oss.str());
        }
    }

    [[nodiscard]] std::string xspec_config_string() const noexcept override
    {
        return FunctionUtility::XSECT();
    }

    [[nodiscard]] std::string shmem_config_string() const noexcept override
    {
        return shmem_->table;
    }

    [[nodiscard]] std::string name() const noexcept override
    {
        return "xsect";
    }
};

using XspecCosmoWrapper = XspecConfigWrapperBase<XspecConfig::Cosmo>;
class CosmoWrapper : public XspecCosmoWrapper
{
   public:
    // Use base class constructor
    using XspecCosmoWrapper::XspecCosmoWrapper;

    [[nodiscard]] float H0() const noexcept { return shmem_->H0; }
    [[nodiscard]] float q0() const noexcept { return shmem_->q0; }
    [[nodiscard]] float lambda0() const noexcept { return shmem_->lambda0; }
    void H0(const float H0) const noexcept { shmem_->H0 = H0; }
    void q0(const float q0) const noexcept { shmem_->q0 = q0; }
    void lambda0(const float lambda0) const noexcept
    {
        shmem_->lambda0 = lambda0;
    }

    [[nodiscard]] bool match_to_shmem() const noexcept override
    {
        return FunctionUtility::getH0() == shmem_->H0 &&
               FunctionUtility::getq0() == shmem_->q0 &&
               FunctionUtility::getlambda0() == shmem_->lambda0;
    }

   protected:
    void do_sync_to_shmem() override
    {
        shmem_->H0 = FunctionUtility::getH0();
        shmem_->q0 = FunctionUtility::getq0();
        shmem_->lambda0 = FunctionUtility::getlambda0();
    }

    void do_sync_from_shmem() override
    {
        FunctionUtility::setH0(shmem_->H0);
        FunctionUtility::setq0(shmem_->q0);
        FunctionUtility::setlambda0(shmem_->lambda0);
    }

    [[nodiscard]] std::string xspec_config_string() const noexcept override
    {
        std::ostringstream oss;
        oss << "H0=" << FunctionUtility::getH0()
            << " q0=" << FunctionUtility::getq0()
            << " lambda0=" << FunctionUtility::getlambda0();
        return oss.str();
    }

    [[nodiscard]] std::string shmem_config_string() const noexcept override
    {
        std::ostringstream oss;
        oss << "H0=" << shmem_->H0 << " q0=" << shmem_->q0
            << " lambda0=" << shmem_->lambda0;
        return oss.str();
    }

    [[nodiscard]] std::string name() const noexcept override
    {
        return "cosmo";
    }
};

using XsVersionWrapper = XspecConfigWrapperBase<XspecConfig::Version>;
class XspecVersionWrapper : public XsVersionWrapper
{
   public:
    // Use base class constructor
    using XsVersionWrapper::XsVersionWrapper;

    [[nodiscard]] std::string version() const noexcept
    {
        return shmem_->version;
    }

    [[nodiscard]] bool match_to_shmem() const noexcept override
    {
        return XSutility::xs_version() == shmem_->version;
    }

   protected:
    void do_sync_to_shmem() override
    {
        utils::copy_string(XSutility::xs_version(),
                           shmem_->version,
                           constant::version_length);
    }

    void do_sync_from_shmem() override
    {
        throw std::runtime_error("XSPEC version cannot be set");
    }

    [[nodiscard]] std::string xspec_config_string() const noexcept override
    {
        return XSutility::xs_version();
    }

    [[nodiscard]] std::string shmem_config_string() const noexcept override
    {
        return shmem_->version;
    }

    [[nodiscard]] std::string name() const noexcept override
    {
        return "XSPEC version";
    }
};

using XspecAtomDBVersionWrapper = XspecConfigWrapperBase<XspecConfig::Version>;
class AtomDBVersionWrapper : public XspecAtomDBVersionWrapper
{
   public:
    // Use base class constructor
    using XspecAtomDBVersionWrapper::XspecAtomDBVersionWrapper;

    [[nodiscard]] std::string version() const noexcept
    {
        return shmem_->version;
    }

    void version(const std::string& version) const
    {
        utils::copy_string(version, shmem_->version, constant::version_length);
    }

    [[nodiscard]] bool match_to_shmem() const noexcept override
    {
        return FunctionUtility::atomdbVersion() == shmem_->version;
    }

   protected:
    void do_sync_to_shmem() override
    {
        utils::copy_string(FunctionUtility::atomdbVersion(),
                           shmem_->version,
                           constant::version_length);
    }

    void do_sync_from_shmem() override
    {
        FunctionUtility::atomdbVersion(shmem_->version);
    }

    [[nodiscard]] std::string xspec_config_string() const noexcept override
    {
        return FunctionUtility::atomdbVersion();
    }

    [[nodiscard]] std::string shmem_config_string() const noexcept override
    {
        return shmem_->version;
    }

    [[nodiscard]] std::string name() const noexcept override
    {
        return "AtomDB version";
    }
};

using SPEXVersionWrapper_ = XspecConfigWrapperBase<XspecConfig::Version>;
class SPEXVersionWrapper : public SPEXVersionWrapper_
{
   public:
    // Use base class constructor
    using SPEXVersionWrapper_::SPEXVersionWrapper_;

    [[nodiscard]] std::string version() const noexcept
    {
        return shmem_->version;
    }

    void version(const std::string& version) const
    {
        utils::copy_string(version, shmem_->version, constant::version_length);
    }

    [[nodiscard]] bool match_to_shmem() const noexcept override
    {
        return FunctionUtility::spexVersion() == shmem_->version;
    }

   protected:
    void do_sync_to_shmem() override
    {
        utils::copy_string(FunctionUtility::spexVersion(),
                           shmem_->version,
                           constant::version_length);
    }

    void do_sync_from_shmem() override
    {
        FunctionUtility::spexVersion(shmem_->version);
    }

    [[nodiscard]] std::string xspec_config_string() const noexcept override
    {
        return FunctionUtility::spexVersion();
    }

    [[nodiscard]] std::string shmem_config_string() const noexcept override
    {
        return shmem_->version;
    }

    [[nodiscard]] std::string name() const noexcept override
    {
        return "SPEX version";
    }
};

using XspecNEIVersionWrapper = XspecConfigWrapperBase<XspecConfig::Version>;
class NEIVersionWrapper : public XspecNEIVersionWrapper
{
   public:
    // Use base class constructor
    using XspecNEIVersionWrapper::XspecNEIVersionWrapper;

    [[nodiscard]] std::string version() const noexcept
    {
        return shmem_->version;
    }

    void version(const std::string& version) const
    {
        utils::copy_string(version, shmem_->version, constant::version_length);
    }

    [[nodiscard]] bool match_to_shmem() const noexcept override
    {
        return FunctionUtility::neiVersion() == shmem_->version;
    }

   protected:
    void do_sync_to_shmem() override
    {
        utils::copy_string(FunctionUtility::neiVersion(),
                           shmem_->version,
                           constant::version_length);
    }

    void do_sync_from_shmem() override
    {
        FunctionUtility::neiVersion(shmem_->version);
    }

    [[nodiscard]] std::string xspec_config_string() const noexcept override
    {
        return FunctionUtility::neiVersion();
    }

    [[nodiscard]] std::string shmem_config_string() const noexcept override
    {
        return shmem_->version;
    }

    [[nodiscard]] std::string name() const noexcept override
    {
        return "NEI version";
    }
};

using MStrKey = std::string;
using MStrValue = std::string;
using MStrMap = std::map<MStrKey, MStrValue>;
using XspecMStrWrapper = XspecConfigWrapperBase<XspecConfig::MStrDB>;

class MStrWrapper : public XspecMStrWrapper
{
   public:
    // Use base class constructor
    using XspecMStrWrapper::XspecMStrWrapper;

    // Get single model string
    [[nodiscard]] MStrValue mstr(const MStrKey& key) const
    {
        for (uint32_t i = 0; i < shmem_->size; ++i) {
            if (shmem_->entries[i].key == key) {
                return shmem_->entries[i].value;
            }
        }
        throw std::runtime_error("model string not found: " + key);
    }

    // Get all model strings
    [[nodiscard]] MStrMap mstr() const noexcept
    {
        MStrMap result;
        for (uint32_t i = 0; i < shmem_->size; ++i) {
            result[shmem_->entries[i].key] = shmem_->entries[i].value;
        }
        return result;
    }

    // Set single model string
    void mstr(const MStrKey& key, const MStrValue& value) const
    {
        // Check if the key already exists
        for (uint32_t i = 0; i < shmem_->size; ++i) {
            if (shmem_->entries[i].key == key) {
                // Overwrite the value
                utils::copy_string(
                    value, shmem_->entries[i].value, constant::mstr_length);
                return;
            }
        }

        // Check if the database is full
        if (shmem_->size >= constant::mstr_db_size) {
            std::ostringstream oss;
            oss << "model string database is full (" << constant::mstr_db_size
                << "), "
                << "failed to add model string (" << key << " = " << value
                << ")";
            throw std::runtime_error(oss.str());
        }

        // Add the key and value to the database
        utils::copy_string(
            key, shmem_->entries[shmem_->size].key, constant::mstr_length);
        utils::copy_string(
            value, shmem_->entries[shmem_->size].value, constant::mstr_length);
        shmem_->size++;
    }

    // Set multiple model strings
    void mstr(const MStrMap& map) const
    {
        for (const auto& [key, value] : map) {
            mstr(key, value);
        }
    }

    // Clear model string database
    void clear() const noexcept
    {
        std::memset(shmem_->entries, 0, sizeof(shmem_->entries));
        shmem_->size = 0;
    }

    [[nodiscard]] bool match_to_shmem() const noexcept override
    {
        return FunctionUtility::modelStringDataBase() == mstr();
    }

   protected:
    void do_sync_to_shmem() override
    {
        clear();
        mstr(FunctionUtility::modelStringDataBase());
    }

    void do_sync_from_shmem() override
    {
        FunctionUtility::eraseModelStringDataBase();
        for (uint32_t i = 0; i < shmem_->size; ++i) {
            FunctionUtility::setModelString(shmem_->entries[i].key,
                                            shmem_->entries[i].value);
        }
    }

    [[nodiscard]] std::string xspec_config_string() const noexcept override
    {
        return utils::map_to_json(FunctionUtility::modelStringDataBase());
    }

    [[nodiscard]] std::string shmem_config_string() const noexcept override
    {
        return utils::map_to_json(mstr());
    }

    [[nodiscard]] std::string name() const noexcept override
    {
        return "model string";
    }
};

using SpecNum = int;
using XFLTKey = std::string;
using XFLTValue = double;
using XFLTValueStr = std::string;
using XFLTMap = std::map<XFLTKey, XFLTValue>;
using XFLTMapStr = std::map<XFLTKey, XFLTValueStr>;
using XFLTMaps = std::map<SpecNum, XFLTMap>;
using XFLTMapsStr = std::map<SpecNum, XFLTMapStr>;
using XspecXFLTWrapper = XspecConfigWrapperBase<XspecConfig::XFLTDB>;

class XFLTWrapper : public XspecXFLTWrapper
{
   public:
    // Use base class constructor
    using XspecXFLTWrapper::XspecXFLTWrapper;

    // Get XFLT entries (in double) for a given spectrum number
    [[nodiscard]] XFLTMap xflt(const SpecNum spec_num) const noexcept
    {
        // Check if the spectrum number is in the database
        bool found = false;
        for (uint32_t i = 0; i < shmem_->spec_num_count; ++i) {
            if (shmem_->spec_num[i] == spec_num) {
                found = true;
                break;
            }
        }
        if (!found) {
            return {};
        }

        // Get the XFLT entries for the spectrum number
        XFLTMap map;
        for (uint32_t i = 0; i < shmem_->size; ++i) {
            if (shmem_->entries[i].spec_num == spec_num) {
                map[shmem_->entries[i].key] = shmem_->entries[i].value;
            }
        }
        return map;
    }

    // Get XFLT entries (in string) for a given spectrum number
    [[nodiscard]] XFLTMapStr xflt_str(const SpecNum spec_num) const noexcept
    {
        XFLTMap map = xflt(spec_num);
        XFLTMapStr map_str;
        for (const auto& [key, value] : map) {
            map_str[key] = utils::double_to_string(value);
        }
        return map_str;
    }

    // Get all XFLT entries (in double)
    [[nodiscard]] XFLTMaps xflt() const noexcept
    {
        XFLTMaps maps;
        for (uint32_t i = 0; i < shmem_->size; ++i) {
            const auto& entry = shmem_->entries[i];
            maps[entry.spec_num][entry.key] = entry.value;
        }
        return maps;
    }

    // Get all XFLT entries (in string)
    [[nodiscard]] XFLTMapsStr xflt_str() const noexcept
    {
        XFLTMapsStr map;
        for (uint32_t i = 0; i < shmem_->size; ++i) {
            const auto& entry = shmem_->entries[i];
            map[entry.spec_num][entry.key] =
                utils::double_to_string(entry.value);
        }
        return map;
    }

    // Set XFLT entries for a given spectrum number
    void xflt(const SpecNum spec_num, const XFLTMap& map)
    {
        if (version_ != shmem_->version) {
            throw std::runtime_error("XFLT database is modified");
        }

        // Count the number of XFLT entries for the given spec_num
        uint32_t existing_count = 0;
        for (uint32_t i = 0; i < shmem_->size; ++i) {
            if (shmem_->entries[i].spec_num == spec_num) {
                ++existing_count;
            }
        }

        uint32_t insert_count = map.size();

        // Check if the database is full
        if (shmem_->size - existing_count + insert_count >
            constant::xflt_db_size) {
            std::ostringstream oss;
            oss << "XFLT database does not have enough space (" << shmem_->size
                << "/" << constant::xflt_db_size << ") to remove "
                << existing_count << " and add " << insert_count
                << " XFLT values for spectrum number " << spec_num;
            throw std::runtime_error(oss.str());
        }

        // Clear the entries for the given spec_num
        clear(spec_num);

        // Insert all new entries
        for (const auto& [key, val] : map) {
            auto& entry = shmem_->entries[shmem_->size];
            entry.spec_num = spec_num;
            utils::copy_string(key, entry.key, constant::xflt_key_length);
            entry.value = val;
            shmem_->size++;
        }

        // Update spec_num and spec_num_count
        if (existing_count == 0) {
            shmem_->spec_num[shmem_->spec_num_count] = spec_num;
            shmem_->spec_num_count++;
        }

        version_++;
        shmem_->version = version_;
    }

    // Set multiple XFLT entries for multiple spectra
    void xflt(const XFLTMaps& maps)
    {
        for (const auto& [spec_num, map] : maps) {
            xflt(spec_num, map);
        }
    }

    // Clear XFLT entries for a given spectrum number
    void clear(const SpecNum spec_num)
    {
        if (version_ != shmem_->version) {
            throw std::runtime_error("XFLT database is modified");
        }

        // Find the first matching index and counts
        uint32_t first_del_idx = shmem_->size;
        uint32_t del_count = 0;
        for (uint32_t i = 0; i < shmem_->size; ++i) {
            if (shmem_->entries[i].spec_num == spec_num) {
                if (first_del_idx == shmem_->size) {
                    first_del_idx = i;
                }
                ++del_count;
            }
        }

        if (del_count == 0) {
            return;
        }

        using XFLTEntry = XspecConfig::XFLTDB::XFLTEntry;

        // Address of the subsequent entries
        XFLTEntry* src = &shmem_->entries[first_del_idx + del_count];
        // Destination to move the subsequent entries to
        XFLTEntry* dest = &shmem_->entries[first_del_idx];
        // Count of entries to move
        size_t move_count = shmem_->size - (first_del_idx + del_count);

        if (move_count > 0) {
            // Move the subsequent entries to the destination
            std::memmove(dest, src, move_count * sizeof(XFLTEntry));
        }

        // Clear the trailing extra space
        std::memset(&shmem_->entries[shmem_->size - del_count],
                    0,
                    del_count * sizeof(XFLTEntry));
        shmem_->size -= del_count;

        // Update spec_num and spec_num_count
        for (uint32_t i = 0; i < shmem_->spec_num_count; ++i) {
            if (shmem_->spec_num[i] == spec_num) {
                // Replace the spec_num with the last spec_num
                shmem_->spec_num[i] =
                    shmem_->spec_num[--shmem_->spec_num_count];

                // Clear the last spec_num
                shmem_->spec_num[shmem_->spec_num_count] = 0;
                break;
            }
        }

        version_++;
        shmem_->version = version_;
    }

    // Clear all XFLT entries
    void clear()
    {
        if (version_ != shmem_->version) {
            throw std::runtime_error("XFLT database is modified");
        }

        shmem_->spec_num_count = 0;
        shmem_->size = 0;
        std::memset(shmem_->spec_num, 0, sizeof(shmem_->spec_num));
        std::memset(shmem_->entries, 0, sizeof(shmem_->entries));

        version_++;
        shmem_->version = version_;
    }

    [[nodiscard]] bool match_to_shmem() const noexcept override
    {
        if (version_ != shmem_->version) {
            return false;
        }

        for (uint32_t i = 0; i < shmem_->spec_num_count; ++i) {
            const int spec_num = shmem_->spec_num[i];
            if (FunctionUtility::getNumberXFLT(spec_num) == 0) {
                return false;
            }
            if (FunctionUtility::getAllXFLTstr(spec_num) !=
                xflt_str(spec_num)) {
                return false;
            }
        }
        return true;
    }

   protected:
    void do_sync_to_shmem() override
    {
        throw std::runtime_error("cannot sync XFLT to shared memory");
    }

    void do_sync_from_shmem() override
    {
        FunctionUtility::clearXFLT();
        for (uint32_t i = 0; i < shmem_->spec_num_count; ++i) {
            const int spec_num = shmem_->spec_num[i];
            FunctionUtility::loadXFLT(spec_num, xflt_str(spec_num));
        }
        version_ = shmem_->version;
    }

    [[nodiscard]] std::string xspec_config_string() const noexcept override
    {
        std::map<SpecNum, XFLTMapStr> map;
        for (uint32_t i = 0; i < shmem_->spec_num_count; ++i) {
            const int spec_num = shmem_->spec_num[i];
            map[spec_num] = xflt_str(spec_num);
        }
        return utils::map_to_json(map);
    }

    [[nodiscard]] std::string shmem_config_string() const noexcept override
    {
        return utils::map_to_json(xflt_str());
    }

    [[nodiscard]] std::string name() const noexcept override { return "XFLT"; }

   private:
    uint64_t version_{0};
};

class XspecConfigManager
{
   private:
    bool is_owner_;
    shmem::SharedMemory<XspecConfig> shmem_;

    void update_wrappers()
    {
        auto* config = shmem_.ptr();
        chatter.shmem(&config->chatter);
        abund.shmem(&config->abund);
        xsect.shmem(&config->xsect);
        cosmo.shmem(&config->cosmo);
        xspec_version.shmem(&config->xspec_version);
        atomdb_version.shmem(&config->atomdb_version);
        spex_version.shmem(&config->spex_version);
        nei_version.shmem(&config->nei_version);
        mstr.shmem(&config->mstr);
        xflt.shmem(&config->xflt);
    }

   public:
    ChatterWrapper chatter;
    AbundWrapper abund;
    XsectWrapper xsect;
    CosmoWrapper cosmo;
    XspecVersionWrapper xspec_version;
    AtomDBVersionWrapper atomdb_version;
    SPEXVersionWrapper spex_version;
    NEIVersionWrapper nei_version;
    MStrWrapper mstr;
    XFLTWrapper xflt;

    // Constructor
    XspecConfigManager(pid_t parent_pid, bool create)
        : is_owner_{create},
          shmem_{constant::shm_name_of_xspec_config(parent_pid),
                 1,
                 create,
                 false}
    {
        update_wrappers();
    }

    // Delete default constructor
    XspecConfigManager() = delete;

    // Use default destructor
    ~XspecConfigManager() = default;

    // Delete copy constructor and copy assignment operator
    XspecConfigManager(const XspecConfigManager&) = delete;
    XspecConfigManager& operator=(const XspecConfigManager&) = delete;

    // Move constructor
    XspecConfigManager(XspecConfigManager&& other) noexcept
        : is_owner_{other.is_owner_}, shmem_{std::move(other.shmem_)}
    {
        other.is_owner_ = false;
        update_wrappers();
    }

    // Move assignment operator
    XspecConfigManager& operator=(XspecConfigManager&& other) noexcept
    {
        if (this != &other) {
            // Move resources from other
            is_owner_ = other.is_owner_;
            other.is_owner_ = false;
            shmem_ = std::move(other.shmem_);

            // Update all wrappers
            update_wrappers();
        }
        return *this;
    }

    void sync_to_shmem()
    {
        chatter.sync_to_shmem();
        abund.sync_to_shmem();
        xsect.sync_to_shmem();
        cosmo.sync_to_shmem();
        xspec_version.sync_to_shmem();
        atomdb_version.sync_to_shmem();
        spex_version.sync_to_shmem();
        nei_version.sync_to_shmem();
        mstr.sync_to_shmem();
        // xflt.sync_to_shmem();
    }

    void sync_from_shmem()
    {
        chatter.sync_from_shmem();
        abund.sync_from_shmem();
        xsect.sync_from_shmem();
        cosmo.sync_from_shmem();
        // xspec_version.sync_from_shmem();
        atomdb_version.sync_from_shmem();
        spex_version.sync_from_shmem();
        nei_version.sync_from_shmem();
        mstr.sync_from_shmem();
        xflt.sync_from_shmem();
    }

    [[nodiscard]] XspecConfig config() const noexcept { return *shmem_.ptr(); }

    void restore(const XspecConfig& config) const noexcept
    {
        *shmem_.ptr() = config;
    }
};
}  // namespace xspex::config::xspec

namespace xspex::config::worker
{
enum class Task : uint32_t {
    EvaluateModel,
    ResizeBuffer,

    // XSPEC configuration
    InitializeXSPEC,
    SyncConfigToShmem,
    SyncConfigFromShmem,
    SyncChatterFromShmem,
    SyncAbundFromShmem,
    SyncXsectFromShmem,
    SyncCosmoFromShmem,
    SyncModelStringFromShmem,
    SyncXFLTFromShmem,
    SyncAtomDBVersionFromShmem,
    SyncSPEXVersionFromShmem,
    SyncNEIVersionFromShmem,
};

struct WorkerConfig {
    using String = char[constant::config_content_length];

    // Worker status
    bool running;

    // Task configuration
    Task task;
    bool success;
    String message;

    // Buffer configuration
    uint32_t buf_size;

    // Model task configuration
    uint32_t func_id;
    uint32_t n_params;
    int n_out;
    int spec_num;
    String init_string;

    // Task synchronization primitives
    struct MutexCond {
        bool flag;
        pthread_mutex_t mutex;
        pthread_cond_t cond;
    } task_start, task_end;
};
STATIC_ASSERT_SHM_ELIGIBLE(WorkerConfig);

class MutexCondWrapper
{
   public:
    using MutexCond = WorkerConfig::MutexCond;

    // Constructor
    MutexCondWrapper(MutexCond* shmem, bool is_owner)
        : shmem_{shmem}, is_owner_{is_owner}
    {
        if (is_owner) {
            init();
        }
    }

    // Delete default constructor
    MutexCondWrapper() = delete;

    // Destructor
    ~MutexCondWrapper()
    {
        if (is_owner_) {
            destroy();
        }
    }

    // Delete copy constructor and copy assignment operator
    MutexCondWrapper(const MutexCondWrapper&) = delete;
    MutexCondWrapper& operator=(const MutexCondWrapper&) = delete;

    // Move constructor
    MutexCondWrapper(MutexCondWrapper&& other) noexcept
        : shmem_{other.shmem_}, is_owner_{other.is_owner_}
    {
        other.is_owner_ = false;
    }

    // Move assignment operator
    MutexCondWrapper& operator=(MutexCondWrapper&& other) noexcept
    {
        if (this != &other) {
            if (is_owner_) {
                destroy();
            }
            is_owner_ = other.is_owner_;
            shmem_ = other.shmem_;
            other.is_owner_ = false;
        }
        return *this;
    }

    void wait() const
    {
        pthread_mutex_lock(&shmem_->mutex);
        while (!shmem_->flag) {
            pthread_cond_wait(&shmem_->cond, &shmem_->mutex);
        }
        shmem_->flag = false;
        pthread_mutex_unlock(&shmem_->mutex);
    }

    void notify() const
    {
        pthread_mutex_lock(&shmem_->mutex);
        shmem_->flag = true;
        pthread_cond_signal(&shmem_->cond);
        pthread_mutex_unlock(&shmem_->mutex);
    }

   private:
    MutexCond* shmem_;
    bool is_owner_;

    void init() const
    {
        std::ostringstream oss;

        shmem_->flag = false;

        // Initialize mutex
        auto mutex_attr = std::unique_ptr<pthread_mutexattr_t,
                                          int (*)(pthread_mutexattr_t*)>(
            new pthread_mutexattr_t(), pthread_mutexattr_destroy);

        if (pthread_mutexattr_init(mutex_attr.get())) {
            oss << "pthread_mutexattr_init failed: " << strerror(errno);
            throw std::runtime_error(oss.str());
        }

        if (pthread_mutexattr_setpshared(mutex_attr.get(),
                                         PTHREAD_PROCESS_SHARED)) {
            oss << "pthread_mutexattr_setpshared failed: " << strerror(errno);
            throw std::runtime_error(oss.str());
        }
        if (pthread_mutex_init(&shmem_->mutex, mutex_attr.get())) {
            oss << "pthread_mutex_init failed: " << strerror(errno);
            throw std::runtime_error(oss.str());
        }

        // Initialize cond
        auto cond_attr =
            std::unique_ptr<pthread_condattr_t, int (*)(pthread_condattr_t*)>(
                new pthread_condattr_t(), pthread_condattr_destroy);

        if (pthread_condattr_init(cond_attr.get())) {
            pthread_mutex_destroy(&shmem_->mutex);  // Clean up mutex
            oss << "pthread_condattr_init failed: " << strerror(errno);
            throw std::runtime_error(oss.str());
        }

        if (pthread_condattr_setpshared(cond_attr.get(),
                                        PTHREAD_PROCESS_SHARED)) {
            pthread_mutex_destroy(&shmem_->mutex);  // Clean up mutex
            oss << "pthread_condattr_setpshared failed: " << strerror(errno);
            throw std::runtime_error(oss.str());
        }
        if (pthread_cond_init(&shmem_->cond, cond_attr.get())) {
            pthread_mutex_destroy(&shmem_->mutex);  // Clean up mutex
            oss << "pthread_cond_init failed: " << strerror(errno);
            throw std::runtime_error(oss.str());
        }
    }

    void destroy() const noexcept
    {
        pthread_mutex_destroy(&shmem_->mutex);
        pthread_cond_destroy(&shmem_->cond);
    }
};

class WorkerShmemManager
{
   public:
    // Constructor
    WorkerShmemManager(pid_t parent_pid, int32_t device_id, bool create)
        : config_shmem_{constant::shm_name_of_worker_config(parent_pid,
                                                            device_id),
                        1,
                        create,
                        true},
          buffer_shmem_{
              constant::shm_name_of_worker_buffer(parent_pid, device_id),
              constant::shared_memory_size_init,
              create,
              true},
          task_start_{&config_shmem_.ptr()->task_start, create},
          task_end_{&config_shmem_.ptr()->task_end, create}
    {
    }

    // Delete default constructor
    WorkerShmemManager() = delete;

    // Use default destructor
    ~WorkerShmemManager() = default;

    // Delete copy constructor and copy assignment operator
    WorkerShmemManager(const WorkerShmemManager&) = delete;
    WorkerShmemManager& operator=(const WorkerShmemManager&) = delete;

    // Move constructor
    WorkerShmemManager(WorkerShmemManager&& other) noexcept
        : config_shmem_{std::move(other.config_shmem_)},
          buffer_shmem_{std::move(other.buffer_shmem_)},
          task_start_{std::move(other.task_start_)},
          task_end_{std::move(other.task_end_)}
    {
    }

    // Move assignment operator
    WorkerShmemManager& operator=(WorkerShmemManager&& other) noexcept
    {
        if (this != &other) {
            config_shmem_ = std::move(other.config_shmem_);
            buffer_shmem_ = std::move(other.buffer_shmem_);
            task_start_ = std::move(other.task_start_);
            task_end_ = std::move(other.task_end_);
        }
        return *this;
    }

    // >>> Interfaces for task management >>>

    // Running flag
    [[nodiscard]] bool running() const noexcept
    {
        return config_shmem_.ptr()->running;
    }

    void running(const bool running) const noexcept
    {
        config_shmem_.ptr()->running = running;
    }

    // Task
    [[nodiscard]] Task task() const noexcept
    {
        return config_shmem_.ptr()->task;
    }

    // Submit task
    void task(const Task task) const noexcept
    {
        auto* config = config_shmem_.ptr();
        config->task = task;
        config->success = false;
    }

    // Submit model task, return the model buffer pointer
    // Buffer size should be checked to be large enough before submitting
    double* model_task(const uint32_t func_id,
                       const double* params,
                       const uint32_t n_params,
                       const double* egrid,
                       const uint32_t n_out,
                       const int spec_num,
                       const std::string& init_string,
                       const double* input_model = nullptr) const
    {
        auto* config = config_shmem_.ptr();
        config->func_id = func_id;
        config->n_params = n_params;
        config->n_out = static_cast<int>(n_out);
        config->spec_num = spec_num;
        utils::copy_string(init_string,
                           config->init_string,
                           constant::config_content_length,
                           true);

        auto* buffer = buffer_shmem_.ptr();
        const auto n_egrid = n_out + 1;
        auto* params_buf = buffer;
        auto* egrid_buf = params_buf + n_params;
        auto* model_buf = egrid_buf + n_egrid;
        memcpy(params_buf, params, n_params * sizeof(double));
        memcpy(egrid_buf, egrid, n_egrid * sizeof(double));
        if (input_model != nullptr) {
            memcpy(model_buf, input_model, n_out * sizeof(double));
        }

        task(Task::EvaluateModel);

        return model_buf;
    }

    // Get the model task configuration
    [[nodiscard]] auto model_task() const noexcept
    {
        auto* config = config_shmem_.ptr();
        auto* buffer = buffer_shmem_.ptr();

        const auto func_id = config->func_id;
        const auto n_params = config->n_params;
        const auto n_flux = config->n_out;
        const auto spec_num = config->spec_num;
        const auto init_string = config->init_string;
        const auto params = buffer;
        const auto egrid = buffer + n_params;
        const auto model = egrid + n_flux + 1;
        const auto model_error = model + n_flux;
        return std::make_tuple(func_id,
                               egrid,
                               n_flux,
                               params,
                               spec_num,
                               model,
                               model_error,
                               init_string);
    }

    // Task status flag
    [[nodiscard]] bool success() const noexcept
    {
        return config_shmem_.ptr()->success;
    }

    void success(const bool success) const noexcept
    {
        config_shmem_.ptr()->success = success;
    }

    // Message when task fails
    [[nodiscard]] std::string message() const noexcept
    {
        return config_shmem_.ptr()->message;
    }

    void message(const std::string& content) const
    {
        utils::copy_string(content,
                           config_shmem_.ptr()->message,
                           constant::config_content_length);
    }

    // Task synchronization primitive

    void wait_for_task_start() const noexcept { task_start_.wait(); }

    void notify_task_start() const noexcept { task_start_.notify(); }

    void wait_for_task_end() const noexcept { task_end_.wait(); }

    void notify_task_end() const noexcept { task_end_.notify(); }

    // <<< Interfaces for task management <<<

    // >>> Interfaces for buffer management >>>

    // Resize the buffer to the requested size, intended to be called by the
    // main process. Return true if the buffer is resized, false otherwise.
    [[nodiscard]] bool resize_buffer(const uint32_t requested_size)
    {
        if (requested_size <= buffer_shmem_.size()) {
            return false;
        }

        // Round up to the next power of 2
        auto new_buffer_shmem = shmem::SharedMemory<double>(
            buffer_shmem_.name(),
            utils::next_power_of_two(requested_size),
            buffer_shmem_.is_owner(),
            true);
        buffer_shmem_ = std::move(new_buffer_shmem);

        if (config_shmem_.is_owner()) {
            config_shmem_.ptr()->buf_size = buffer_shmem_.size();
        }
        return true;
    }

    // Resize the buffer to the size specified in the config, intended to be
    // the worker process
    void resize_buffer()
    {
        const auto _ = resize_buffer(config_shmem_.ptr()->buf_size);
    }

    // <<< Interfaces for buffer management <<<

   private:
    shmem::SharedMemory<WorkerConfig> config_shmem_;
    shmem::SharedMemory<double> buffer_shmem_;
    MutexCondWrapper task_start_;
    MutexCondWrapper task_end_;
};
}  // namespace xspex::config::worker

#endif  // XSPEX_CONFIG_HPP_
