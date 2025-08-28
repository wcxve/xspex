#ifndef XSPEX_SHMEM_HPP_
#define XSPEX_SHMEM_HPP_

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace xspex::shmem
{
template <typename T>
class SharedMemory
{
   public:
    // Delete default constructor
    SharedMemory() = delete;

    explicit SharedMemory(const std::string& name,
                          const size_t size,
                          const bool create,
                          const bool unlink_after_open)
        : shm_name_{name}, ptr_{nullptr}, size_{size}, is_owner_{create}
    {
        int shm_fd;
        if (create) {
            // Remove the named shared memory if it already exists
            shm_unlink(name.c_str());

            // Create the named shared memory =
            shm_fd = shm_open(name.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
            if (shm_fd == -1) {
                std::ostringstream oss;
                oss << "failed to create shared memory (" << name
                    << "): " << strerror(errno);
                throw std::runtime_error(oss.str());
            }

            if (ftruncate(shm_fd, static_cast<off_t>(size_in_bytes())) == -1) {
                close(shm_fd);
                shm_unlink(name.c_str());
                std::ostringstream oss;
                oss << "failed to allocate size=" << size_in_bytes()
                    << " for shared memory (" << name
                    << "): " << strerror(errno);
                throw std::runtime_error(oss.str());
            }
        } else {
            shm_fd = shm_open(name.c_str(), O_RDWR, 0600);
            if (shm_fd == -1) {
                std::ostringstream oss;
                oss << "failed to open shared memory (" << name
                    << "): " << strerror(errno);
                throw std::runtime_error(oss.str());
            }
            if (unlink_after_open) {
                shm_unlink(name.c_str());
            }
        }

        // Map the memory
        void* addr = mmap(nullptr,
                          size_in_bytes(),
                          PROT_READ | PROT_WRITE,
                          MAP_SHARED,
                          shm_fd,
                          0);
        if (addr == MAP_FAILED) {
            close(shm_fd);
            if (is_owner_) {
                shm_unlink(name.c_str());
            }
            std::ostringstream oss;
            oss << "failed to map shared memory (" << name
                << "): " << strerror(errno);
            throw std::runtime_error(oss.str());
        }

        ptr_ = static_cast<T*>(addr);

        close(shm_fd);

        if (is_owner_) {
            static_assert(
                std::is_trivially_destructible_v<T>,
                "T must be trivially destructible when placed in shmem");
            if (size_ == 1) {
                // Only value-initialize a single object payload (e.g.,
                // structs).
                new (ptr_) T();
            } else {
                // For POD buffers, zero-initialize for deterministic reads.
                std::memset(ptr_, 0, size_in_bytes());
            }
        }
    }

    ~SharedMemory()
    {
        cleanup();
        if (is_owner_) {
            shm_unlink(shm_name_.c_str());
        }
    }

    // Delete copy constructor and copy assignment operator
    SharedMemory(const SharedMemory&) = delete;
    SharedMemory& operator=(const SharedMemory&) = delete;

    // Delete move constructor and move assignment operator
    SharedMemory(SharedMemory&& other) = delete;
    SharedMemory& operator=(SharedMemory&& other) = delete;

    [[nodiscard]] std::string name() const { return shm_name_; }

    [[nodiscard]] bool is_owner() const { return is_owner_; }

    // Get the pointer to the mapped memory
    [[nodiscard]] T* ptr() const { return ptr_; }

    // Get the size
    [[nodiscard]] size_t size() const { return size_; }

    // Get the size in bytes
    [[nodiscard]] size_t size_in_bytes() const { return size_ * sizeof(T); }

   private:
    std::string shm_name_;
    T* ptr_;
    size_t size_;
    bool is_owner_;

    void cleanup() noexcept
    {
        if (ptr_) {
            munmap(ptr_, size_in_bytes());
            ptr_ = nullptr;
            size_ = 0;
        }
    }
};
}  // namespace xspex::shmem

#endif  // XSPEX_SHMEM_HPP_
